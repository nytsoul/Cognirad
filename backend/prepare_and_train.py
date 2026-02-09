"""
CogniRad++ Complete Training Pipeline
Automatically downloads IU-Xray dataset and trains the model
"""

import os
import sys
import json
import zipfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from PIL import Image
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import random
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration"""
    # Data
    data_dir: str = "./data/iuxray"
    output_dir: str = "./outputs"
    
    # Model
    image_size: int = 224
    hidden_dim: int = 512
    num_classes: int = 14  # CheXpert labels
    vocab_size: int = 50257  # GPT-2 vocab
    max_seq_len: int = 128
    
    # Training
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_accumulation: int = 4
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    num_workers: int = 0  # Windows compatible
    
    # Checkpointing
    save_every: int = 5
    eval_every: int = 1


# =============================================================================
# Dataset Download
# =============================================================================

class DownloadProgress(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_iuxray_dataset(data_dir: str) -> bool:
    """
    Download and extract IU-Xray dataset from OpenI
    
    Returns:
        bool: True if download successful or data exists
    """
    data_path = Path(data_dir)
    images_path = data_path / "images"
    reports_path = data_path / "reports"
    
    # Check if already exists
    if images_path.exists() and len(list(images_path.glob("*.png"))) > 100:
        print(f"✓ IU-Xray dataset already exists at {data_path}")
        return True
    
    data_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Downloading IU-Xray Dataset")
    print("=" * 70)
    
    # URLs for IU-Xray from OpenI (NIH NLM)
    urls = {
        "images": "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz",
        "reports": "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
    }
    
    try:
        for name, url in urls.items():
            print(f"\n📥 Downloading {name}...")
            filename = data_path / f"{name}.tgz"
            
            with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc=name) as t:
                urllib.request.urlretrieve(url, filename, reporthook=t.update_to)
            
            print(f"   Extracting {name}...")
            import tarfile
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall(data_path)
            
            # Clean up archive
            filename.unlink()
        
        print(f"\n✓ Dataset downloaded and extracted to {data_path}")
        return True
        
    except Exception as e:
        print(f"\n⚠️  Download failed: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://openi.nlm.nih.gov/faq#collection")
        print("2. Download NLMCXR_png.tgz and NLMCXR_reports.tgz")
        print(f"3. Extract to: {data_path}")
        print("\nAlternatively, use demo mode: --demo")
        return False


# =============================================================================
# IU-Xray Dataset
# =============================================================================

class IUXrayDataset(Dataset):
    """IU-Xray Dataset for CogniRad++ training"""
    
    # CheXpert label mapping
    CHEXPERT_LABELS = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices"
    ]
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 224,
        max_seq_len: int = 128
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        
        # Load tokenizer
        try:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except:
            self.tokenizer = None
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load samples
        self.samples = self._load_samples()
        
        # Split data
        random.seed(42)
        random.shuffle(self.samples)
        n = len(self.samples)
        if split == "train":
            self.samples = self.samples[:int(0.8 * n)]
        elif split == "val":
            self.samples = self.samples[int(0.8 * n):int(0.9 * n)]
        else:  # test
            self.samples = self.samples[int(0.9 * n):]
        
        print(f"  {split}: {len(self.samples)} samples")
    
    def _load_samples(self) -> List[Dict]:
        """Load and parse IU-Xray XML reports"""
        samples = []
        
        # Find images and reports
        images_dir = self.data_dir / "NLMCXR_png"
        if not images_dir.exists():
            images_dir = self.data_dir / "images"
        
        reports_dir = self.data_dir / "NLMCXR_reports" / "ecgen-radiology"
        if not reports_dir.exists():
            reports_dir = self.data_dir / "reports" / "ecgen-radiology"
        if not reports_dir.exists():
            reports_dir = self.data_dir / "ecgen-radiology"
        
        # Parse XML reports
        xml_files = list(reports_dir.glob("*.xml")) if reports_dir.exists() else []
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Get report text
                findings = ""
                impression = ""
                for elem in root.iter():
                    if elem.tag == "AbstractText":
                        label = elem.get("Label", "")
                        if label == "FINDINGS":
                            findings = elem.text or ""
                        elif label == "IMPRESSION":
                            impression = elem.text or ""
                
                report = f"{findings} {impression}".strip()
                if not report or len(report) < 20:
                    continue
                
                # Get associated image
                image_files = []
                for parent_elem in root.iter("parentImage"):
                    img_id = parent_elem.get("id", "")
                    if img_id:
                        # Try different image path patterns
                        for pattern in [f"{img_id}.png", f"CXR{img_id}.png"]:
                            img_path = images_dir / pattern
                            if img_path.exists():
                                image_files.append(img_path)
                                break
                
                # Also search by report ID
                report_id = xml_file.stem
                for img_path in images_dir.glob(f"*{report_id}*.png"):
                    if img_path not in image_files:
                        image_files.append(img_path)
                
                if image_files:
                    samples.append({
                        "image_path": str(image_files[0]),
                        "report": report,
                        "report_id": report_id
                    })
                    
            except Exception as e:
                continue
        
        # If no structured data found, try simple pairing
        if len(samples) < 10 and images_dir.exists():
            print("  Using simple image-report pairing...")
            for img_path in images_dir.glob("*.png"):
                samples.append({
                    "image_path": str(img_path),
                    "report": "Normal chest X-ray. No acute cardiopulmonary abnormality.",
                    "report_id": img_path.stem
                })
                if len(samples) >= 1000:
                    break
        
        return samples
    
    def _extract_labels(self, report: str) -> torch.Tensor:
        """Extract CheXpert-style labels from report text"""
        labels = torch.zeros(14)
        report_lower = report.lower()
        
        # Simple keyword matching for demo
        keywords = {
            0: ["normal", "no acute", "unremarkable", "clear"],  # No Finding
            2: ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
            3: ["opacity", "opacities", "infiltrate"],
            4: ["lesion", "mass", "nodule"],
            5: ["edema", "pulmonary edema", "vascular congestion"],
            6: ["consolidation"],
            7: ["pneumonia"],
            8: ["atelectasis", "collapse"],
            9: ["pneumothorax"],
            10: ["effusion", "pleural effusion"],
            12: ["fracture"],
            13: ["support", "device", "pacemaker", "tube", "catheter"]
        }
        
        for idx, kws in keywords.items():
            for kw in kws:
                if kw in report_lower:
                    labels[idx] = 1.0
                    break
        
        # If nothing found, assume no finding
        if labels.sum() == 0:
            labels[0] = 1.0
        
        return labels
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load and transform image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            image = self.transform(image)
        except:
            # Return random tensor if image loading fails
            image = torch.randn(3, self.image_size, self.image_size)
        
        # Get report
        report = sample["report"]
        
        # Tokenize report
        if self.tokenizer:
            tokens = self.tokenizer(
                report,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
            attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)
        
        # Extract labels
        labels = self._extract_labels(report)
        
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "report": report
        }


# =============================================================================
# Demo/Synthetic Dataset
# =============================================================================

class DemoDataset(Dataset):
    """Synthetic dataset for testing without real data"""
    
    DEMO_REPORTS = [
        "The heart size is normal. The lungs are clear. No pleural effusion or pneumothorax. No acute cardiopulmonary abnormality.",
        "Mild cardiomegaly. Pulmonary vasculature is within normal limits. No focal consolidation, pleural effusion, or pneumothorax.",
        "Low lung volumes. Bibasilar atelectasis. No focal consolidation. Small bilateral pleural effusions.",
        "Right lower lobe opacity concerning for pneumonia. Recommend clinical correlation.",
        "Status post median sternotomy. Cardiomegaly. Mild pulmonary edema. Bilateral pleural effusions.",
        "Normal chest radiograph. No acute findings.",
        "Left-sided pleural effusion with associated atelectasis. No pneumothorax.",
        "Stable appearance of pacemaker. No acute cardiopulmonary process.",
        "Diffuse bilateral pulmonary opacities. Consider pulmonary edema versus infection.",
        "No acute intrathoracic abnormality. Normal heart size and pulmonary vasculature."
    ]
    
    def __init__(self, size: int = 500, image_size: int = 224, max_seq_len: int = 128):
        self.size = size
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        
        try:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except:
            self.tokenizer = None
        
        print(f"  Demo mode: {size} synthetic samples")
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict:
        # Generate consistent random image based on idx
        torch.manual_seed(idx)
        image = torch.randn(3, self.image_size, self.image_size) * 0.1 + 0.5
        image = torch.clamp(image, 0, 1)
        
        # Select report
        report = self.DEMO_REPORTS[idx % len(self.DEMO_REPORTS)]
        
        # Tokenize
        if self.tokenizer:
            tokens = self.tokenizer(
                report,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
            attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)
        
        # Simple labels
        labels = torch.zeros(14)
        if "normal" in report.lower() or "no acute" in report.lower():
            labels[0] = 1.0
        if "cardiomegaly" in report.lower():
            labels[2] = 1.0
        if "effusion" in report.lower():
            labels[10] = 1.0
        if "opacity" in report.lower() or "pneumonia" in report.lower():
            labels[3] = 1.0
        if "atelectasis" in report.lower():
            labels[8] = 1.0
        if "edema" in report.lower():
            labels[5] = 1.0
        
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "report": report
        }


# =============================================================================
# Simplified CogniRad++ Model for Training
# =============================================================================

class CogniRadLite(nn.Module):
    """
    Lightweight CogniRad++ for training demonstration
    Uses simpler architecture that works without full dependencies
    """
    
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        
        # Visual Encoder (simplified PRO-FA)
        try:
            import timm
            self.backbone = timm.create_model(
                'resnet50',
                pretrained=True,
                features_only=True,
                out_indices=[2, 3, 4]
            )
            self.visual_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(2048, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU()
            )
        except:
            # Fallback to simple CNN
            self.backbone = None
            self.visual_encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, 2, 1),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.GELU()
            )
        
        # Disease Classifier (simplified MIX-MLP)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
        
        # Report Decoder (simplified RCTA)
        try:
            from transformers import GPT2LMHeadModel, GPT2Config
            gpt_config = GPT2Config(
                vocab_size=config.vocab_size,
                n_embd=config.hidden_dim,
                n_layer=4,
                n_head=8
            )
            self.decoder = GPT2LMHeadModel(gpt_config)
        except:
            self.decoder = None
        
        # Visual-to-text projection
        self.visual_to_decoder = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Disease embedding for decoder conditioning
        self.disease_embed = nn.Linear(config.num_classes, config.hidden_dim)
    
    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to visual features"""
        if self.backbone is not None:
            features = self.backbone(images)
            return self.visual_proj(features[-1])
        else:
            return self.visual_encoder(images)
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training
        
        Returns:
            Dict with loss, cls_logits, lm_logits
        """
        batch_size = images.size(0)
        device = images.device
        
        # 1. Visual encoding
        visual_features = self.encode_visual(images)  # [B, hidden_dim]
        
        # 2. Disease classification
        cls_logits = self.classifier(visual_features)  # [B, num_classes]
        
        # 3. Report generation (if decoder available and input_ids provided)
        lm_loss = torch.tensor(0.0, device=device)
        lm_logits = None
        
        if self.decoder is not None and input_ids is not None:
            # Create prefix from visual + disease features
            disease_probs = torch.sigmoid(cls_logits)
            disease_features = self.disease_embed(disease_probs)
            
            combined = visual_features + disease_features
            prefix = self.visual_to_decoder(combined).unsqueeze(1)  # [B, 1, hidden_dim]
            
            # Get token embeddings
            token_embeds = self.decoder.transformer.wte(input_ids)  # [B, seq_len, hidden_dim]
            
            # Concatenate prefix with token embeddings
            inputs_embeds = torch.cat([prefix, token_embeds[:, :-1, :]], dim=1)
            
            # Forward through decoder
            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=input_ids
            )
            lm_loss = outputs.loss
            lm_logits = outputs.logits
        
        # 4. Compute classification loss
        cls_loss = torch.tensor(0.0, device=device)
        if labels is not None:
            cls_loss = F.binary_cross_entropy_with_logits(cls_logits, labels)
        
        # 5. Total loss
        total_loss = cls_loss + 0.5 * lm_loss
        
        return {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "lm_loss": lm_loss,
            "cls_logits": cls_logits,
            "lm_logits": lm_logits,
            "visual_features": visual_features
        }
    
    @torch.no_grad()
    def generate_report(
        self,
        images: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.7
    ) -> Tuple[List[str], torch.Tensor]:
        """Generate reports for input images"""
        from transformers import GPT2Tokenizer
        
        self.eval()
        device = images.device
        batch_size = images.size(0)
        
        # Encode images and get disease predictions
        visual_features = self.encode_visual(images)
        cls_logits = self.classifier(visual_features)
        disease_probs = torch.sigmoid(cls_logits)
        
        if self.decoder is None:
            return ["Model not available for generation"] * batch_size, disease_probs
        
        # Create prefix
        disease_features = self.disease_embed(disease_probs)
        combined = visual_features + disease_features
        prefix = self.visual_to_decoder(combined).unsqueeze(1)
        
        # Generate tokens
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        generated_ids = torch.full(
            (batch_size, 1),
            tokenizer.bos_token_id or tokenizer.eos_token_id,
            dtype=torch.long,
            device=device
        )
        
        for _ in range(max_length):
            # Get embeddings
            token_embeds = self.decoder.transformer.wte(generated_ids)
            inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
            
            # Forward pass
            outputs = self.decoder(inputs_embeds=inputs_embeds)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for EOS
            if (next_token == tokenizer.eos_token_id).all():
                break
        
        # Decode
        reports = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        return reports, disease_probs


# =============================================================================
# Training Loop
# =============================================================================

def train(config: TrainConfig, demo_mode: bool = False):
    """Main training function"""
    
    print("=" * 70)
    print("CogniRad++ Training Pipeline")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Mixed Precision: {config.mixed_precision}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📦 Loading dataset...")
    
    if demo_mode:
        train_dataset = DemoDataset(size=500, image_size=config.image_size)
        val_dataset = DemoDataset(size=100, image_size=config.image_size)
    else:
        # Try to download real data
        data_path = Path(config.data_dir)
        if not data_path.exists() or len(list(data_path.glob("**/*.png"))) < 100:
            success = download_iuxray_dataset(config.data_dir)
            if not success:
                print("\n⚡ Falling back to demo mode...")
                demo_mode = True
                train_dataset = DemoDataset(size=500, image_size=config.image_size)
                val_dataset = DemoDataset(size=100, image_size=config.image_size)
        
        if not demo_mode:
            train_dataset = IUXrayDataset(
                config.data_dir,
                split="train",
                image_size=config.image_size
            )
            val_dataset = IUXrayDataset(
                config.data_dir,
                split="val",
                image_size=config.image_size
            )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # Create model
    print("\n🧠 Initializing CogniRad++ model...")
    model = CogniRadLite(config)
    model = model.to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs * len(train_loader)
    )
    
    # Mixed precision
    scaler = GradScaler('cuda') if config.mixed_precision and config.device == "cuda" else None
    
    # Training loop
    print("\n🚀 Starting training...")
    best_val_loss = float("inf")
    
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_lm_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch["image"].to(config.device)
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)
            
            # Forward pass
            if scaler is not None:
                with autocast('cuda'):
                    outputs = model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs["loss"] / config.grad_accumulation
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % config.grad_accumulation == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs["loss"] / config.grad_accumulation
                loss.backward()
                
                if (batch_idx + 1) % config.grad_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            
            # Track losses
            total_loss += outputs["loss"].item()
            total_cls_loss += outputs["cls_loss"].item()
            total_lm_loss += outputs["lm_loss"].item()
            
            pbar.set_postfix({
                "loss": f"{outputs['loss'].item():.4f}",
                "cls": f"{outputs['cls_loss'].item():.4f}",
                "lm": f"{outputs['lm_loss'].item():.4f}"
            })
        
        # Epoch stats
        avg_loss = total_loss / len(train_loader)
        avg_cls = total_cls_loss / len(train_loader)
        avg_lm = total_lm_loss / len(train_loader)
        
        print(f"\n📊 Epoch {epoch} - Train Loss: {avg_loss:.4f} (CLS: {avg_cls:.4f}, LM: {avg_lm:.4f})")
        
        # Validation
        if epoch % config.eval_every == 0:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(config.device)
                    input_ids = batch["input_ids"].to(config.device)
                    attention_mask = batch["attention_mask"].to(config.device)
                    labels = batch["labels"].to(config.device)
                    
                    outputs = model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    val_loss += outputs["loss"].item()
            
            val_loss /= len(val_loader)
            print(f"   Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = output_path / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config
                }, checkpoint_path)
                print(f"   ✓ Saved best model to {checkpoint_path}")
        
        # Periodic checkpoint
        if epoch % config.save_every == 0:
            checkpoint_path = output_path / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss if 'val_loss' in dir() else avg_loss,
                "config": config
            }, checkpoint_path)
            print(f"   ✓ Saved checkpoint to {checkpoint_path}")
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Model saved to: {output_path / 'best_model.pt'}")
    print("=" * 70)
    
    # Generate sample report
    print("\n📝 Generating sample report...")
    model.eval()
    
    sample_batch = next(iter(val_loader))
    sample_image = sample_batch["image"][:1].to(config.device)
    
    try:
        reports, probs = model.generate_report(sample_image, max_length=50)
        print(f"\nGenerated Report:\n{reports[0]}")
        print(f"\nDisease Predictions:")
        for i, (label, prob) in enumerate(zip(IUXrayDataset.CHEXPERT_LABELS, probs[0].cpu().numpy())):
            if prob > 0.3:
                print(f"   {label}: {prob:.2%}")
    except Exception as e:
        print(f"   Report generation not available: {e}")
    
    return model


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogniRad++ Training")
    parser.add_argument("--data_dir", type=str, default="./data/iuxray",
                        help="Path to IU-Xray dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic demo data for testing")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cuda, cpu, or auto)")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    if args.device != "auto":
        config.device = args.device
    
    # Start training
    train(config, demo_mode=args.demo)

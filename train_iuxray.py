"""
CogniRad++ Training on IU-Xray Dataset
Uses the Indiana University Chest X-ray dataset from D:\Files
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import argparse

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
import pandas as pd
from tqdm import tqdm
import random
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class IUXrayConfig:
    """Training configuration for IU-Xray"""
    # Data paths
    images_dir: str = r"D:\Files\images\images_normalized"
    reports_csv: str = r"D:\Files\indiana_reports.csv"
    projections_csv: str = r"D:\Files\indiana_projections.csv"
    output_dir: str = "./outputs"
    
    # Model
    image_size: int = 224
    hidden_dim: int = 512
    num_classes: int = 14  # CheXpert labels
    vocab_size: int = 50257  # GPT-2 vocab
    max_seq_len: int = 256  # Longer for real reports
    
    # Training
    batch_size: int = 8
    num_epochs: int = 30
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
# IU-Xray Dataset
# =============================================================================

class IUXrayRealDataset(Dataset):
    """
    IU-Xray Dataset using real Indiana University chest X-rays
    """
    
    # CheXpert label mapping
    CHEXPERT_LABELS = [
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
        "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
        "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
        "Pleural Other", "Fracture", "Support Devices"
    ]
    
    # Keywords for label extraction
    LABEL_KEYWORDS = {
        0: ["normal", "no acute", "unremarkable", "clear lungs", "no abnormal", 
            "within normal", "no focal", "no significant"],  # No Finding
        1: ["mediastinum", "mediastinal"],  # Enlarged Cardiomediastinum
        2: ["cardiomegaly", "enlarged heart", "cardiac enlargement", "heart size enlarged",
            "cardiac silhouette enlarged", "enlarged cardiac"],  # Cardiomegaly
        3: ["opacity", "opacities", "infiltrate", "infiltrates", "density", "densities",
            "haziness", "hazy"],  # Lung Opacity
        4: ["lesion", "mass", "nodule", "nodules", "tumor"],  # Lung Lesion
        5: ["edema", "pulmonary edema", "vascular congestion", "congestion",
            "cephalization"],  # Edema
        6: ["consolidation", "consolidations", "consolidated"],  # Consolidation
        7: ["pneumonia", "infectious"],  # Pneumonia
        8: ["atelectasis", "atelectatic", "collapse", "collapsed"],  # Atelectasis
        9: ["pneumothorax"],  # Pneumothorax
        10: ["effusion", "effusions", "pleural effusion", "pleural fluid"],  # Pleural Effusion
        11: ["pleural thickening", "pleural scarring", "pleural"],  # Pleural Other
        12: ["fracture", "fractured", "rib fracture"],  # Fracture
        13: ["pacemaker", "catheter", "tube", "device", "devices", "wire", "wires",
             "stent", "port", "line", "support"]  # Support Devices
    }
    
    def __init__(
        self,
        config: IUXrayConfig,
        split: str = "train",
        use_frontal_only: bool = True
    ):
        self.config = config
        self.split = split
        self.use_frontal_only = use_frontal_only
        
        # Load reports and projections
        self.reports_df = pd.read_csv(config.reports_csv)
        self.projections_df = pd.read_csv(config.projections_csv)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load tokenizer
        try:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            self.tokenizer = None
        
        # Build samples
        self.samples = self._build_samples()
        
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
    
    def _build_samples(self) -> List[Dict]:
        """Build sample list from reports and projections"""
        samples = []
        images_dir = Path(self.config.images_dir)
        
        # Filter for frontal projections if specified
        if self.use_frontal_only:
            projections = self.projections_df[
                self.projections_df['projection'] == 'Frontal'
            ]
        else:
            projections = self.projections_df
        
        # Group projections by uid
        proj_by_uid = projections.groupby('uid')['filename'].first().to_dict()
        
        # Match with reports
        for _, row in self.reports_df.iterrows():
            uid = row['uid']
            
            if uid not in proj_by_uid:
                continue
            
            # Get image path
            image_filename = proj_by_uid[uid]
            image_path = images_dir / image_filename
            
            if not image_path.exists():
                continue
            
            # Combine findings and impression for report
            findings = str(row.get('findings', '')) if pd.notna(row.get('findings')) else ''
            impression = str(row.get('impression', '')) if pd.notna(row.get('impression')) else ''
            indication = str(row.get('indication', '')) if pd.notna(row.get('indication')) else ''
            
            report = f"{findings} {impression}".strip()
            if len(report) < 10:
                continue
            
            # Clean up report (replace XXXX placeholders)
            report = report.replace('XXXX', '[REDACTED]')
            
            samples.append({
                'uid': uid,
                'image_path': str(image_path),
                'report': report,
                'indication': indication,
                'problems': str(row.get('Problems', '')) if pd.notna(row.get('Problems')) else '',
                'mesh': str(row.get('MeSH', '')) if pd.notna(row.get('MeSH')) else ''
            })
        
        print(f"  Built {len(samples)} valid samples from IU-Xray dataset")
        return samples
    
    def _extract_labels(self, report: str, mesh: str, problems: str) -> torch.Tensor:
        """Extract CheXpert-style labels from report text and MeSH terms"""
        labels = torch.zeros(14)
        
        # Combine all text for label extraction
        all_text = f"{report} {mesh} {problems}".lower()
        
        # Check for each label
        for idx, keywords in self.LABEL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in all_text:
                    labels[idx] = 1.0
                    break
        
        # If nothing found and text suggests normal, set no finding
        if labels.sum() == 0:
            normal_indicators = ["normal", "unremarkable", "no acute", "clear"]
            for indicator in normal_indicators:
                if indicator in all_text:
                    labels[0] = 1.0
                    break
        
        # If still nothing, set uncertain (0.5) for no finding
        if labels.sum() == 0:
            labels[0] = 0.5
        
        return labels
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load and transform image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            # Return random tensor if image loading fails
            print(f"Warning: Failed to load {sample['image_path']}: {e}")
            image = torch.randn(3, self.config.image_size, self.config.image_size)
        
        # Get report
        report = sample['report']
        
        # Tokenize report
        if self.tokenizer:
            tokens = self.tokenizer(
                report,
                max_length=self.config.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(self.config.max_seq_len, dtype=torch.long)
            attention_mask = torch.ones(self.config.max_seq_len, dtype=torch.long)
        
        # Extract labels
        labels = self._extract_labels(
            sample['report'], 
            sample['mesh'], 
            sample['problems']
        )
        
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "report": report,
            "uid": sample['uid']
        }


# =============================================================================
# CogniRad++ Model (Same as prepare_and_train.py)
# =============================================================================

class CogniRadLite(nn.Module):
    """
    Lightweight CogniRad++ for training
    """
    
    def __init__(self, config: IUXrayConfig):
        super().__init__()
        self.config = config
        
        # Visual Encoder
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
            # Fallback CNN
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
        
        # Disease Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
        
        # Report Decoder
        try:
            from transformers import GPT2LMHeadModel, GPT2Config
            gpt_config = GPT2Config(
                vocab_size=config.vocab_size,
                n_embd=config.hidden_dim,
                n_layer=6,  # More layers for real data
                n_head=8
            )
            self.decoder = GPT2LMHeadModel(gpt_config)
        except:
            self.decoder = None
        
        # Projections
        self.visual_to_decoder = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.disease_embed = nn.Linear(config.num_classes, config.hidden_dim)
    
    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
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
        device = images.device
        
        # Visual encoding
        visual_features = self.encode_visual(images)
        
        # Disease classification
        cls_logits = self.classifier(visual_features)
        
        # Report generation
        lm_loss = torch.tensor(0.0, device=device)
        lm_logits = None
        
        if self.decoder is not None and input_ids is not None:
            disease_probs = torch.sigmoid(cls_logits)
            disease_features = self.disease_embed(disease_probs)
            
            combined = visual_features + disease_features
            prefix = self.visual_to_decoder(combined).unsqueeze(1)
            
            token_embeds = self.decoder.transformer.wte(input_ids)
            inputs_embeds = torch.cat([prefix, token_embeds[:, :-1, :]], dim=1)
            
            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=input_ids
            )
            lm_loss = outputs.loss
            lm_logits = outputs.logits
        
        # Classification loss
        cls_loss = torch.tensor(0.0, device=device)
        if labels is not None:
            # Handle uncertain labels (0.5)
            mask = labels != 0.5
            if mask.any():
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_logits[mask], 
                    labels[mask]
                )
        
        # Total loss (weight LM loss higher for real data)
        total_loss = cls_loss + 1.0 * lm_loss
        
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
        max_length: int = 150,
        temperature: float = 0.7
    ) -> Tuple[List[str], torch.Tensor]:
        from transformers import GPT2Tokenizer
        
        self.eval()
        device = images.device
        batch_size = images.size(0)
        
        visual_features = self.encode_visual(images)
        cls_logits = self.classifier(visual_features)
        disease_probs = torch.sigmoid(cls_logits)
        
        if self.decoder is None:
            return ["Model not available"] * batch_size, disease_probs
        
        disease_features = self.disease_embed(disease_probs)
        combined = visual_features + disease_features
        prefix = self.visual_to_decoder(combined).unsqueeze(1)
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        generated_ids = torch.full(
            (batch_size, 1),
            tokenizer.bos_token_id or tokenizer.eos_token_id,
            dtype=torch.long,
            device=device
        )
        
        for _ in range(max_length):
            token_embeds = self.decoder.transformer.wte(generated_ids)
            inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
            
            outputs = self.decoder(inputs_embeds=inputs_embeds)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            if (next_token == tokenizer.eos_token_id).all():
                break
        
        reports = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        return reports, disease_probs


# =============================================================================
# Training Loop
# =============================================================================

def train(config: IUXrayConfig):
    """Main training function for IU-Xray"""
    
    print("=" * 70)
    print("CogniRad++ Training on IU-Xray Dataset")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Images: {config.images_dir}")
    print(f"Reports: {config.reports_csv}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📦 Loading IU-Xray dataset...")
    train_dataset = IUXrayRealDataset(config, split="train")
    val_dataset = IUXrayRealDataset(config, split="val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda"
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
    print("\n🚀 Starting training on real chest X-rays...")
    best_val_loss = float("inf")
    
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_lm_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(config.device)
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)
            
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
            
            total_loss += outputs["loss"].item()
            total_cls_loss += outputs["cls_loss"].item()
            total_lm_loss += outputs["lm_loss"].item()
            
            pbar.set_postfix({
                "loss": f"{outputs['loss'].item():.4f}",
                "cls": f"{outputs['cls_loss'].item():.4f}",
                "lm": f"{outputs['lm_loss'].item():.4f}"
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_cls = total_cls_loss / len(train_loader)
        avg_lm = total_lm_loss / len(train_loader)
        
        print(f"\n📊 Epoch {epoch} - Train Loss: {avg_loss:.4f} (CLS: {avg_cls:.4f}, LM: {avg_lm:.4f})")
        
        # Validation
        if epoch % config.eval_every == 0:
            model.eval()
            val_loss = 0.0
            val_cls_loss = 0.0
            val_lm_loss = 0.0
            
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
                    val_cls_loss += outputs["cls_loss"].item()
                    val_lm_loss += outputs["lm_loss"].item()
            
            val_loss /= len(val_loader)
            val_cls_loss /= len(val_loader)
            val_lm_loss /= len(val_loader)
            
            print(f"   Val Loss: {val_loss:.4f} (CLS: {val_cls_loss:.4f}, LM: {val_lm_loss:.4f})")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = output_path / "best_model_iuxray.pt"
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
            checkpoint_path = output_path / f"checkpoint_iuxray_epoch_{epoch}.pt"
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
    print(f"   Model saved to: {output_path / 'best_model_iuxray.pt'}")
    print("=" * 70)
    
    # Generate sample report
    print("\n📝 Generating sample report from real X-ray...")
    model.eval()
    
    sample_batch = next(iter(val_loader))
    sample_image = sample_batch["image"][:1].to(config.device)
    ground_truth = sample_batch["report"][0]
    
    try:
        reports, probs = model.generate_report(sample_image, max_length=100)
        print(f"\nGround Truth Report:")
        print(f"  {ground_truth[:300]}...")
        print(f"\nGenerated Report:")
        print(f"  {reports[0][:300]}...")
        print(f"\nDisease Predictions:")
        for i, (label, prob) in enumerate(zip(IUXrayRealDataset.CHEXPERT_LABELS, probs[0].cpu().numpy())):
            if prob > 0.3:
                print(f"   {label}: {prob:.1%}")
    except Exception as e:
        print(f"   Report generation error: {e}")
    
    return model


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogniRad++ IU-Xray Training")
    parser.add_argument("--images_dir", type=str, 
                        default=r"D:\Files\images\images_normalized",
                        help="Path to images")
    parser.add_argument("--reports_csv", type=str,
                        default=r"D:\Files\indiana_reports.csv",
                        help="Path to reports CSV")
    parser.add_argument("--projections_csv", type=str,
                        default=r"D:\Files\indiana_projections.csv",
                        help="Path to projections CSV")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cuda, cpu, or auto)")
    
    args = parser.parse_args()
    
    config = IUXrayConfig(
        images_dir=args.images_dir,
        reports_csv=args.reports_csv,
        projections_csv=args.projections_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    if args.device != "auto":
        config.device = args.device
    
    train(config)

"""
Test the trained IU-Xray model with real chest X-rays
"""

import torch
import os
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass
import random

@dataclass
class CheXpertLabels:
    """CheXpert-style labels for chest X-ray findings"""
    LABELS = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices'
    ]

# CogniRadLite model (same as training)
import torch.nn as nn
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class CogniRadLite(nn.Module):
    """CogniRad++ model architecture"""
    
    def __init__(self, num_classes: int = 14, embed_dim: int = 768):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        if HAS_TIMM:
            self.encoder = timm.create_model('resnet50', pretrained=False, num_classes=0)
            encoder_dim = 2048
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
            )
            encoder_dim = 64
        
        self.projection = nn.Linear(encoder_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Skip decoder for testing - classification is the main evaluation
        self.decoder = None
        self.tokenizer = None
    
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.projection(features)
    
    def forward(self, images: torch.Tensor, input_ids=None, attention_mask=None):
        visual_features = self.encode_image(images)
        classifications = self.classifier(visual_features)
        return {'classifications': classifications, 'visual_features': visual_features}
    
    def generate_report(self, images: torch.Tensor, max_length: int = 100) -> list:
        """Generate radiology report from image"""
        self.eval()
        with torch.no_grad():
            visual_features = self.encode_image(images)
            
            if self.decoder is None or self.tokenizer is None:
                # Fallback: generate based on classification results  
                logits = self.classifier(visual_features)
                probs = torch.sigmoid(logits)
                
                reports = []
                for i in range(images.size(0)):
                    findings = []
                    for j, label in enumerate(CheXpertLabels.LABELS):
                        if probs[i, j] > 0.5:
                            findings.append(label)
                    
                    if 'No Finding' in findings or len(findings) == 0:
                        report = "The chest radiograph is normal. No acute cardiopulmonary abnormality."
                    else:
                        findings_text = ", ".join([f.lower() for f in findings if f != 'No Finding'])
                        report = f"Findings suggest {findings_text}. Clinical correlation recommended."
                    reports.append(report)
                return reports
            
            # Use GPT-2 for more sophisticated report generation
            visual_prefix = self.visual_to_text(visual_features)
            
            reports = []
            for i in range(images.size(0)):
                # Get classification context
                logits = self.classifier(visual_features[i:i+1])
                probs = torch.sigmoid(logits)[0]
                
                findings = []
                for j, label in enumerate(CheXpertLabels.LABELS):
                    if probs[j] > 0.5:
                        findings.append(label.lower())
                
                if len(findings) == 0 or 'no finding' in findings:
                    context = "Normal chest radiograph. "
                else:
                    context = f"Chest X-ray shows: {', '.join(findings)}. "
                
                # Generate with GPT-2
                input_ids = self.tokenizer.encode(context, return_tensors='pt').to(images.device)
                
                outputs = self.decoder.generate(
                    input_ids,
                    max_length=max_length,
                    num_beams=3,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                report = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                reports.append(report)
            
            return reports


def test_model():
    print("=" * 70)
    print("Testing CogniRad++ Model Trained on IU-Xray Dataset")
    print("=" * 70)
    
    # Load model
    model_path = "outputs/best_model_iuxray.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"\n📦 Loading model from {model_path}...")
    model = CogniRadLite(num_classes=14, embed_dim=768)
    
    # Add safe globals for loading pickle
    import pickle
    
    # Custom unpickler to handle IUXrayConfig
    class _IUXrayConfigPlaceholder:
        pass
    
    # Temporarily add to __main__ for unpickling
    import __main__
    __main__.IUXrayConfig = _IUXrayConfigPlaceholder
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        best_val_loss = checkpoint.get('val_loss', 'N/A')
        print(f"   Loaded model with val_loss: {best_val_loss}")
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    print("   ✓ Model loaded successfully")
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get test images
    image_dir = r"D:\Files\images\images_normalized"
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')][:5]
    
    print(f"\n🔬 Testing on {len(image_files)} real chest X-rays...")
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        print(f"\n{'─' * 50}")
        print(f"Image: {img_file}")
        
        # Load and process image
        image = Image.open(img_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.sigmoid(output['classifications'])[0]
        
        # Show findings
        print("\n📋 Disease Classification:")
        findings = []
        for i, label in enumerate(CheXpertLabels.LABELS):
            prob = probs[i].item()
            status = "✓" if prob > 0.5 else " "
            if prob > 0.5:
                findings.append(label)
            print(f"   [{status}] {label}: {prob:.1%}")
        
        # Generate report
        print("\n📝 Generated Report:")
        try:
            reports = model.generate_report(img_tensor)
            print(f"   {reports[0]}")
        except Exception as e:
            print(f"   (Report generation error: {e})")
    
    print(f"\n{'=' * 70}")
    print("✅ Model testing complete!")
    print("=" * 70)


if __name__ == "__main__":
    test_model()

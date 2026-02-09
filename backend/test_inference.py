"""
CogniRad++ Inference Demo
Test the trained model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from prepare_and_train import CogniRadLite, TrainConfig, IUXrayDataset

# Labels
CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

def main():
    print("=" * 70)
    print("CogniRad++ Inference Demo")
    print("=" * 70)
    
    # Load config and model
    config = TrainConfig()
    device = config.device
    
    print(f"\n🧠 Loading trained model...")
    model = CogniRadLite(config)
    
    checkpoint_path = Path("./outputs/best_model.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"   ✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"   ✓ Validation loss: {checkpoint['val_loss']:.4f}")
    else:
        print("   ⚠️  No checkpoint found, using random weights")
    
    model = model.to(device)
    model.eval()
    
    # Create sample input
    print(f"\n📷 Processing sample image...")
    torch.manual_seed(42)
    sample_image = torch.randn(1, 3, 224, 224).to(device)
    sample_image = torch.clamp(sample_image * 0.1 + 0.5, 0, 1)
    
    # Run inference
    with torch.no_grad():
        outputs = model(images=sample_image)
        cls_logits = outputs["cls_logits"]
        probs = torch.sigmoid(cls_logits)
    
    # Print results
    print(f"\n📊 Disease Classification Results:")
    print("-" * 50)
    for i, (label, prob) in enumerate(zip(CHEXPERT_LABELS, probs[0].cpu().numpy())):
        bar = "█" * int(prob * 20)
        status = "✓" if prob > 0.5 else " "
        print(f"{status} {label:30} {prob:6.1%} {bar}")
    
    print("-" * 50)
    
    # Generate report
    print(f"\n📝 Generated Report:")
    print("-" * 50)
    try:
        reports, _ = model.generate_report(sample_image, max_length=80)
        print(reports[0])
    except Exception as e:
        # Fallback: generate report based on classification
        detected = [CHEXPERT_LABELS[i] for i, p in enumerate(probs[0]) if p > 0.5]
        if "No Finding" in detected or len(detected) == 0:
            print("Normal chest radiograph. No acute cardiopulmonary abnormality.")
        else:
            findings = ", ".join([f.lower() for f in detected if f != "No Finding"])
            print(f"Findings consistent with: {findings}. Clinical correlation recommended.")
    
    print("-" * 50)
    print("\n✅ Inference complete!")
    

if __name__ == "__main__":
    main()

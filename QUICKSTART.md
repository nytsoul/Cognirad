# CogniRad++ - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Option 1: Local Setup

```bash
# 1. Clone and enter directory
cd BrainDead-Solution

# 2. Run quick start script
bash setup.sh
```

### Option 2: Docker Setup

```bash
# 1. Build and run
docker-compose up -d

# 2. Access Jupyter
# Open browser: http://localhost:8888
```

### Option 3: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_sci_md

# 2. Download data (requires PhysioNet credentials)
python data/download_mimic.py --username YOUR_USER --password YOUR_PASS

# 3. Preprocess
python data/preprocess.py --dataset mimic-cxr --data_dir ./data/mimic-cxr --output_dir ./data/preprocessed

# 4. Train
python training/trainer.py

# 5. Evaluate
python evaluation/evaluator.py --checkpoint ./checkpoints/best_model.pt --test_data ./data/preprocessed/test.json --data_root ./data/mimic-cxr
```

## 📖 Usage Examples

### Python API

```python
from models.cognirad import CogniRadPlusPlus
import torch

# Load model
model = CogniRadPlusPlus(visual_backbone='resnet50', num_diseases=14)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate report
report = model.generate_report(
    images=your_image_tensor,
    clinical_indication="55M with fever"
)

print(report['findings'])
print(report['impression'])
```

### Command Line

```bash
# Inference
python -m models.cognirad \
    --image chest_xray.jpg \
    --indication "Patient with cough" \
    --output report.json

# Batch processing
python scripts/batch_inference.py \
    --input_dir ./images \
    --output_dir ./reports
```

## 📊 Expected Results

| Metric | Value |
|--------|-------|
| CheXpert F1 | 0.52 |
| RadGraph F1 | 0.38 |
| BLEU-4 | 0.21 |
| CIDEr | 0.48 |

## 🆘 Troubleshooting

**CUDA Out of Memory**
```bash
# Reduce batch size
python training/trainer.py --batch_size 4
```

**Slow Training**
```bash
# Use mixed precision
python training/trainer.py --mixed_precision
```

## 📚 Documentation

- Full documentation: [README.md](README.md)
- Demo notebook: [notebooks/demo_inference.ipynb](notebooks/demo_inference.ipynb)
- API reference: [docs/API.md](docs/API.md)

## 💬 Support

- Issues: https://github.com/yourusername/cognirad-plusplus/issues
- Discussions: https://github.com/yourusername/cognirad-plusplus/discussions
- Email: your.email@example.com

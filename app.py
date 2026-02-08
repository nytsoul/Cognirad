"""
CogniRad++ Flask API
Serves the radiology report generation model via REST API
"""

import os
import io
import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from typing import Dict, Any

# Try to import torch and dependencies
try:
    import torch
    import torchvision.transforms as transforms
    from models.cognirad import CogniRadPlusPlus
    MODEL_AVAILABLE = True
    print("Torch and Model modules loaded successfully")
except ImportError as e:
    MODEL_AVAILABLE = False
    print(f"Warning: Deep learning dependencies not found ({e}). Running in MOCK mode.")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
DEVICE = 'cuda' if (MODEL_AVAILABLE and torch.cuda.is_available()) else 'cpu'
CHECKPOINT_PATH = os.environ.get('MODEL_CHECKPOINT', './checkpoints/best_model.pt')
# Force mock if model not available
USE_MOCK = os.environ.get('USE_MOCK', 'false').lower() == 'true' or not MODEL_AVAILABLE

# Image preprocessing
if MODEL_AVAILABLE:
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
else:
    image_transform = None

class MockModel:
    """Mock model for testing when actual model is unavailable"""
    
    def __init__(self):
        self.device = DEVICE
    
    def to(self, device):
        return self
    
    def eval(self):
        return self
    
    def generate_report(self, images, clinical_indication="", confidence_threshold=0.7, include_evidence=True):
        """Generate a mock report with enhanced cognitive architecture data"""
        return {
            'findings': "The lungs are clear without focal consolidation, pleural effusion, or pneumothorax. "
                       "The cardiac silhouette is normal in size. The mediastinal and hilar contours are unremarkable. "
                       "No acute bony abnormality is identified. Minimal degenerative changes are seen in the thoracic spine.",
            'impression': "No acute cardiopulmonary abnormality.",
            'predicted_diseases': [
                {'label': 'No Finding', 'probability': 0.92, 'confidence': 0.95}
            ],
            'uncertain_findings': [],
            'clinical_indication': clinical_indication,
            'warnings': [],
            'perception_layers': [
                {'name': 'Left Lung', 'path': 'M 120,80 Q 80,100 80,250 Q 100,320 150,300 Q 180,280 170,150 Z', 'confidence': 0.98},
                {'name': 'Right Lung', 'path': 'M 280,80 Q 320,100 320,250 Q 300,320 250,300 Q 220,280 230,150 Z', 'confidence': 0.97},
                {'name': 'Heart', 'path': 'M 180,200 Q 200,180 230,220 Q 250,300 200,320 Q 150,300 170,220 Z', 'confidence': 0.94}
            ],
            'reasoning_steps': [
                {'stage': 'Perception', 'desc': 'Extracted hierarchical features from 3 anatomical regions.', 'status': 'complete'},
                {'stage': 'Diagnosis', 'desc': 'MIX-MLP identified "No Finding" as primary clinical hypothesis.', 'status': 'complete'},
                {'stage': 'Verification', 'desc': 'RCTA triangular loop verified hypothesis against visual evidence.', 'status': 'complete'}
            ],
            'attention_maps': {
                'lungs': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==', # Fake pixel
            }
        }

def load_model():
    """Load the CogniRad++ model or return mock"""
    global USE_MOCK
    
    if USE_MOCK:
        print("Using mock model")
        return MockModel()
    
    try:
        if os.path.exists(CHECKPOINT_PATH):
            print(f"Loading model from {CHECKPOINT_PATH}")
            model = CogniRadPlusPlus.from_pretrained(CHECKPOINT_PATH)
        else:
            print("No checkpoint found, initializing new model")
            model = CogniRadPlusPlus(
                visual_backbone='resnet50',
                num_diseases=14,
                pretrained=True
            )
        
        model = model.to(DEVICE)
        model.eval()
        print(f"Model loaded successfully on {DEVICE}")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to mock model")
        USE_MOCK = True
        return MockModel()


# Initialize model
print("Initializing CogniRad++ API...")
model = load_model()
print("API ready!")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': not USE_MOCK,
        'device': str(DEVICE),
        'mock_mode': USE_MOCK
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Generate radiology report from chest X-ray
    
    Expected form data:
        - image: Image file (JPEG/PNG)
        - indication: Clinical indication text (optional)
        - confidence_threshold: Confidence threshold (optional, default 0.7)
    """
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Get optional parameters
        clinical_indication = request.form.get('indication', 'Chest X-ray')
        confidence_threshold = float(request.form.get('confidence_threshold', 0.7))
        
        # Load and preprocess image
        try:
            image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        except Exception as e:
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
        
        # Transform image
        if MODEL_AVAILABLE and image_transform:
            image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)
        else:
            # For mock mode, just pass None or raw image
            image_tensor = None
        
        # Generate report
        with torch.no_grad() if (MODEL_AVAILABLE and 'torch' in globals()) else io.BytesIO(): # Dummy context
            report = model.generate_report(
                images=image_tensor,
                clinical_indication=clinical_indication,
                confidence_threshold=confidence_threshold,
                include_evidence=True
            )
        
        # Format response
        response = {
            'findings': report['findings'],
            'impression': report['impression'],
            'predicted_diseases': report['predicted_diseases'],
            'uncertain_findings': report['uncertain_findings'],
            'clinical_indication': report['clinical_indication'],
            'warnings': report.get('warnings', []),
            'perception_layers': report.get('perception_layers', []),
            'reasoning_steps': report.get('reasoning_steps', []),
            'attention_maps': report.get('attention_maps', {}),
            'mock_mode': USE_MOCK
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_name': 'CogniRad++',
        'version': '1.0',
        'description': 'Knowledge-Grounded, Explainable Cognitive Radiology Assistant',
        'supported_diseases': [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ],
        'mock_mode': USE_MOCK
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'true').lower() == 'true'
    
    print(f"\n{'='*60}")
    print(f"CogniRad++ API Server")
    print(f"{'='*60}")
    print(f"Running on: http://localhost:{port}")
    print(f"Device: {DEVICE}")
    print(f"Mock mode: {USE_MOCK}")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)

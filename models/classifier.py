"""
MIX-MLP: Multi-Path Disease Classifier
Predicts CheXpert pathologies with confidence scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class ResidualPath(nn.Module):
    """
    Residual path preserves core visual semantics
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        
        self.path = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Residual connection
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.path(x) + self.residual_proj(x)


class ExpansionPath(nn.Module):
    """
    Expansion path models disease co-occurrence patterns
    """
    
    def __init__(self, input_dim: int, expansion_factor: int = 4, dropout: float = 0.2):
        super().__init__()
        
        expanded_dim = input_dim * expansion_factor
        
        self.expansion = nn.Sequential(
            nn.Linear(input_dim, expanded_dim),
            nn.LayerNorm(expanded_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, expanded_dim // 2),
            nn.LayerNorm(expanded_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_dim // 2, input_dim),
            nn.LayerNorm(input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expansion(x)


class DiseaseCooccurrenceModule(nn.Module):
    """
    Models relationships between diseases
    e.g., Edema often co-occurs with Cardiomegaly
    """
    
    def __init__(self, num_diseases: int, hidden_dim: int = 128):
        super().__init__()
        
        self.num_diseases = num_diseases
        
        # Learnable disease relationship matrix
        self.disease_relations = nn.Parameter(
            torch.randn(num_diseases, num_diseases) * 0.01
        )
        
        # Attention for disease interactions
        self.interaction_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Disease embedding
        self.disease_embeddings = nn.Embedding(num_diseases, hidden_dim)
    
    def forward(self, disease_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            disease_logits: [B, num_diseases] raw disease predictions
        
        Returns:
            refined_logits: [B, num_diseases] co-occurrence refined predictions
        """
        batch_size = disease_logits.size(0)
        
        # Get disease embeddings
        disease_ids = torch.arange(
            self.num_diseases,
            device=disease_logits.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        disease_emb = self.disease_embeddings(disease_ids)  # [B, num_diseases, hidden_dim]
        
        # Apply self-attention for disease interactions
        refined_emb, _ = self.interaction_attention(
            disease_emb, disease_emb, disease_emb
        )
        
        # Combine with relation matrix
        relation_weights = torch.sigmoid(self.disease_relations)  # [num_diseases, num_diseases]
        
        # Apply co-occurrence relationships
        disease_probs = torch.sigmoid(disease_logits)  # [B, num_diseases]
        cooccurrence_influence = torch.matmul(disease_probs, relation_weights)  # [B, num_diseases]
        
        # Combine original predictions with co-occurrence
        refined_logits = disease_logits + 0.3 * cooccurrence_influence
        
        return refined_logits


class MIXMLPClassifier(nn.Module):
    """
    MIX-MLP: Multi-Path Disease Classifier
    Combines residual and expansion paths with disease co-occurrence modeling
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_diseases: int = 14,  # CheXpert has 14 labels
        expansion_factor: int = 4,
        dropout: float = 0.2
    ):
        """
        Args:
            input_dim: Input feature dimension from encoder
            hidden_dim: Hidden dimension
            num_diseases: Number of disease labels to predict
            expansion_factor: Expansion factor for expansion path
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_diseases = num_diseases
        
        # Residual path (preserves semantics)
        self.residual_path = ResidualPath(input_dim, hidden_dim, dropout)
        
        # Expansion path (models co-occurrence)
        self.expansion_path = ExpansionPath(input_dim, expansion_factor, dropout)
        
        # Combine paths
        self.path_fusion = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Disease classifier heads
        self.disease_classifier = nn.Linear(hidden_dim, num_diseases)
        
        # Co-occurrence module
        self.cooccurrence_module = DiseaseCooccurrenceModule(num_diseases, hidden_dim)
        
        # Confidence estimator (predicts uncertainty)
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_diseases),
            nn.Sigmoid()  # Confidence in [0, 1]
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        return_confidence: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            visual_features: [B, input_dim] visual features from encoder
            return_confidence: Whether to compute confidence scores
        
        Returns:
            Dictionary containing:
                - disease_logits: [B, num_diseases] raw predictions
                - disease_probs: [B, num_diseases] probabilities
                - confidence: [B, num_diseases] confidence scores
                - intermediate_features: [B, hidden_dim] for decoder
        """
        # Residual path
        residual_features = self.residual_path(visual_features)  # [B, hidden_dim]
        
        # Expansion path
        expansion_features = self.expansion_path(visual_features)  # [B, input_dim]
        
        # Fuse paths
        combined = torch.cat([residual_features, expansion_features], dim=1)  # [B, hidden_dim + input_dim]
        fused_features = self.path_fusion(combined)  # [B, hidden_dim]
        
        # Predict diseases
        disease_logits = self.disease_classifier(fused_features)  # [B, num_diseases]
        
        # Apply co-occurrence refinement
        refined_logits = self.cooccurrence_module(disease_logits)  # [B, num_diseases]
        
        # Compute probabilities
        disease_probs = torch.sigmoid(refined_logits)
        
        output = {
            'disease_logits': refined_logits,
            'disease_probs': disease_probs,
            'intermediate_features': fused_features
        }
        
        # Estimate confidence
        if return_confidence:
            confidence = self.confidence_estimator(fused_features)
            output['confidence'] = confidence
        
        return output
    
    def predict_with_threshold(
        self,
        visual_features: torch.Tensor,
        threshold: float = 0.5,
        confidence_threshold: float = 0.7
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict diseases with thresholding and uncertainty filtering
        
        Args:
            visual_features: [B, input_dim] visual features
            threshold: Probability threshold for positive prediction
            confidence_threshold: Minimum confidence to report
        
        Returns:
            predictions: [B, num_diseases] binary predictions
            probabilities: [B, num_diseases] disease probabilities
            uncertainties: [B, num_diseases] uncertainty flags
        """
        with torch.no_grad():
            output = self.forward(visual_features, return_confidence=True)
            
            probs = output['disease_probs']
            confidence = output['confidence']
            
            # Binary predictions
            predictions = (probs > threshold).float()
            
            # Flag uncertain predictions
            uncertainties = (confidence < confidence_threshold).float()
        
        return predictions, probs, uncertainties


class CheXpertLabelEncoder:
    """
    Encodes CheXpert labels with proper handling of uncertain/missing values
    """
    
    def __init__(self):
        self.label_names = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices'
        ]
        
        self.label_to_idx = {name: idx for idx, name in enumerate(self.label_names)}
    
    def encode(self, labels: Dict[str, float]) -> torch.Tensor:
        """
        Encode label dictionary to tensor
        CheXpert encoding: 1.0 = positive, 0.0 = negative, -1.0 = uncertain, NaN = missing
        
        Args:
            labels: Dictionary mapping label name to value
        
        Returns:
            encoded: [num_diseases] tensor with values in {0, 1}
        """
        encoded = torch.zeros(len(self.label_names))
        
        for name, value in labels.items():
            if name in self.label_to_idx:
                idx = self.label_to_idx[name]
                if value == 1.0:
                    encoded[idx] = 1.0
                # Treat uncertain (-1) as negative for training
                # Can be modified based on strategy
        
        return encoded
    
    def decode(self, predictions: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
        """
        Decode predictions to label dictionary
        
        Args:
            predictions: [num_diseases] probability tensor
            threshold: Threshold for positive prediction
        
        Returns:
            labels: Dictionary mapping label name to probability
        """
        labels = {}
        for idx, name in enumerate(self.label_names):
            prob = predictions[idx].item()
            if prob > threshold:
                labels[name] = prob
        
        return labels


if __name__ == "__main__":
    # Test classifier
    print("Testing MIX-MLP Classifier...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    classifier = MIXMLPClassifier(
        input_dim=768,
        hidden_dim=512,
        num_diseases=14
    ).to(device)
    
    # Test forward pass
    batch_size = 4
    visual_features = torch.randn(batch_size, 768).to(device)
    
    output = classifier(visual_features, return_confidence=True)
    
    print("\nOutput shapes:")
    print(f"Disease logits: {output['disease_logits'].shape}")
    print(f"Disease probs: {output['disease_probs'].shape}")
    print(f"Confidence: {output['confidence'].shape}")
    print(f"Intermediate features: {output['intermediate_features'].shape}")
    
    # Test prediction with threshold
    predictions, probs, uncertainties = classifier.predict_with_threshold(visual_features)
    print(f"\nPredictions: {predictions.shape}")
    print(f"Uncertainties: {uncertainties.shape}")
    
    # Test label encoder
    encoder = CheXpertLabelEncoder()
    test_labels = {'Cardiomegaly': 1.0, 'Edema': 1.0, 'Pneumonia': 0.0}
    encoded = encoder.encode(test_labels)
    print(f"\nEncoded labels: {encoded}")
    
    decoded = encoder.decode(probs[0])
    print(f"Decoded predictions: {decoded}")
    
    print("\n✅ MIX-MLP Classifier test passed!")

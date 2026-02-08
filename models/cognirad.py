"""
CogniRad++: Complete Model Architecture
Integrates PRO-FA Encoder, MIX-MLP Classifier, and RCTA Decoder
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from .encoder import PROFAEncoder
from .classifier import MIXMLPClassifier, CheXpertLabelEncoder
from .decoder import RCTADecoder, ReportGenerator
from transformers import AutoModel, AutoTokenizer


class CogniRadPlusPlus(nn.Module):
    """
    CogniRad++: Knowledge-Grounded Cognitive Radiology Assistant
    
    Pipeline:
        Image → PRO-FA Encoder → Visual Features
                                ↓
        Clinical Text → Text Encoder → Text Features
                                       ↓
        Visual Features → MIX-MLP → Disease Predictions
                                    ↓
        [Visual, Text, Disease] → RCTA Decoder → Report
    """
    
    def __init__(
        self,
        # Encoder parameters
        visual_backbone: str = 'resnet50',
        text_encoder: str = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        num_concepts: int = 50,
        visual_hidden_dim: int = 512,
        encoder_output_dim: int = 768,
        
        # Classifier parameters
        num_diseases: int = 14,
        classifier_hidden_dim: int = 512,
        
        # Decoder parameters
        decoder_hidden_dim: int = 768,
        decoder_num_layers: int = 6,
        decoder_num_heads: int = 8,
        
        # Generation parameters
        max_length: int = 512,
        num_beams: int = 4,
        
        # Training parameters
        pretrained: bool = True,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        self.num_diseases = num_diseases
        self.max_length = max_length
        
        # ========== PRO-FA Visual Encoder ==========
        self.visual_encoder = PROFAEncoder(
            backbone=visual_backbone,
            num_concepts=num_concepts,
            hidden_dim=visual_hidden_dim,
            output_dim=encoder_output_dim,
            pretrained=pretrained
        )
        
        # ========== Clinical Text Encoder ==========
        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder)
        
        # Text feature projection
        text_dim = self.text_encoder.config.hidden_size
        self.text_projection = nn.Linear(text_dim, encoder_output_dim)
        
        # ========== MIX-MLP Disease Classifier ==========
        self.disease_classifier = MIXMLPClassifier(
            input_dim=encoder_output_dim,
            hidden_dim=classifier_hidden_dim,
            num_diseases=num_diseases
        )
        
        # ========== RCTA Report Generator ==========
        self.report_generator = ReportGenerator(
            encoder_dim=encoder_output_dim + classifier_hidden_dim,  # Concatenate visual + disease features
            max_length=max_length,
            num_beams=num_beams
        )
        
        # Label encoder for disease names
        self.label_encoder = CheXpertLabelEncoder()
        
        # Freeze encoder if specified
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze visual and text encoders"""
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode clinical indication text
        
        Args:
            input_ids: [B, seq_len] token IDs
            attention_mask: [B, seq_len] attention mask
        
        Returns:
            text_features: [B, encoder_output_dim]
        """
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [B, text_dim]
        
        # Project to common dimension
        text_features = self.text_projection(text_features)  # [B, encoder_output_dim]
        
        return text_features
    
    def forward(
        self,
        images: torch.Tensor,
        indication_input_ids: torch.Tensor,
        indication_attention_mask: torch.Tensor,
        report_input_ids: Optional[torch.Tensor] = None,
        report_attention_mask: Optional[torch.Tensor] = None,
        chexpert_labels: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: [B, 3, H, W] chest X-ray images
            indication_input_ids: [B, seq_len] clinical indication tokens
            indication_attention_mask: [B, seq_len] attention mask
            report_input_ids: [B, seq_len] target report tokens (for training)
            report_attention_mask: [B, seq_len] report attention mask
            chexpert_labels: [B, num_diseases] ground truth labels
            return_attention: Whether to return attention maps
        
        Returns:
            Dictionary with outputs and losses
        """
        # ========== Stage 1: Visual Encoding (PRO-FA) ==========
        visual_output = self.visual_encoder(images, return_attention=return_attention)
        visual_features = visual_output['aligned_features']  # [B, encoder_output_dim]
        
        # ========== Stage 2: Text Encoding ==========
        text_features = self.encode_text(
            indication_input_ids,
            indication_attention_mask
        )  # [B, encoder_output_dim]
        
        # ========== Stage 3: Disease Classification (MIX-MLP) ==========
        disease_output = self.disease_classifier(
            visual_features,
            return_confidence=True
        )
        
        disease_probs = disease_output['disease_probs']  # [B, num_diseases]
        disease_features = disease_output['intermediate_features']  # [B, classifier_hidden_dim]
        confidence = disease_output['confidence']  # [B, num_diseases]
        
        # ========== Stage 4: Report Generation (RCTA) ==========
        # Combine visual and disease features
        combined_features = torch.cat([
            visual_features,
            disease_features
        ], dim=1)  # [B, encoder_output_dim + classifier_hidden_dim]
        
        # Generate report
        if report_input_ids is not None:
            # Training mode
            generation_output = self.report_generator(
                encoder_features=combined_features,
                target_ids=report_input_ids,
                attention_mask=report_attention_mask
            )
            
            report_loss = generation_output['loss']
        else:
            # Inference mode
            generation_output = {}
            report_loss = None
        
        # ========== Compute Losses ==========
        total_loss = 0
        losses = {}
        
        # Disease classification loss
        if chexpert_labels is not None:
            disease_loss = nn.BCEWithLogitsLoss()(
                disease_output['disease_logits'],
                chexpert_labels
            )
            losses['disease_loss'] = disease_loss
            total_loss += disease_loss
        
        # Report generation loss
        if report_loss is not None:
            losses['report_loss'] = report_loss
            total_loss += 2.0 * report_loss  # Weight report loss higher
        
        losses['total_loss'] = total_loss
        
        # ========== Prepare Output ==========
        output = {
            'visual_features': visual_features,
            'text_features': text_features,
            'disease_probs': disease_probs,
            'confidence': confidence,
            'losses': losses,
            'disease_output': disease_output
        }
        
        if return_attention:
            output['visual_attention'] = visual_output.get('concept_attention')
        
        return output
    
    @torch.no_grad()
    def generate_report(
        self,
        images: torch.Tensor,
        clinical_indication: str = "Chest X-ray",
        confidence_threshold: float = 0.7,
        include_evidence: bool = True
    ) -> Dict[str, any]:
        """
        Generate complete radiology report with evidence
        
        Args:
            images: [B, 3, H, W] chest X-ray images
            clinical_indication: Clinical indication text
            confidence_threshold: Threshold for reporting findings
            include_evidence: Whether to include attention maps
        
        Returns:
            Dictionary with generated report and metadata
        """
        self.eval()
        batch_size = images.size(0)
        
        # Tokenize clinical indication
        indication_tokens = self.text_tokenizer(
            [clinical_indication] * batch_size,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=64
        ).to(images.device)
        
        # ========== Forward Pass ==========
        visual_output = self.visual_encoder(images, return_attention=include_evidence)
        visual_features = visual_output['aligned_features']
        
        text_features = self.encode_text(
            indication_tokens['input_ids'],
            indication_tokens['attention_mask']
        )
        
        disease_output = self.disease_classifier(
            visual_features,
            return_confidence=True
        )
        
        disease_probs = disease_output['disease_probs']
        confidence = disease_output['confidence']
        disease_features = disease_output['intermediate_features']
        
        # ========== Generate Report Text ==========
        combined_features = torch.cat([visual_features, disease_features], dim=1)
        
        findings_text, _ = self.report_generator.generate(
            encoder_features=combined_features,
            prompt="FINDINGS:",
            max_length=256
        )
        
        impression_text, _ = self.report_generator.generate(
            encoder_features=combined_features,
            prompt="IMPRESSION:",
            max_length=128
        )
        
        # ========== Format Predictions ==========
        # Get positive predictions above threshold
        positive_diseases = []
        uncertain_diseases = []
        
        for i, (prob, conf) in enumerate(zip(disease_probs[0], confidence[0])):
            label_name = self.label_encoder.label_names[i]
            
            if prob > 0.5:  # Positive prediction
                disease_info = {
                    'label': label_name,
                    'probability': prob.item(),
                    'confidence': conf.item()
                }
                
                if conf < confidence_threshold:
                    uncertain_diseases.append(disease_info)
                else:
                    positive_diseases.append(disease_info)
        
        # ========== Compile Report ==========
        report = {
            'findings': findings_text,
            'impression': impression_text,
            'predicted_diseases': positive_diseases,
            'uncertain_findings': uncertain_diseases,
            'clinical_indication': clinical_indication
        }
        
        # Add evidence if requested
        if include_evidence:
            report['attention_maps'] = {
                'concept_attention': visual_output.get('concept_attention'),
                'multi_scale_features': visual_output.get('multi_scale_features')
            }
        
        # Generate uncertainty warnings
        if uncertain_diseases:
            report['warnings'] = [
                "Low confidence predictions detected. Further clinical correlation recommended."
            ]
        
        return report
    
    def get_explanation(
        self,
        images: torch.Tensor,
        disease_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Get visual explanation for a specific disease prediction
        
        Args:
            images: [B, 3, H, W] chest X-ray images
            disease_idx: Index of disease to explain
        
        Returns:
            Dictionary with attention maps and feature attributions
        """
        self.eval()
        
        # Forward pass with attention
        visual_output = self.visual_encoder(images, return_attention=True)
        visual_features = visual_output['aligned_features']
        
        # Get disease predictions
        disease_output = self.disease_classifier(visual_features, return_confidence=True)
        
        # Get attention for specific disease
        # This is simplified - in production, use proper gradient-based attribution
        attention_maps = visual_output.get('concept_attention')
        
        return {
            'attention_maps': attention_maps,
            'prediction': disease_output['disease_probs'][:, disease_idx],
            'confidence': disease_output['confidence'][:, disease_idx]
        }


if __name__ == "__main__":
    # Test complete model
    print("Testing CogniRad++ Complete Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CogniRadPlusPlus(
        visual_backbone='resnet50',
        num_diseases=14,
        pretrained=False  # Set to False for faster testing
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Create dummy text inputs
    indication_ids = torch.randint(0, 1000, (batch_size, 32)).to(device)
    indication_mask = torch.ones(batch_size, 32).to(device)
    report_ids = torch.randint(0, 1000, (batch_size, 128)).to(device)
    report_mask = torch.ones(batch_size, 128).to(device)
    labels = torch.randint(0, 2, (batch_size, 14)).float().to(device)
    
    print("\nTesting training forward pass...")
    output = model(
        images=images,
        indication_input_ids=indication_ids,
        indication_attention_mask=indication_mask,
        report_input_ids=report_ids,
        report_attention_mask=report_mask,
        chexpert_labels=labels
    )
    
    print("\nOutput keys:", output.keys())
    print(f"Disease probabilities: {output['disease_probs'].shape}")
    print(f"Confidence: {output['confidence'].shape}")
    print(f"Total loss: {output['losses']['total_loss'].item():.4f}")
    
    print("\nTesting inference mode...")
    report = model.generate_report(
        images=images,
        clinical_indication="55M with fever and cough"
    )
    
    print("\nGenerated Report:")
    print(f"Findings: {report['findings'][:100]}...")
    print(f"Impression: {report['impression'][:100]}...")
    print(f"Predicted diseases: {len(report['predicted_diseases'])}")
    
    print("\n✅ CogniRad++ complete model test passed!")

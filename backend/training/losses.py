"""
Loss Functions for CogniRad++
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class DiseaseClassificationLoss(nn.Module):
    """
    Multi-label disease classification loss with class balancing
    """
    
    def __init__(
        self,
        num_classes: int = 14,
        pos_weight: torch.Tensor = None,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Args:
            num_classes: Number of disease classes
            pos_weight: Weight for positive class (handles imbalance)
            use_focal_loss: Whether to use focal loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if pos_weight is None:
            # Default weights (can be computed from dataset statistics)
            pos_weight = torch.ones(num_classes) * 2.0
        
        self.register_buffer('pos_weight', pos_weight)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes] predicted logits
            targets: [B, num_classes] ground truth labels {0, 1}
            mask: [B, num_classes] optional mask for uncertain labels
        
        Returns:
            loss: Scalar loss value
        """
        if self.use_focal_loss:
            return self._focal_loss(logits, targets, mask)
        else:
            return self._bce_loss(logits, targets, mask)
    
    def _bce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Binary cross-entropy loss"""
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss
    
    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Focal loss for handling class imbalance"""
        probs = torch.sigmoid(logits)
        
        # Compute focal weight
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none'
        )
        
        # Apply focal weight and alpha
        focal_loss = self.focal_alpha * focal_weight * bce
        
        if mask is not None:
            focal_loss = focal_loss * mask
            focal_loss = focal_loss.sum() / (mask.sum() + 1e-8)
        else:
            focal_loss = focal_loss.mean()
        
        return focal_loss


class ReportGenerationLoss(nn.Module):
    """
    Report generation loss with label smoothing
    """
    
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        label_smoothing: float = 0.1,
        ignore_index: int = -100
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            pad_token_id: ID of padding token
            label_smoothing: Label smoothing factor
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, seq_len, vocab_size] predicted logits
            targets: [B, seq_len] target token IDs
            attention_mask: [B, seq_len] mask for padding
        
        Returns:
            loss: Scalar loss value
        """
        # Reshape for cross entropy
        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = targets.view(-1)
        
        # Create mask
        if attention_mask is not None:
            mask = attention_mask.view(-1).bool()
        else:
            mask = targets_flat != self.pad_token_id
        
        # Compute cross entropy with label smoothing
        loss = F.cross_entropy(
            logits_flat[mask],
            targets_flat[mask],
            label_smoothing=self.label_smoothing,
            reduction='mean'
        )
        
        return loss


class ConsistencyLoss(nn.Module):
    """
    Enforces consistency between predicted diseases and generated text
    """
    
    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        disease_probs: torch.Tensor,
        report_embeddings: torch.Tensor,
        disease_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency between disease predictions and report
        
        Args:
            disease_probs: [B, num_diseases] disease probabilities
            report_embeddings: [B, hidden_dim] report features
            disease_embeddings: [B, hidden_dim] disease features
        
        Returns:
            loss: Consistency loss
        """
        # Cosine similarity
        similarity = F.cosine_similarity(
            report_embeddings,
            disease_embeddings,
            dim=1
        )
        
        # We want high similarity
        loss = (1 - similarity).mean()
        
        return self.weight * loss


class CombinedLoss(nn.Module):
    """
    Combined loss for CogniRad++ training
    """
    
    def __init__(
        self,
        num_diseases: int = 14,
        vocab_size: int = 30522,
        disease_weight: float = 1.0,
        report_weight: float = 2.0,
        consistency_weight: float = 0.5,
        use_focal_loss: bool = True
    ):
        """
        Args:
            num_diseases: Number of disease classes
            vocab_size: Vocabulary size
            disease_weight: Weight for disease classification loss
            report_weight: Weight for report generation loss
            consistency_weight: Weight for consistency loss
            use_focal_loss: Use focal loss for disease classification
        """
        super().__init__()
        
        self.disease_weight = disease_weight
        self.report_weight = report_weight
        self.consistency_weight = consistency_weight
        
        # Component losses
        self.disease_loss_fn = DiseaseClassificationLoss(
            num_classes=num_diseases,
            use_focal_loss=use_focal_loss
        )
        
        self.report_loss_fn = ReportGenerationLoss(
            vocab_size=vocab_size
        )
        
        self.consistency_loss_fn = ConsistencyLoss(
            weight=consistency_weight
        )
    
    def forward(self, outputs: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute all losses
        
        Args:
            outputs: Model outputs dictionary
            batch: Batch data dictionary
        
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # Disease classification loss
        if 'disease_logits' in outputs and 'chexpert_labels' in batch:
            disease_loss = self.disease_loss_fn(
                outputs['disease_logits'],
                batch['chexpert_labels']
            )
            losses['disease_loss'] = disease_loss
        else:
            disease_loss = 0
        
        # Report generation loss
        if 'report_logits' in outputs and 'report_input_ids' in batch:
            report_loss = self.report_loss_fn(
                outputs['report_logits'],
                batch['report_input_ids'],
                batch.get('report_attention_mask')
            )
            losses['report_loss'] = report_loss
        else:
            report_loss = 0
        
        # Consistency loss
        if all(k in outputs for k in ['visual_features', 'disease_output']):
            consistency_loss = self.consistency_loss_fn(
                outputs['disease_output']['disease_probs'],
                outputs['visual_features'],
                outputs['disease_output']['intermediate_features']
            )
            losses['consistency_loss'] = consistency_loss
        else:
            consistency_loss = 0
        
        # Total loss
        total_loss = (
            self.disease_weight * disease_loss +
            self.report_weight * report_loss +
            consistency_loss
        )
        
        losses['total_loss'] = total_loss
        
        return losses


if __name__ == "__main__":
    # Test losses
    print("Testing loss functions...")
    
    batch_size = 4
    num_diseases = 14
    seq_len = 128
    vocab_size = 30522
    
    # Test disease classification loss
    disease_loss = DiseaseClassificationLoss(num_classes=num_diseases)
    logits = torch.randn(batch_size, num_diseases)
    targets = torch.randint(0, 2, (batch_size, num_diseases)).float()
    
    loss = disease_loss(logits, targets)
    print(f"Disease classification loss: {loss.item():.4f}")
    
    # Test report generation loss
    report_loss = ReportGenerationLoss(vocab_size=vocab_size)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss = report_loss(logits, targets)
    print(f"Report generation loss: {loss.item():.4f}")
    
    # Test combined loss
    combined_loss = CombinedLoss(num_diseases=num_diseases, vocab_size=vocab_size)
    
    outputs = {
        'disease_logits': torch.randn(batch_size, num_diseases),
        'disease_output': {
            'disease_probs': torch.sigmoid(torch.randn(batch_size, num_diseases)),
            'intermediate_features': torch.randn(batch_size, 512)
        },
        'visual_features': torch.randn(batch_size, 512),
        'report_logits': torch.randn(batch_size, seq_len, vocab_size)
    }
    
    batch = {
        'chexpert_labels': torch.randint(0, 2, (batch_size, num_diseases)).float(),
        'report_input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
        'report_attention_mask': torch.ones(batch_size, seq_len)
    }
    
    losses = combined_loss(outputs, batch)
    print(f"\nCombined losses:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")
    
    print("\n✅ Loss functions test passed!")

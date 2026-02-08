"""
PRO-FA: Hierarchical Visual Alignment Encoder
Extracts pixel-level, region-level, and organ-level features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import timm
from transformers import AutoModel


class VisualFeatureExtractor(nn.Module):
    """
    Multi-scale visual feature extraction using pretrained vision models
    Extracts features at different hierarchical levels
    """
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        freeze_layers: int = 0
    ):
        """
        Args:
            model_name: Name of pretrained model from timm
            pretrained: Whether to use pretrained weights
            freeze_layers: Number of initial layers to freeze
        """
        super().__init__()
        
        # Load backbone model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,  # Return intermediate features
            out_indices=(1, 2, 3, 4)  # 4 hierarchical levels
        )
        
        # Get feature dimensions for each level
        self.feature_dims = self.backbone.feature_info.channels()
        print(f"Feature dimensions: {self.feature_dims}")
        
        # Freeze initial layers if specified
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
    
    def _freeze_layers(self, num_layers: int):
        """Freeze initial layers for transfer learning"""
        count = 0
        for param in self.backbone.parameters():
            if count < num_layers:
                param.requires_grad = False
                count += 1
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            List of feature maps at different scales
            [feat1, feat2, feat3, feat4]
        """
        features = self.backbone(x)
        return features


class OntologyEmbedding(nn.Module):
    """
    RadLex ontology embeddings for anatomical concepts
    Maps visual features to medical concept space
    """
    
    def __init__(
        self,
        num_concepts: int = 50,
        embedding_dim: int = 512,
        visual_dim: int = 2048
    ):
        """
        Args:
            num_concepts: Number of anatomical concepts
            embedding_dim: Dimension of concept embeddings
            visual_dim: Dimension of visual features
        """
        super().__init__()
        
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        
        # Learnable concept embeddings
        self.concept_embeddings = nn.Embedding(num_concepts, embedding_dim)
        
        # Visual-to-concept projection
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Concept attention
        self.concept_attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(self, visual_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align visual features with medical concepts
        
        Args:
            visual_features: [B, N, D] visual feature sequence
        
        Returns:
            aligned_features: [B, N, D] concept-aligned features
            attention_weights: [B, N, num_concepts] attention scores
        """
        batch_size = visual_features.size(0)
        
        # Project visual features
        visual_proj = self.visual_projection(visual_features)  # [B, N, D]
        
        # Get concept embeddings
        concept_ids = torch.arange(
            self.num_concepts,
            device=visual_features.device
        ).unsqueeze(0).expand(batch_size, -1)  # [B, num_concepts]
        
        concepts = self.concept_embeddings(concept_ids)  # [B, num_concepts, D]
        
        # Attend to concepts
        aligned_features, attention_weights = self.concept_attention(
            query=visual_proj,
            key=concepts,
            value=concepts
        )
        
        return aligned_features, attention_weights


class HierarchicalFeatureFusion(nn.Module):
    """
    Fuses features from different hierarchical levels
    Pixel → Region → Organ
    """
    
    def __init__(
        self,
        feature_dims: List[int],
        hidden_dim: int = 512,
        output_dim: int = 768
    ):
        """
        Args:
            feature_dims: List of feature dimensions from each level
            hidden_dim: Hidden dimension for fusion
            output_dim: Output feature dimension
        """
        super().__init__()
        
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Projection layers for each level
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            )
            for dim in feature_dims
        ])
        
        # Attention weights for each level
        self.level_attention = nn.Sequential(
            nn.Linear(hidden_dim * len(feature_dims), len(feature_dims)),
            nn.Softmax(dim=1)
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-scale features
        
        Args:
            multi_scale_features: List of feature maps at different scales
        
        Returns:
            fused_features: [B, output_dim, H, W]
        """
        batch_size = multi_scale_features[0].size(0)
        
        # Project all features to same dimension and size
        target_size = multi_scale_features[-1].shape[2:]  # Use smallest size
        
        projected = []
        for feat, proj in zip(multi_scale_features, self.projections):
            # Project channels
            feat_proj = proj(feat)  # [B, hidden_dim, H, W]
            
            # Resize to target size
            if feat_proj.shape[2:] != target_size:
                feat_proj = F.interpolate(
                    feat_proj,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            
            projected.append(feat_proj)
        
        # Stack features
        stacked = torch.stack(projected, dim=1)  # [B, num_levels, hidden_dim, H, W]
        
        # Global pool for attention computation
        pooled = F.adaptive_avg_pool2d(stacked, 1).squeeze(-1).squeeze(-1)  # [B, num_levels, hidden_dim]
        pooled_flat = pooled.view(batch_size, -1)  # [B, num_levels * hidden_dim]
        
        # Compute attention weights
        attention = self.level_attention(pooled_flat).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, num_levels, 1, 1, 1]
        
        # Weighted sum
        weighted = (stacked * attention).sum(dim=1)  # [B, hidden_dim, H, W]
        
        # Global average pooling
        pooled_features = F.adaptive_avg_pool2d(weighted, 1).squeeze(-1).squeeze(-1)  # [B, hidden_dim]
        
        # Final fusion
        fused = self.fusion(pooled_features)  # [B, output_dim]
        
        return fused


class PROFAEncoder(nn.Module):
    """
    PRO-FA: Hierarchical Visual Alignment Encoder
    Implements pixel-level, region-level, and organ-level perception
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        num_concepts: int = 50,
        hidden_dim: int = 512,
        output_dim: int = 768,
        pretrained: bool = True
    ):
        """
        Args:
            backbone: Backbone CNN architecture
            num_concepts: Number of RadLex concepts
            hidden_dim: Hidden dimension
            output_dim: Output feature dimension
            pretrained: Use pretrained weights
        """
        super().__init__()
        
        # Visual feature extractor
        self.visual_extractor = VisualFeatureExtractor(
            model_name=backbone,
            pretrained=pretrained
        )
        
        feature_dims = self.visual_extractor.feature_dims
        
        # Hierarchical fusion
        self.hierarchical_fusion = HierarchicalFeatureFusion(
            feature_dims=feature_dims,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        # Ontology alignment
        self.ontology_embedding = OntologyEmbedding(
            num_concepts=num_concepts,
            embedding_dim=output_dim,
            visual_dim=output_dim
        )
        
        self.output_dim = output_dim
    
    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: Input images [B, 3, H, W]
            return_attention: Whether to return attention maps
        
        Returns:
            Dictionary containing:
                - visual_features: [B, output_dim] global visual features
                - aligned_features: [B, 1, output_dim] ontology-aligned features
                - attention_maps: Optional attention weights
        """
        # Extract multi-scale features
        multi_scale_features = self.visual_extractor(images)
        
        # Fuse hierarchical features
        fused_features = self.hierarchical_fusion(multi_scale_features)  # [B, output_dim]
        
        # Align with medical ontology
        # Add sequence dimension for attention
        fused_seq = fused_features.unsqueeze(1)  # [B, 1, output_dim]
        aligned_features, attention_weights = self.ontology_embedding(fused_seq)
        
        output = {
            'visual_features': fused_features,
            'aligned_features': aligned_features.squeeze(1),  # [B, output_dim]
            'multi_scale_features': multi_scale_features  # For visualization
        }
        
        if return_attention:
            output['concept_attention'] = attention_weights
        
        return output


if __name__ == "__main__":
    # Test encoder
    print("Testing PRO-FA Encoder...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = PROFAEncoder(
        backbone='resnet50',
        num_concepts=50,
        output_dim=768
    ).to(device)
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    
    output = encoder(images, return_attention=True)
    
    print("\nOutput shapes:")
    print(f"Visual features: {output['visual_features'].shape}")
    print(f"Aligned features: {output['aligned_features'].shape}")
    print(f"Concept attention: {output['concept_attention'].shape}")
    
    print("\n✅ PRO-FA Encoder test passed!")

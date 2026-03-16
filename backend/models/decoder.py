"""
RCTA: Triangular Cognitive Attention Decoder
Implements closed-loop reasoning for report generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, GPT2Config


class TriangularAttention(nn.Module):
    """
    Implements triangular attention mechanism:
    Image → Text → Diagnosis → Image (closed loop)
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Dimension of hidden states
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Image queries Text (context formation)
        self.image_text_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Context queries Diagnosis (hypothesis creation)
        self.context_diagnosis_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Hypothesis queries Image (visual verification)
        self.hypothesis_image_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Gate mechanism for adaptive fusion
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        diagnosis_features: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Triangular attention forward pass
        
        Args:
            image_features: [B, N_img, D] visual features
            text_features: [B, N_text, D] clinical text features
            diagnosis_features: [B, N_diag, D] predicted disease features
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary with fused features and optional attention maps
        """
        # Stage 1: Image queries Text (context formation)
        context_features, attn_img_text = self.image_text_attention(
            query=image_features,
            key=text_features,
            value=text_features
        )
        
        # Stage 2: Context queries Diagnosis (hypothesis creation)
        hypothesis_features, attn_ctx_diag = self.context_diagnosis_attention(
            query=context_features,
            key=diagnosis_features,
            value=diagnosis_features
        )
        
        # Stage 3: Hypothesis queries Image (visual verification)
        verified_features, attn_hyp_img = self.hypothesis_image_attention(
            query=hypothesis_features,
            key=image_features,
            value=image_features
        )
        
        # Concatenate all stages
        combined = torch.cat([
            context_features,
            hypothesis_features,
            verified_features
        ], dim=-1)  # [B, N, 3*D]
        
        # Compute adaptive gates
        gates = self.gate(combined)  # [B, N, 3]
        
        # Apply gated fusion
        gated_context = context_features * gates[..., 0:1]
        gated_hypothesis = hypothesis_features * gates[..., 1:2]
        gated_verified = verified_features * gates[..., 2:3]
        
        # Final fusion
        fused = self.fusion(torch.cat([
            gated_context,
            gated_hypothesis,
            gated_verified
        ], dim=-1))
        
        output = {
            'fused_features': fused,
            'context_features': context_features,
            'hypothesis_features': hypothesis_features,
            'verified_features': verified_features
        }
        
        if return_attention:
            output['attention_weights'] = {
                'image_text': attn_img_text,
                'context_diagnosis': attn_ctx_diag,
                'hypothesis_image': attn_hyp_img
            }
        
        return output


class RCTADecoder(nn.Module):
    """
    RCTA Decoder with triangular attention and report generation
    """
    
    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 768,
        vocab_size: int = 30522,
        max_length: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            visual_dim: Dimension of visual features
            text_dim: Dimension of text features
            hidden_dim: Hidden dimension
            vocab_size: Vocabulary size
            max_length: Maximum sequence length
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Feature projections
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.diagnosis_proj = nn.Linear(visual_dim, hidden_dim)
        
        # Triangular attention
        self.triangular_attention = TriangularAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_length, dropout)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        diagnosis_features: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        target_embeddings: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            visual_features: [B, visual_dim] visual features
            text_features: [B, text_dim] clinical indication features
            diagnosis_features: [B, visual_dim] disease prediction features
            target_ids: [B, seq_len] target token IDs (for training)
            target_embeddings: [B, seq_len, hidden_dim] target embeddings
            return_attention: Whether to return attention weights
        
        Returns:
            Dictionary with logits and optional attention maps
        """
        batch_size = visual_features.size(0)
        
        # Project features
        visual_proj = self.visual_proj(visual_features).unsqueeze(1)  # [B, 1, hidden_dim]
        text_proj = self.text_proj(text_features).unsqueeze(1)  # [B, 1, hidden_dim]
        diagnosis_proj = self.diagnosis_proj(diagnosis_features).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Apply triangular attention
        triangular_output = self.triangular_attention(
            image_features=visual_proj,
            text_features=text_proj,
            diagnosis_features=diagnosis_proj,
            return_attention=return_attention
        )
        
        memory = triangular_output['fused_features']  # [B, 1, hidden_dim]
        
        # Prepare target sequence
        if target_embeddings is not None:
            tgt = target_embeddings
        elif target_ids is not None:
            # This would need an embedding layer
            raise NotImplementedError("Token embedding not implemented here")
        else:
            # Inference mode - will be handled by generate method
            return {
                'memory': memory,
                'triangular_output': triangular_output
            }
        
        # Add positional encoding
        tgt = self.pos_encoder(tgt)
        
        # Create causal mask
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Decode
        decoder_output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        # Project to vocabulary
        logits = self.output_proj(decoder_output)
        
        output = {
            'logits': logits,
            'memory': memory
        }
        
        if return_attention:
            output['triangular_attention'] = triangular_output.get('attention_weights')
        
        return output
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ReportGenerator(nn.Module):
    """
    End-to-end report generator using pretrained language model
    """
    
    def __init__(
        self,
        encoder_dim: int = 768,
        lm_model: str = "gpt2",
        max_length: int = 512,
        num_beams: int = 4,
        temperature: float = 1.0
    ):
        """
        Args:
            encoder_dim: Dimension of encoder features
            lm_model: Pretrained language model name
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
        """
        super().__init__()
        
        self.max_length = max_length
        self.num_beams = num_beams
        self.temperature = temperature
        
        # Load pretrained language model
        self.lm_config = GPT2Config.from_pretrained(lm_model)
        self.lm = GPT2LMHeadModel.from_pretrained(lm_model)
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Projection from encoder to LM space
        self.encoder_proj = nn.Sequential(
            nn.Linear(encoder_dim, self.lm_config.n_embd),
            nn.LayerNorm(self.lm_config.n_embd),
            nn.Tanh()
        )
    
    def forward(
        self,
        encoder_features: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training
        
        Args:
            encoder_features: [B, encoder_dim] encoder output
            target_ids: [B, seq_len] target token IDs
            attention_mask: [B, seq_len] attention mask
        
        Returns:
            Dictionary with loss and logits
        """
        # Project encoder features
        encoder_emb = self.encoder_proj(encoder_features).unsqueeze(1)  # [B, 1, n_embd]
        
        if target_ids is not None:
            # Training mode
            # Get token embeddings
            target_emb = self.lm.transformer.wte(target_ids)  # [B, seq_len, n_embd]
            
            # Concatenate encoder features with target embeddings
            inputs_embeds = torch.cat([encoder_emb, target_emb], dim=1)  # [B, 1+seq_len, n_embd]
            
            # Create attention mask
            if attention_mask is not None:
                # Add 1 for encoder feature
                encoder_mask = torch.ones(
                    encoder_emb.size(0), 1,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([encoder_mask, attention_mask], dim=1)
            
            # Forward through LM
            outputs = self.lm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=target_ids  # Shifted internally by model
            )
            
            return {
                'loss': outputs.loss,
                'logits': outputs.logits
            }
        else:
            # Return encoded features for generation
            return {
                'encoder_embeddings': encoder_emb
            }
    
    @torch.no_grad()
    def generate(
        self,
        encoder_features: torch.Tensor,
        prompt: Optional[str] = None,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        do_sample: bool = False
    ) -> Tuple[str, torch.Tensor]:
        """
        Generate report text
        
        Args:
            encoder_features: [B, encoder_dim] encoder output
            prompt: Optional text prompt to condition generation
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
        
        Returns:
            generated_text: Generated report
            generated_ids: Token IDs
        """
        max_length = max_length or self.max_length
        num_beams = num_beams or self.num_beams
        temperature = temperature or self.temperature
        
        batch_size = encoder_features.size(0)
        
        # Project encoder features
        encoder_emb = self.encoder_proj(encoder_features).unsqueeze(1)  # [B, 1, n_embd]
        
        # Prepare prompt
        if prompt is None:
            prompt = "FINDINGS:"
        
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(encoder_features.device)
        prompt_ids = prompt_ids.repeat(batch_size, 1)  # [B, prompt_len]
        
        # Get prompt embeddings
        prompt_emb = self.lm.transformer.wte(prompt_ids)  # [B, prompt_len, n_embd]
        
        # Concatenate encoder and prompt embeddings
        inputs_embeds = torch.cat([encoder_emb, prompt_emb], dim=1)  # [B, 1+prompt_len, n_embd]
        
        # Generate
        outputs = self.lm.generate(
            inputs_embeds=inputs_embeds,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            early_stopping=True
        )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text, outputs


if __name__ == "__main__":
    # Test decoder
    print("Testing RCTA Decoder...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test triangular attention
    triangular_attn = TriangularAttention(hidden_dim=768).to(device)
    
    batch_size = 2
    image_feat = torch.randn(batch_size, 1, 768).to(device)
    text_feat = torch.randn(batch_size, 1, 768).to(device)
    diag_feat = torch.randn(batch_size, 1, 768).to(device)
    
    tri_output = triangular_attn(image_feat, text_feat, diag_feat, return_attention=True)
    print(f"Triangular attention output: {tri_output['fused_features'].shape}")
    
    # Test report generator
    print("\nTesting Report Generator...")
    generator = ReportGenerator(encoder_dim=768).to(device)
    
    encoder_feat = torch.randn(batch_size, 768).to(device)
    
    # Test generation
    text, ids = generator.generate(encoder_feat, max_length=50)
    print(f"Generated text: {text[:100]}...")
    
    print("\n✅ RCTA Decoder test passed!")

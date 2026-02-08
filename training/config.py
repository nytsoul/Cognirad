"""
Training Configuration
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingConfig:
    """Configuration for training CogniRad++"""
    
    # Model architecture
    visual_backbone: str = 'resnet50'
    text_encoder: str = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    num_concepts: int = 50
    num_diseases: int = 14
    encoder_output_dim: int = 768
    classifier_hidden_dim: int = 512
    
    # Data
    train_data_file: str = './data/preprocessed/train.json'
    val_data_file: str = './data/preprocessed/validate.json'
    data_root: str = './data/mimic-cxr'
    max_length: int = 512
    img_size: int = 224
    
    # Training hyperparameters
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    
    # Loss weights
    disease_loss_weight: float = 1.0
    report_loss_weight: float = 2.0
    consistency_loss_weight: float = 0.5
    use_focal_loss: bool = True
    
    # Optimization
    optimizer: str = 'adamw'  # 'adamw', 'adam', 'sgd'
    scheduler: str = 'cosine'  # 'cosine', 'linear', 'constant'
    lr_scheduler_patience: int = 5
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.0  # 0 to disable
    
    # Training control
    freeze_encoder_epochs: int = 0  # Freeze encoder for first N epochs
    early_stopping_patience: int = 10
    save_every_n_epochs: int = 5
    validate_every_n_steps: int = 500
    
    # Generation
    num_beams: int = 4
    max_gen_length: int = 256
    temperature: float = 1.0
    
    # Paths
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    output_dir: str = './outputs'
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = 'cognirad-plusplus'
    wandb_entity: Optional[str] = None
    log_every_n_steps: int = 100
    save_samples: bool = True
    num_samples_to_save: int = 5
    
    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True  # Use AMP
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Distributed training
    local_rank: int = -1
    world_size: int = 1
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert 0 <= self.dropout < 1, "Dropout must be in [0, 1)"
        assert self.num_epochs > 0, "Number of epochs must be positive"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    
    # Data
    test_data_file: str = './data/preprocessed/test.json'
    data_root: str = './data/mimic-cxr'
    batch_size: int = 16
    num_workers: int = 4
    
    # Model
    checkpoint_path: str = './checkpoints/best_model.pt'
    
    # Generation
    num_beams: int = 4
    max_length: int = 512
    temperature: float = 1.0
    
    # Evaluation metrics
    compute_clinical_efficacy: bool = True
    compute_nlg_metrics: bool = True
    compute_chexpert_f1: bool = True
    compute_radgraph_f1: bool = True
    
    # Output
    output_dir: str = './evaluation_results'
    save_predictions: bool = True
    save_attention_maps: bool = False
    
    # Device
    device: str = 'cuda'  # 'cuda' or 'cpu'


# Preset configurations
def get_small_config() -> TrainingConfig:
    """Small model for testing"""
    return TrainingConfig(
        visual_backbone='resnet18',
        batch_size=8,
        num_epochs=10,
        encoder_output_dim=512,
        classifier_hidden_dim=256
    )


def get_base_config() -> TrainingConfig:
    """Base model configuration"""
    return TrainingConfig(
        visual_backbone='resnet50',
        batch_size=16,
        num_epochs=100,
        encoder_output_dim=768,
        classifier_hidden_dim=512
    )


def get_large_config() -> TrainingConfig:
    """Large model configuration"""
    return TrainingConfig(
        visual_backbone='resnet101',
        batch_size=8,
        num_epochs=150,
        encoder_output_dim=1024,
        classifier_hidden_dim=768,
        accumulation_steps=2
    )


if __name__ == "__main__":
    # Test configurations
    print("Testing configurations...")
    
    configs = {
        'small': get_small_config(),
        'base': get_base_config(),
        'large': get_large_config()
    }
    
    for name, config in configs.items():
        print(f"\n{name.upper()} Configuration:")
        print(f"  Backbone: {config.visual_backbone}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Encoder dim: {config.encoder_output_dim}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Epochs: {config.num_epochs}")
    
    print("\n✅ Configuration test passed!")

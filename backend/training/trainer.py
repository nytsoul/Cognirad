"""
Training Loop for CogniRad++
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
from pathlib import Path
from typing import Dict, Optional
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.cognirad import CogniRadPlusPlus
from data.dataset import CXRDataset, collate_fn
from .losses import CombinedLoss
from .config import TrainingConfig


class CogniRadTrainer:
    """Trainer for CogniRad++ model"""
    
    def __init__(self, config: TrainingConfig):
        """
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seed
        self._set_seed(config.seed)
        
        # Initialize model
        print("Initializing model...")
        self.model = CogniRadPlusPlus(
            visual_backbone=config.visual_backbone,
            text_encoder=config.text_encoder,
            num_concepts=config.num_concepts,
            num_diseases=config.num_diseases,
            encoder_output_dim=config.encoder_output_dim,
            classifier_hidden_dim=config.classifier_hidden_dim,
            pretrained=True
        ).to(self.device)
        
        # Initialize loss function
        self.criterion = CombinedLoss(
            num_diseases=config.num_diseases,
            vocab_size=self.model.report_generator.lm.config.vocab_size,
            disease_weight=config.disease_loss_weight,
            report_weight=config.report_loss_weight,
            consistency_weight=config.consistency_loss_weight,
            use_focal_loss=config.use_focal_loss
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = None  # Will be created after knowing dataset size
        
        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=vars(config),
                name=f"cognirad_{config.visual_backbone}"
            )
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        # Separate parameters for different learning rates
        encoder_params = list(self.model.visual_encoder.parameters()) + \
                        list(self.model.text_encoder.parameters())
        
        other_params = list(self.model.disease_classifier.parameters()) + \
                       list(self.model.report_generator.parameters())
        
        param_groups = [
            {'params': encoder_params, 'lr': self.config.learning_rate * 0.1},  # Lower LR for pretrained
            {'params': other_params, 'lr': self.config.learning_rate}
        ]
        
        if self.config.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler"""
        if self.config.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=1e-6
            )
        elif self.config.scheduler == 'linear':
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.config.scheduler == 'constant':
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0
            )
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("Starting CogniRad++ Training")
        print("="*70)
        
        # Load datasets
        print("\nLoading datasets...")
        train_dataset = CXRDataset(
            data_file=self.config.train_data_file,
            data_root=self.config.data_root,
            split='train'
        )
        
        val_dataset = CXRDataset(
            data_file=self.config.val_data_file,
            data_root=self.config.data_root,
            split='validate'
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # Create scheduler
        num_training_steps = len(train_loader) * self.config.num_epochs // self.config.accumulation_steps
        self._create_scheduler(num_training_steps)
        
        # Training loop
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Freeze encoder for first N epochs
            if epoch < self.config.freeze_encoder_epochs:
                self._freeze_encoder(True)
            else:
                self._freeze_encoder(False)
            
            # Train one epoch
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self._validate(val_loader, epoch)
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, is_best=False)
            
            # Check for best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self._save_checkpoint(epoch, is_best=True)
                self.patience_counter = 0
                print(f"✅ New best model! Val loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                break
        
        print("\n" + "="*70)
        print("Training completed!")
        print("="*70)
        
        if self.config.use_wandb:
            wandb.finish()
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        disease_loss_sum = 0
        report_loss_sum = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if self.config.mixed_precision:
                with autocast():
                    outputs = self.model(
                        images=batch['image'],
                        indication_input_ids=batch['indication_input_ids'],
                        indication_attention_mask=batch['indication_attention_mask'],
                        report_input_ids=batch['report_input_ids'],
                        report_attention_mask=batch['report_attention_mask'],
                        chexpert_labels=batch['chexpert_labels']
                    )
                    
                    losses = outputs['losses']
                    loss = losses['total_loss'] / self.config.accumulation_steps
            else:
                outputs = self.model(
                    images=batch['image'],
                    indication_input_ids=batch['indication_input_ids'],
                    indication_attention_mask=batch['indication_attention_mask'],
                    report_input_ids=batch['report_input_ids'],
                    report_attention_mask=batch['report_attention_mask'],
                    chexpert_labels=batch['chexpert_labels']
                )
                
                losses = outputs['losses']
                loss = losses['total_loss'] / self.config.accumulation_steps
            
            # Backward pass
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (step + 1) % self.config.accumulation_steps == 0:
                if self.config.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Track metrics
            total_loss += loss.item() * self.config.accumulation_steps
            if 'disease_loss' in losses:
                disease_loss_sum += losses['disease_loss'].item()
            if 'report_loss' in losses:
                report_loss_sum += losses['report_loss'].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item() * self.config.accumulation_steps,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log to wandb
            if self.config.use_wandb and step % self.config.log_every_n_steps == 0:
                wandb.log({
                    'train/loss': loss.item() * self.config.accumulation_steps,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': epoch,
                    'train/step': self.global_step
                })
        
        # Compute average metrics
        num_batches = len(train_loader)
        metrics = {
            'total_loss': total_loss / num_batches,
            'disease_loss': disease_loss_sum / num_batches,
            'report_loss': report_loss_sum / num_batches
        }
        
        return metrics
    
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        disease_loss_sum = 0
        report_loss_sum = 0
        
        for batch in tqdm(val_loader, desc="Validating"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                images=batch['image'],
                indication_input_ids=batch['indication_input_ids'],
                indication_attention_mask=batch['indication_attention_mask'],
                report_input_ids=batch['report_input_ids'],
                report_attention_mask=batch['report_attention_mask'],
                chexpert_labels=batch['chexpert_labels']
            )
            
            losses = outputs['losses']
            
            total_loss += losses['total_loss'].item()
            if 'disease_loss' in losses:
                disease_loss_sum += losses['disease_loss'].item()
            if 'report_loss' in losses:
                report_loss_sum += losses['report_loss'].item()
        
        # Compute average metrics
        num_batches = len(val_loader)
        metrics = {
            'total_loss': total_loss / num_batches,
            'disease_loss': disease_loss_sum / num_batches,
            'report_loss': report_loss_sum / num_batches
        }
        
        return metrics
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics"""
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['total_loss']:.4f}")
        
        if self.config.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/total_loss_epoch': train_metrics['total_loss'],
                'train/disease_loss_epoch': train_metrics['disease_loss'],
                'train/report_loss_epoch': train_metrics['report_loss'],
                'val/total_loss': val_metrics['total_loss'],
                'val/disease_loss': val_metrics['disease_loss'],
                'val/report_loss': val_metrics['report_loss']
            })
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config)
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model to {best_path}")
    
    def _freeze_encoder(self, freeze: bool):
        """Freeze or unfreeze encoder"""
        for param in self.model.visual_encoder.parameters():
            param.requires_grad = not freeze
        for param in self.model.text_encoder.parameters():
            param.requires_grad = not freeze


def main():
    """Main training script"""
    from .config import get_base_config
    
    # Get configuration
    config = get_base_config()
    
    # Override with command line arguments if needed
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    
    # Create trainer
    trainer = CogniRadTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

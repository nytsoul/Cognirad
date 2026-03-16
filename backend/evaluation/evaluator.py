"""
Comprehensive Evaluator for CogniRad++
"""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json
from typing import Dict, List
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.cognirad import CogniRadPlusPlus
from data.dataset import CXRDataset, collate_fn
from .metrics import (
    compute_chexpert_f1,
    compute_radgraph_f1,
    compute_nlg_metrics,
    compute_clinical_efficacy,
    compute_confidence_calibration
)


class CogniRadEvaluator:
    """Comprehensive evaluator for CogniRad++"""
    
    def __init__(
        self,
        checkpoint_path: str,
        test_data_file: str,
        data_root: str,
        output_dir: str = './evaluation_results',
        device: str = None
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            test_data_file: Path to test data JSON
            data_root: Root directory for images
            output_dir: Directory to save results
            device: Device to use ('cuda' or 'cpu')
        """
        self.checkpoint_path = checkpoint_path
        self.test_data_file = test_data_file
        self.data_root = data_root
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading model...")
        self.model = self._load_model()
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def _load_model(self) -> CogniRadPlusPlus:
        """Load model from checkpoint"""
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Get config
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Use default config
            config = {
                'visual_backbone': 'resnet50',
                'text_encoder': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                'num_concepts': 50,
                'num_diseases': 14,
                'encoder_output_dim': 768,
                'classifier_hidden_dim': 512
            }
        
        # Create model
        model = CogniRadPlusPlus(
            visual_backbone=config.get('visual_backbone', 'resnet50'),
            text_encoder=config.get('text_encoder', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'),
            num_concepts=config.get('num_concepts', 50),
            num_diseases=config.get('num_diseases', 14),
            encoder_output_dim=config.get('encoder_output_dim', 768),
            classifier_hidden_dim=config.get('classifier_hidden_dim', 512),
            pretrained=False
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    @torch.no_grad()
    def evaluate(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        save_predictions: bool = True
    ) -> Dict[str, float]:
        """
        Run comprehensive evaluation
        
        Args:
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            save_predictions: Whether to save predictions
        
        Returns:
            Dictionary with all evaluation metrics
        """
        print("\n" + "="*70)
        print("Starting CogniRad++ Evaluation")
        print("="*70)
        
        # Load test dataset
        print("\nLoading test dataset...")
        test_dataset = CXRDataset(
            data_file=self.test_data_file,
            data_root=self.data_root,
            split='test'
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        print(f"Test samples: {len(test_dataset)}")
        
        # Storage for predictions and references
        all_disease_preds = []
        all_disease_targets = []
        all_confidence = []
        all_generated_findings = []
        all_generated_impressions = []
        all_reference_findings = []
        all_reference_impressions = []
        all_predicted_diseases = []
        
        # Evaluate
        print("\nEvaluating...")
        for batch in tqdm(test_loader, desc="Testing"):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                images=batch['image'],
                indication_input_ids=batch['indication_input_ids'],
                indication_attention_mask=batch['indication_attention_mask'],
                chexpert_labels=batch['chexpert_labels']
            )
            
            # Generate reports
            for i in range(batch['image'].size(0)):
                report = self.model.generate_report(
                    images=batch['image'][i:i+1],
                    clinical_indication="Chest X-ray",
                    include_evidence=False
                )
                
                all_generated_findings.append(report['findings'])
                all_generated_impressions.append(report['impression'])
                all_predicted_diseases.append(report['predicted_diseases'])
            
            # Store predictions
            all_disease_preds.append(outputs['disease_probs'])
            all_disease_targets.append(batch['chexpert_labels'])
            all_confidence.append(outputs['confidence'])
            all_reference_findings.extend(batch['findings'])
            all_reference_impressions.extend(batch['impression'])
        
        # Concatenate results
        disease_preds = torch.cat(all_disease_preds, dim=0)
        disease_targets = torch.cat(all_disease_targets, dim=0)
        confidence = torch.cat(all_confidence, dim=0)
        
        # Compute metrics
        print("\nComputing metrics...")
        metrics = {}
        
        # 1. Clinical Accuracy (CheXpert F1)
        print("  - CheXpert F1...")
        chexpert_metrics = compute_chexpert_f1(disease_preds, disease_targets)
        metrics.update({f'chexpert_{k}': v for k, v in chexpert_metrics.items()})
        
        # 2. Structural Logic (RadGraph F1)
        print("  - RadGraph F1...")
        # Combine findings and impression
        gen_reports = [f + " " + i for f, i in zip(all_generated_findings, all_generated_impressions)]
        ref_reports = [f + " " + i for f, i in zip(all_reference_findings, all_reference_impressions)]
        
        radgraph_metrics = compute_radgraph_f1(gen_reports, ref_reports)
        metrics.update({f'radgraph_{k}': v for k, v in radgraph_metrics.items()})
        
        # 3. Language Quality (NLG metrics)
        print("  - NLG metrics...")
        nlg_metrics = compute_nlg_metrics(gen_reports, ref_reports)
        metrics.update({f'nlg_{k}': v for k, v in nlg_metrics.items()})
        
        # 4. Clinical Efficacy
        print("  - Clinical efficacy...")
        efficacy_metrics = compute_clinical_efficacy(
            gen_reports,
            ref_reports,
            all_predicted_diseases,
            []  # Ground truth diseases not available in this format
        )
        metrics.update({f'efficacy_{k}': v for k, v in efficacy_metrics.items()})
        
        # 5. Confidence Calibration
        print("  - Confidence calibration...")
        calib_metrics = compute_confidence_calibration(
            disease_preds,
            confidence,
            disease_targets
        )
        metrics.update({f'calibration_{k}': v for k, v in calib_metrics.items()})
        
        # Print results
        self._print_results(metrics)
        
        # Save metrics
        metrics_file = self.output_dir / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 Saved metrics to {metrics_file}")
        
        # Save predictions
        if save_predictions:
            self._save_predictions(
                all_generated_findings,
                all_generated_impressions,
                all_reference_findings,
                all_reference_impressions,
                disease_preds,
                disease_targets,
                all_predicted_diseases
            )
        
        return metrics
    
    def _print_results(self, metrics: Dict[str, float]):
        """Print evaluation results"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        # Clinical Accuracy (40%)
        print("\n📊 Clinical Accuracy (CheXpert F1):")
        print(f"  Micro F1:     {metrics.get('chexpert_f1_micro', 0):.4f}")
        print(f"  Macro F1:     {metrics.get('chexpert_f1_macro', 0):.4f}")
        print(f"  Precision:    {metrics.get('chexpert_precision_macro', 0):.4f}")
        print(f"  Recall:       {metrics.get('chexpert_recall_macro', 0):.4f}")
        print(f"  AUC:          {metrics.get('chexpert_auc_macro', 0):.4f}")
        
        # Structural Logic (30%)
        print("\n🏗️  Structural Logic (RadGraph F1):")
        print(f"  F1 Score:     {metrics.get('radgraph_radgraph_f1_simple', 0):.4f}")
        print(f"  Precision:    {metrics.get('radgraph_radgraph_precision_simple', 0):.4f}")
        print(f"  Recall:       {metrics.get('radgraph_radgraph_recall_simple', 0):.4f}")
        
        # Language Quality (30%)
        print("\n✍️  Language Quality (NLG Metrics):")
        print(f"  BLEU-4:       {metrics.get('nlg_bleu_4', 0):.4f}")
        print(f"  METEOR:       {metrics.get('nlg_meteor', 0):.4f}")
        print(f"  ROUGE-L:      {metrics.get('nlg_rouge_l', 0):.4f}")
        print(f"  CIDEr:        {metrics.get('nlg_cider', 0):.4f}")
        
        # Clinical Efficacy
        print("\n🏥 Clinical Efficacy:")
        print(f"  Hallucination Rate: {metrics.get('efficacy_hallucination_rate', 0):.4f}")
        print(f"  Completeness:       {metrics.get('efficacy_completeness', 0):.4f}")
        
        # Confidence Calibration
        print("\n📈 Confidence Calibration:")
        print(f"  ECE:          {metrics.get('calibration_expected_calibration_error', 0):.4f}")
        print(f"  Mean Conf:    {metrics.get('calibration_mean_confidence', 0):.4f}")
        print(f"  Mean Acc:     {metrics.get('calibration_mean_accuracy', 0):.4f}")
        
        # Weighted Score
        weighted_score = (
            0.4 * metrics.get('chexpert_f1_macro', 0) +
            0.3 * metrics.get('radgraph_radgraph_f1_simple', 0) +
            0.3 * metrics.get('nlg_cider', 0) / 10  # Normalize CIDEr
        )
        print(f"\n⭐ Weighted Score: {weighted_score:.4f}")
        print("="*70)
    
    def _save_predictions(
        self,
        generated_findings: List[str],
        generated_impressions: List[str],
        reference_findings: List[str],
        reference_impressions: List[str],
        disease_preds: torch.Tensor,
        disease_targets: torch.Tensor,
        predicted_diseases: List[Dict]
    ):
        """Save predictions to CSV"""
        # Create DataFrame
        df = pd.DataFrame({
            'generated_findings': generated_findings,
            'generated_impression': generated_impressions,
            'reference_findings': reference_findings,
            'reference_impression': reference_impressions
        })
        
        # Save
        predictions_file = self.output_dir / 'predictions.csv'
        df.to_csv(predictions_file, index=False)
        print(f"💾 Saved predictions to {predictions_file}")
        
        # Save disease predictions
        disease_file = self.output_dir / 'disease_predictions.pt'
        torch.save({
            'predictions': disease_preds,
            'targets': disease_targets
        }, disease_file)
        print(f"💾 Saved disease predictions to {disease_file}")


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CogniRad++")
    parser.add_argument('--checkpoint', type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument('--test_data', type=str, required=True,
                       help="Path to test data JSON")
    parser.add_argument('--data_root', type=str, required=True,
                       help="Root directory for images")
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help="Output directory for results")
    parser.add_argument('--batch_size', type=int, default=16,
                       help="Batch size")
    parser.add_argument('--num_workers', type=int, default=4,
                       help="Number of workers")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = CogniRadEvaluator(
        checkpoint_path=args.checkpoint,
        test_data_file=args.test_data,
        data_root=args.data_root,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    metrics = evaluator.evaluate(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_predictions=True
    )


if __name__ == "__main__":
    main()

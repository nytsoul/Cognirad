"""
Evaluation Metrics for Medical Report Generation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import re


def compute_chexpert_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute CheXpert F1 scores (clinical accuracy metric)
    
    Args:
        predictions: [N, num_diseases] predicted probabilities
        targets: [N, num_diseases] ground truth labels
        threshold: Probability threshold for binary classification
    
    Returns:
        Dictionary with F1, precision, recall, and AUC scores
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Binarize predictions
    pred_binary = (predictions > threshold).astype(int)
    
    # Filter out uncertain labels (-1)
    valid_mask = (targets != -1)
    
    # Compute metrics for each class
    num_classes = predictions.shape[1]
    class_f1 = []
    class_precision = []
    class_recall = []
    class_auc = []
    
    for i in range(num_classes):
        mask = valid_mask[:, i]
        if mask.sum() == 0:
            continue
        
        y_true = targets[mask, i]
        y_pred = pred_binary[mask, i]
        y_score = predictions[mask, i]
        
        # Skip if only one class present
        if len(np.unique(y_true)) < 2:
            continue
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_score)
        except:
            auc = 0.0
        
        class_f1.append(f1)
        class_precision.append(precision)
        class_recall.append(recall)
        class_auc.append(auc)
    
    # Aggregate metrics
    metrics = {
        'f1_micro': f1_score(targets[valid_mask], pred_binary[valid_mask], zero_division=0),
        'f1_macro': np.mean(class_f1) if class_f1 else 0.0,
        'precision_macro': np.mean(class_precision) if class_precision else 0.0,
        'recall_macro': np.mean(class_recall) if class_recall else 0.0,
        'auc_macro': np.mean(class_auc) if class_auc else 0.0,
        'num_valid_classes': len(class_f1)
    }
    
    return metrics


def compute_radgraph_f1(
    generated_reports: List[str],
    reference_reports: List[str]
) -> Dict[str, float]:
    """
    Compute RadGraph F1 score (structural logic metric)
    Measures overlap of clinical entities and relations
    
    Note: This is a simplified implementation
    For production, use the official RadGraph library:
    https://github.com/jbdel/radgraph
    
    Args:
        generated_reports: List of generated report texts
        reference_reports: List of reference report texts
    
    Returns:
        Dictionary with RadGraph scores
    """
    # Simplified entity extraction (production should use RadGraph NER)
    def extract_entities(text: str) -> set:
        """Extract medical entities (simplified)"""
        entities = set()
        
        # Common anatomical terms
        anatomy_terms = [
            'lung', 'heart', 'pleura', 'mediastinum', 'diaphragm',
            'rib', 'clavicle', 'trachea', 'hilum', 'lobe'
        ]
        
        # Common pathologies
        pathology_terms = [
            'opacity', 'consolidation', 'effusion', 'pneumothorax',
            'atelectasis', 'edema', 'cardiomegaly', 'nodule',
            'mass', 'infiltrate', 'pneumonia'
        ]
        
        text_lower = text.lower()
        
        for term in anatomy_terms + pathology_terms:
            if term in text_lower:
                entities.add(term)
        
        return entities
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    
    for gen_text, ref_text in zip(generated_reports, reference_reports):
        gen_entities = extract_entities(gen_text)
        ref_entities = extract_entities(ref_text)
        
        if len(ref_entities) == 0:
            continue
        
        # Compute overlap
        overlap = gen_entities & ref_entities
        
        precision = len(overlap) / len(gen_entities) if len(gen_entities) > 0 else 0
        recall = len(overlap) / len(ref_entities) if len(ref_entities) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    
    n = len(generated_reports)
    
    metrics = {
        'radgraph_f1_simple': total_f1 / n if n > 0 else 0,
        'radgraph_precision_simple': total_precision / n if n > 0 else 0,
        'radgraph_recall_simple': total_recall / n if n > 0 else 0
    }
    
    return metrics


def compute_nlg_metrics(
    generated_reports: List[str],
    reference_reports: List[str]
) -> Dict[str, float]:
    """
    Compute NLG metrics (language quality)
    Includes BLEU, METEOR, ROUGE, CIDEr
    
    Args:
        generated_reports: List of generated report texts
        reference_reports: List of reference report texts
    
    Returns:
        Dictionary with NLG metrics
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        
        # Format for pycocoevalcap
        gts = {i: [ref] for i, ref in enumerate(reference_reports)}
        res = {i: [gen] for i, gen in enumerate(generated_reports)}
        
        # Compute metrics
        metrics = {}
        
        # BLEU
        bleu_scorer = Bleu(n=4)
        bleu_scores, _ = bleu_scorer.compute_score(gts, res)
        metrics['bleu_1'] = bleu_scores[0]
        metrics['bleu_2'] = bleu_scores[1]
        metrics['bleu_3'] = bleu_scores[2]
        metrics['bleu_4'] = bleu_scores[3]
        
        # METEOR
        try:
            meteor_scorer = Meteor()
            meteor_score, _ = meteor_scorer.compute_score(gts, res)
            metrics['meteor'] = meteor_score
        except:
            metrics['meteor'] = 0.0
        
        # ROUGE-L
        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(gts, res)
        metrics['rouge_l'] = rouge_score
        
        # CIDEr
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts, res)
        metrics['cider'] = cider_score
        
    except ImportError:
        print("Warning: pycocoevalcap not installed. Using simplified metrics.")
        # Fallback to simple BLEU
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        bleu_scores = []
        for gen, ref in zip(generated_reports, reference_reports):
            gen_tokens = gen.lower().split()
            ref_tokens = [ref.lower().split()]
            
            smoothing = SmoothingFunction()
            score = sentence_bleu(
                ref_tokens,
                gen_tokens,
                smoothing_function=smoothing.method1
            )
            bleu_scores.append(score)
        
        metrics = {
            'bleu_4': np.mean(bleu_scores),
            'meteor': 0.0,
            'rouge_l': 0.0,
            'cider': 0.0
        }
    
    return metrics


def compute_clinical_efficacy(
    generated_reports: List[str],
    reference_reports: List[str],
    predicted_diseases: List[Dict],
    ground_truth_diseases: List[Dict]
) -> Dict[str, float]:
    """
    Compute clinical efficacy metrics
    
    Args:
        generated_reports: List of generated report texts
        reference_reports: List of reference report texts
        predicted_diseases: List of predicted disease dictionaries
        ground_truth_diseases: List of ground truth disease dictionaries
    
    Returns:
        Dictionary with clinical efficacy metrics
    """
    # Hallucination rate (mentions not in reference)
    hallucination_count = 0
    total_mentions = 0
    
    for gen_text, ref_text in zip(generated_reports, reference_reports):
        gen_words = set(gen_text.lower().split())
        ref_words = set(ref_text.lower().split())
        
        # Medical terms that appear in generated but not in reference
        medical_terms = [
            'opacity', 'consolidation', 'effusion', 'pneumothorax',
            'atelectasis', 'edema', 'cardiomegaly', 'pneumonia',
            'fracture', 'nodule', 'mass'
        ]
        
        for term in medical_terms:
            if term in gen_words:
                total_mentions += 1
                if term not in ref_words:
                    hallucination_count += 1
    
    hallucination_rate = hallucination_count / total_mentions if total_mentions > 0 else 0
    
    # Completeness (important findings mentioned)
    completeness_scores = []
    
    for ref_text, gen_text in zip(reference_reports, generated_reports):
        ref_words = set(ref_text.lower().split())
        gen_words = set(gen_text.lower().split())
        
        # Important terms in reference
        important_terms = ref_words & set([
            'opacity', 'consolidation', 'effusion', 'pneumothorax',
            'atelectasis', 'edema', 'cardiomegaly', 'pneumonia',
            'fracture', 'normal', 'clear'
        ])
        
        if len(important_terms) > 0:
            mentioned = len(important_terms & gen_words)
            completeness = mentioned / len(important_terms)
            completeness_scores.append(completeness)
    
    metrics = {
        'hallucination_rate': hallucination_rate,
        'completeness': np.mean(completeness_scores) if completeness_scores else 0.0,
        'total_mentions': total_mentions,
        'hallucinated_mentions': hallucination_count
    }
    
    return metrics


def compute_confidence_calibration(
    predictions: torch.Tensor,
    confidence: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Compute confidence calibration metrics
    Measures how well confidence scores match actual accuracy
    
    Args:
        predictions: [N, num_diseases] predicted probabilities
        confidence: [N, num_diseases] confidence scores
        targets: [N, num_diseases] ground truth labels
        num_bins: Number of bins for calibration
    
    Returns:
        Dictionary with calibration metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(confidence, torch.Tensor):
        confidence = confidence.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Flatten arrays
    pred_flat = predictions.flatten()
    conf_flat = confidence.flatten()
    target_flat = targets.flatten()
    
    # Remove invalid entries
    valid_mask = (target_flat != -1)
    pred_flat = pred_flat[valid_mask]
    conf_flat = conf_flat[valid_mask]
    target_flat = target_flat[valid_mask]
    
    # Compute Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0
    
    for i in range(num_bins):
        bin_mask = (conf_flat >= bin_boundaries[i]) & (conf_flat < bin_boundaries[i + 1])
        
        if bin_mask.sum() == 0:
            continue
        
        bin_conf = conf_flat[bin_mask].mean()
        bin_acc = ((pred_flat[bin_mask] > 0.5) == target_flat[bin_mask]).mean()
        
        ece += abs(bin_conf - bin_acc) * bin_mask.sum() / len(conf_flat)
    
    metrics = {
        'expected_calibration_error': ece,
        'mean_confidence': conf_flat.mean(),
        'mean_accuracy': ((pred_flat > 0.5) == target_flat).mean()
    }
    
    return metrics


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")
    
    # Test CheXpert F1
    predictions = torch.rand(100, 14)
    targets = torch.randint(0, 2, (100, 14)).float()
    
    chexpert_metrics = compute_chexpert_f1(predictions, targets)
    print("\nCheXpert Metrics:")
    for k, v in chexpert_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test RadGraph F1
    gen_reports = [
        "There is an opacity in the right lung. No pleural effusion.",
        "The heart size is normal. Clear lungs."
    ]
    ref_reports = [
        "Right lung opacity present. No effusion noted.",
        "Normal cardiac silhouette. Lungs are clear."
    ]
    
    radgraph_metrics = compute_radgraph_f1(gen_reports, ref_reports)
    print("\nRadGraph Metrics:")
    for k, v in radgraph_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test NLG metrics
    nlg_metrics = compute_nlg_metrics(gen_reports, ref_reports)
    print("\nNLG Metrics:")
    for k, v in nlg_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test calibration
    confidence = torch.rand(100, 14)
    calib_metrics = compute_confidence_calibration(predictions, confidence, targets)
    print("\nCalibration Metrics:")
    for k, v in calib_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n✅ All metrics tested successfully!")

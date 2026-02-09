"""
PyTorch Dataset classes for chest X-ray report generation
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
import torchvision.transforms as transforms


class CXRDataset(Dataset):
    """Chest X-ray report generation dataset"""
    
    def __init__(
        self,
        data_file: str,
        data_root: str,
        tokenizer_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        max_length: int = 512,
        img_size: int = 224,
        split: str = 'train'
    ):
        """
        Args:
            data_file: Path to JSON file with processed data
            data_root: Root directory containing images
            tokenizer_name: Pretrained tokenizer name
            max_length: Maximum sequence length for text
            img_size: Image size for resizing
            split: 'train', 'validate', or 'test'
        """
        self.data_root = Path(data_root)
        self.max_length = max_length
        self.img_size = img_size
        self.split = split
        
        # Load data
        with open(data_file, 'r') as f:
            self.samples = json.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Image transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # CheXpert labels
        self.chexpert_labels = [
            'no_finding', 'enlarged_cardiomediastinum', 'cardiomegaly',
            'lung_opacity', 'lung_lesion', 'edema', 'consolidation',
            'pneumonia', 'atelectasis', 'pneumothorax', 'pleural_effusion',
            'pleural_other', 'fracture', 'support_devices'
        ]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        img_path = self.data_root / sample['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return black image as fallback
            image = torch.zeros(3, self.img_size, self.img_size)
        
        # Get reports
        findings = sample.get('findings', '')
        impression = sample.get('impression', '')
        
        # Combine for full report
        full_report = f"FINDINGS: {findings} IMPRESSION: {impression}"
        
        # Tokenize clinical indication (if available)
        indication = sample.get('indication', 'Chest X-ray')
        indication_tokens = self.tokenizer(
            indication,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize report
        report_tokens = self.tokenizer(
            full_report,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get CheXpert labels
        chexpert = []
        for label in self.chexpert_labels:
            value = sample.get(label, -1)  # -1 for missing
            # Map: 1.0 → 1, 0.0 → 0, -1.0 → 0, NaN → 0
            if value == 1.0:
                chexpert.append(1)
            else:
                chexpert.append(0)
        
        chexpert = torch.tensor(chexpert, dtype=torch.float32)
        
        return {
            'image': image,
            'indication_input_ids': indication_tokens['input_ids'].squeeze(0),
            'indication_attention_mask': indication_tokens['attention_mask'].squeeze(0),
            'report_input_ids': report_tokens['input_ids'].squeeze(0),
            'report_attention_mask': report_tokens['attention_mask'].squeeze(0),
            'chexpert_labels': chexpert,
            'findings': findings,
            'impression': impression,
            'study_id': sample.get('study_id', '')
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched dictionary
    """
    collated = {
        'image': torch.stack([item['image'] for item in batch]),
        'indication_input_ids': torch.stack([item['indication_input_ids'] for item in batch]),
        'indication_attention_mask': torch.stack([item['indication_attention_mask'] for item in batch]),
        'report_input_ids': torch.stack([item['report_input_ids'] for item in batch]),
        'report_attention_mask': torch.stack([item['report_attention_mask'] for item in batch]),
        'chexpert_labels': torch.stack([item['chexpert_labels'] for item in batch]),
    }
    
    # Keep text data as lists (not tensors)
    collated['findings'] = [item['findings'] for item in batch]
    collated['impression'] = [item['impression'] for item in batch]
    collated['study_id'] = [item['study_id'] for item in batch]
    
    return collated


class RadLexOntology:
    """
    Simplified RadLex medical ontology for anatomical concepts
    In production, use full RadLex database
    """
    
    def __init__(self):
        # Anatomical concepts (simplified)
        self.concepts = {
            # Organs
            'lung': ['right lung', 'left lung', 'lungs bilateral'],
            'heart': ['cardiac silhouette', 'heart size', 'cardiomediastinal'],
            'pleura': ['pleural space', 'costophrenic angle'],
            'mediastinum': ['mediastinal contour', 'hilum'],
            'diaphragm': ['hemidiaphragm', 'diaphragmatic'],
            
            # Regions
            'upper_lobe': ['upper lobe', 'apex'],
            'middle_lobe': ['middle lobe', 'lingula'],
            'lower_lobe': ['lower lobe', 'base'],
            
            # Pathologies
            'opacity': ['opacity', 'infiltrate', 'consolidation'],
            'effusion': ['effusion', 'fluid'],
            'pneumothorax': ['pneumothorax', 'air'],
            'cardiomegaly': ['cardiomegaly', 'enlarged heart'],
            'atelectasis': ['atelectasis', 'collapse'],
            'edema': ['edema', 'pulmonary congestion']
        }
        
        # Build reverse mapping
        self.term_to_concept = {}
        for concept, terms in self.concepts.items():
            for term in terms:
                self.term_to_concept[term.lower()] = concept
    
    def get_concept(self, term: str) -> Optional[str]:
        """Get concept for a given term"""
        return self.term_to_concept.get(term.lower())
    
    def get_embedding_dim(self) -> int:
        """Get number of concepts"""
        return len(self.concepts)
    
    def get_concept_names(self) -> List[str]:
        """Get list of all concept names"""
        return list(self.concepts.keys())


if __name__ == "__main__":
    # Test dataset
    print("Testing CXR Dataset...")
    
    # This is a test - adjust paths as needed
    test_file = "./data/preprocessed/train.json"
    test_root = "./data/mimic-cxr"
    
    if os.path.exists(test_file):
        dataset = CXRDataset(test_file, test_root, split='train')
        print(f"Dataset size: {len(dataset)}")
        
        # Load first sample
        sample = dataset[0]
        print("\nSample keys:", sample.keys())
        print("Image shape:", sample['image'].shape)
        print("CheXpert labels:", sample['chexpert_labels'])
        print("Findings:", sample['findings'][:100], "...")
    else:
        print(f"Test file not found: {test_file}")
        print("Run preprocessing first: python data/preprocess.py")

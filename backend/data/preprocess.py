"""
Data Preprocessing Pipeline for MIMIC-CXR and IU-Xray
Handles image preprocessing, report parsing, and dataset splitting
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import pickle


class DataPreprocessor:
    """Preprocesses chest X-ray images and reports"""
    
    def __init__(self, dataset_name: str, data_dir: str, output_dir: str):
        """
        Args:
            dataset_name: 'mimic-cxr' or 'iu-xray'
            data_dir: Directory containing raw dataset
            output_dir: Directory to save preprocessed data
        """
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard image size
        self.img_size = (224, 224)
        
        # CheXpert disease labels (14 observations)
        self.chexpert_labels = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess a single chest X-ray image
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Preprocessed image array (224, 224, 3)
        """
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Resize
            img = img.resize(self.img_size, Image.LANCZOS)
            
            # Convert to array
            img_array = np.array(img, dtype=np.float32)
            
            # Normalize to [0, 1]
            img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def parse_mimic_report(self, report_text: str) -> Dict[str, str]:
        """
        Parse MIMIC-CXR report into sections
        
        Args:
            report_text: Raw report text
        
        Returns:
            Dictionary with 'findings' and 'impression'
        """
        sections = {
            'findings': '',
            'impression': ''
        }
        
        # Clean text
        text = report_text.strip()
        
        # Extract FINDINGS section
        findings_match = re.search(
            r'FINDINGS?:(.*?)(?:IMPRESSION|$)',
            text,
            re.DOTALL | re.IGNORECASE
        )
        if findings_match:
            sections['findings'] = findings_match.group(1).strip()
        
        # Extract IMPRESSION section
        impression_match = re.search(
            r'IMPRESSION:(.*?)$',
            text,
            re.DOTALL | re.IGNORECASE
        )
        if impression_match:
            sections['impression'] = impression_match.group(1).strip()
        
        # Clean up sections
        for key in sections:
            sections[key] = self._clean_text(sections[key])
        
        return sections
    
    def parse_iuxray_report(self, report_text: str) -> Dict[str, str]:
        """
        Parse IU-Xray report into sections
        
        Args:
            report_text: Raw report text
        
        Returns:
            Dictionary with 'findings' and 'impression'
        """
        sections = {
            'findings': '',
            'impression': ''
        }
        
        # IU-Xray reports have different section headers
        text = report_text.strip()
        
        # Extract FINDINGS
        findings_match = re.search(
            r'(?:FINDINGS?|COMPARISON):(.*?)(?:IMPRESSION|CONCLUSION|$)',
            text,
            re.DOTALL | re.IGNORECASE
        )
        if findings_match:
            sections['findings'] = findings_match.group(1).strip()
        
        # Extract IMPRESSION
        impression_match = re.search(
            r'(?:IMPRESSION|CONCLUSION):(.*?)$',
            text,
            re.DOTALL | re.IGNORECASE
        )
        if impression_match:
            sections['impression'] = impression_match.group(1).strip()
        
        # Clean up sections
        for key in sections:
            sections[key] = self._clean_text(sections[key])
        
        return sections
    
    def _clean_text(self, text: str) -> str:
        """Clean report text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\;\:\-\(\)]', '', text)
        
        # Remove numbered lists
        text = re.sub(r'\d+\.', '', text)
        
        return text.strip()
    
    def process_mimic_cxr(self):
        """Process MIMIC-CXR dataset"""
        print("Processing MIMIC-CXR dataset...")
        
        # Load metadata
        metadata_file = self.data_dir / "mimic-cxr-2.0.0-metadata.csv.gz"
        split_file = self.data_dir / "mimic-cxr-2.0.0-split.csv.gz"
        chexpert_file = self.data_dir / "mimic-cxr-2.0.0-chexpert.csv.gz"
        
        if not metadata_file.exists():
            print(f"❌ Metadata file not found: {metadata_file}")
            print("Please run download_mimic.py first")
            return
        
        # Load data
        print("Loading metadata...")
        metadata = pd.read_csv(metadata_file)
        splits = pd.read_csv(split_file)
        chexpert = pd.read_csv(chexpert_file)
        
        # Merge dataframes
        data = metadata.merge(splits, on=['subject_id', 'study_id'])
        data = data.merge(chexpert, on=['subject_id', 'study_id'])
        
        print(f"Total samples: {len(data)}")
        print(f"Train: {len(data[data['split'] == 'train'])}")
        print(f"Validate: {len(data[data['split'] == 'validate'])}")
        print(f"Test: {len(data[data['split'] == 'test'])}")
        
        # Process each split
        for split in ['train', 'validate', 'test']:
            split_data = data[data['split'] == split]
            self._process_split(split_data, split, 'mimic-cxr')
        
        print("✅ MIMIC-CXR preprocessing complete!")
    
    def process_iuxray(self):
        """Process IU-Xray dataset"""
        print("Processing IU-Xray dataset...")
        
        images_dir = self.data_dir / "images"
        reports_dir = self.data_dir / "reports"
        
        if not images_dir.exists():
            print(f"❌ Images directory not found: {images_dir}")
            print("Please download IU-Xray dataset first")
            return
        
        # Scan for images and reports
        print("Scanning dataset...")
        samples = []
        
        for img_file in tqdm(list(images_dir.glob("*.png")), desc="Processing"):
            # Format: CXR1_1_IM-0001-3001.png
            img_id = img_file.stem
            
            # Find corresponding report (this is dataset-specific)
            # You may need to adjust based on actual report file structure
            report_file = reports_dir / "ecgen-radiology" / f"{img_id}.xml"
            
            if report_file.exists():
                samples.append({
                    'image_path': str(img_file),
                    'report_path': str(report_file),
                    'image_id': img_id
                })
        
        print(f"Found {len(samples)} image-report pairs")
        
        # Create train/val/test split (80/10/10)
        np.random.seed(42)
        indices = np.random.permutation(len(samples))
        
        train_idx = indices[:int(0.8 * len(samples))]
        val_idx = indices[int(0.8 * len(samples)):int(0.9 * len(samples))]
        test_idx = indices[int(0.9 * len(samples)):]
        
        splits = {
            'train': [samples[i] for i in train_idx],
            'validate': [samples[i] for i in val_idx],
            'test': [samples[i] for i in test_idx]
        }
        
        for split_name, split_samples in splits.items():
            print(f"\nProcessing {split_name} split ({len(split_samples)} samples)...")
            # Save split info
            split_file = self.output_dir / f"{split_name}.json"
            with open(split_file, 'w') as f:
                json.dump(split_samples, f, indent=2)
        
        print("✅ IU-Xray preprocessing complete!")
    
    def _process_split(self, data: pd.DataFrame, split_name: str, dataset: str):
        """Process a single data split"""
        print(f"\nProcessing {split_name} split...")
        
        processed = []
        
        for idx, row in tqdm(data.iterrows(), total=len(data), desc=split_name):
            sample = {
                'image_path': row.get('path', ''),
                'subject_id': row.get('subject_id', ''),
                'study_id': row.get('study_id', ''),
                'findings': row.get('findings', ''),
                'impression': row.get('impression', ''),
            }
            
            # Add CheXpert labels
            for label in self.chexpert_labels:
                col_name = label.replace(' ', '_').lower()
                sample[col_name] = row.get(label, 0)
            
            processed.append(sample)
        
        # Save to JSON
        output_file = self.output_dir / f"{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(processed, f, indent=2)
        
        print(f"Saved {len(processed)} samples to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess CXR datasets")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=['mimic-cxr', 'iu-xray'],
                       help="Dataset to preprocess")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing raw dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save preprocessed data")
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(args.dataset, args.data_dir, args.output_dir)
    
    if args.dataset == 'mimic-cxr':
        preprocessor.process_mimic_cxr()
    else:
        preprocessor.process_iuxray()


if __name__ == "__main__":
    main()

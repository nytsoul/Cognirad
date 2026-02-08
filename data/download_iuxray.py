"""
IU-Xray Dataset Downloader
Open access dataset from Indiana University
"""

import os
import zipfile
from pathlib import Path
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads"""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_iuxray(output_dir: str):
    """
    Download IU-Xray dataset
    
    Args:
        output_dir: Directory to save the dataset
    
    Note:
        Dataset is publicly available but may require registration
        at https://openi.nlm.nih.gov/
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("IU-Xray Dataset Download")
    print("=" * 70)
    
    # IU-Xray URLs (these are example URLs - actual links may vary)
    urls = {
        "images": "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz",
        "reports": "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
    }
    
    print("\n📥 Downloading IU-Xray dataset...")
    print("Note: This dataset is from Indiana University chest X-ray collection")
    print("      Contains ~7,000+ chest X-rays with associated reports\n")
    
    # Alternative: Provide Kaggle instructions
    print("⚠️  Recommended: Download from Kaggle (easier access)")
    print("   https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university")
    print("\nSteps:")
    print("1. Install Kaggle CLI: pip install kaggle")
    print("2. Setup credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
    print("3. Download: kaggle datasets download -d raddar/chest-xrays-indiana-university")
    print(f"4. Extract to: {output_path}")
    
    print("\n" + "=" * 70)
    print("Manual Download Instructions:")
    print("=" * 70)
    print("1. Visit: https://openi.nlm.nih.gov/faq#collection")
    print("2. Download image collection (PNG format)")
    print("3. Download reports XML file")
    print(f"4. Extract all files to: {output_path}")
    print("   Expected structure:")
    print("   " + str(output_path) + "/")
    print("   ├── images/")
    print("   │   ├── CXR1_1_IM-0001-3001.png")
    print("   │   └── ...")
    print("   └── reports/")
    print("       ├── ecgen-radiology/")
    print("       └── ...")
    
    # Create directory structure
    (output_path / "images").mkdir(exist_ok=True)
    (output_path / "reports").mkdir(exist_ok=True)
    
    print(f"\n✅ Directory structure created at {output_path}")
    print("\nAfter downloading, run preprocessing:")
    print("   python data/preprocess.py --dataset iuxray")


def download_with_kaggle(output_dir: str):
    """
    Download IU-Xray using Kaggle API
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        api = KaggleApi()
        api.authenticate()
        
        print("Downloading IU-Xray from Kaggle...")
        api.dataset_download_files(
            'raddar/chest-xrays-indiana-university',
            path=str(output_path),
            unzip=True
        )
        
        print(f"✅ Dataset downloaded to {output_path}")
        
    except ImportError:
        print("❌ Kaggle API not installed. Install with: pip install kaggle")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Kaggle API credentials are configured")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download IU-Xray dataset")
    parser.add_argument("--output_dir", type=str, default="./data/iu-xray",
                       help="Output directory for dataset")
    parser.add_argument("--use_kaggle", action="store_true",
                       help="Use Kaggle API for download")
    
    args = parser.parse_args()
    
    if args.use_kaggle:
        download_with_kaggle(args.output_dir)
    else:
        download_iuxray(args.output_dir)

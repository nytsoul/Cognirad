"""
MIMIC-CXR Dataset Downloader
Requires PhysioNet credentials
"""

import os
import sys
from pathlib import Path
import pandas as pd
import urllib.request
from tqdm import tqdm


def download_mimic_cxr(output_dir: str, username: str = None, password: str = None):
    """
    Download MIMIC-CXR dataset from PhysioNet
    
    Args:
        output_dir: Directory to save the dataset
        username: PhysioNet username (required for download)
        password: PhysioNet password (required for download)
    
    Note:
        You must have completed CITI training and signed the DUA at:
        https://physionet.org/content/mimic-cxr/2.0.0/
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MIMIC-CXR Dataset Download")
    print("=" * 70)
    
    if username is None or password is None:
        print("\n⚠️  IMPORTANT: MIMIC-CXR requires PhysioNet credentials")
        print("\nSteps to download:")
        print("1. Create account at https://physionet.org/")
        print("2. Complete CITI training")
        print("3. Sign data use agreement")
        print("4. Run this script with your credentials:")
        print(f"   python -c \"from data import download_mimic_cxr; download_mimic_cxr('{output_dir}', 'username', 'password')\"")
        print("\nAlternatively, manually download from:")
        print("https://physionet.org/content/mimic-cxr/2.0.0/")
        return
    
    base_url = "https://physionet.org/files/mimic-cxr/2.0.0/"
    
    # Files to download
    files = [
        "mimic-cxr-2.0.0-metadata.csv.gz",
        "mimic-cxr-2.0.0-split.csv.gz",
        "mimic-cxr-2.0.0-chexpert.csv.gz",
        "mimic-cxr-2.0.0-negbio.csv.gz"
    ]
    
    # Setup authentication
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, base_url, username, password)
    handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)
    
    print("\nDownloading metadata files...")
    for file in tqdm(files, desc="Files"):
        url = base_url + file
        output_file = output_path / file
        
        try:
            urllib.request.urlretrieve(url, output_file)
        except Exception as e:
            print(f"\n❌ Error downloading {file}: {e}")
            print("Please verify your credentials and access permissions.")
            return
    
    print("\n✅ Metadata downloaded successfully!")
    print(f"📁 Location: {output_path}")
    print("\n⚠️  Note: Image files (JPG) must be downloaded separately")
    print("   They are organized in p10/, p11/, ..., p19/ directories")
    print("   Total size: ~470 GB")
    print("\nNext steps:")
    print("1. Download image files from PhysioNet")
    print("2. Run preprocessing: python data/preprocess.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download MIMIC-CXR dataset")
    parser.add_argument("--output_dir", type=str, default="./data/mimic-cxr",
                       help="Output directory for dataset")
    parser.add_argument("--username", type=str, help="PhysioNet username")
    parser.add_argument("--password", type=str, help="PhysioNet password")
    
    args = parser.parse_args()
    download_mimic_cxr(args.output_dir, args.username, args.password)

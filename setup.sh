#!/bin/bash

# CogniRad++ Quick Start Script
# This script helps you get started quickly

echo "================================================"
echo "  CogniRad++ Quick Start"
echo "================================================"
echo ""

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA available"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo "⚠ CUDA not found - will use CPU"
fi

echo ""
echo "================================================"
echo "  Step 1: Install Dependencies"
echo "================================================"
read -p "Install Python packages? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -r requirements.txt
    python -m spacy download en_core_sci_md
    echo "✓ Dependencies installed"
fi

echo ""
echo "================================================"
echo "  Step 2: Download Datasets"
echo "================================================"
echo "Note: You need PhysioNet credentials for MIMIC-CXR"
echo ""
read -p "Download datasets now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "1. MIMIC-CXR (requires credentials)"
    echo "2. IU-Xray (via Kaggle)"
    read -p "Choose dataset (1/2): " dataset_choice
    
    if [ "$dataset_choice" = "1" ]; then
        read -p "PhysioNet username: " username
        read -sp "PhysioNet password: " password
        echo
        python data/download_mimic.py \
            --output_dir ./data/mimic-cxr \
            --username "$username" \
            --password "$password"
    elif [ "$dataset_choice" = "2" ]; then
        python data/download_iuxray.py \
            --output_dir ./data/iu-xray \
            --use_kaggle
    fi
fi

echo ""
echo "================================================"
echo "  Step 3: Preprocess Data"
echo "================================================"
read -p "Preprocess data now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Preprocessing... (this may take a while)"
    python data/preprocess.py \
        --dataset mimic-cxr \
        --data_dir ./data/mimic-cxr \
        --output_dir ./data/preprocessed
    echo "✓ Data preprocessed"
fi

echo ""
echo "================================================"
echo "  Step 4: Choose Action"
echo "================================================"
echo "1. Start training"
echo "2. Run demo notebook"
echo "3. Evaluate model"
echo "4. Exit"
echo ""
read -p "Choose action (1-4): " action

case $action in
    1)
        echo "Starting training..."
        python training/trainer.py
        ;;
    2)
        echo "Starting Jupyter notebook..."
        jupyter notebook notebooks/demo_inference.ipynb
        ;;
    3)
        echo "Running evaluation..."
        python evaluation/evaluator.py \
            --checkpoint ./checkpoints/best_model.pt \
            --test_data ./data/preprocessed/test.json \
            --data_root ./data/mimic-cxr
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        ;;
esac

echo ""
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  - Edit training/config.py to customize training"
echo "  - Check notebooks/demo_inference.ipynb for examples"
echo "  - Read README.md for detailed documentation"
echo ""
echo "For help: https://github.com/yourusername/cognirad-plusplus"
echo "================================================"

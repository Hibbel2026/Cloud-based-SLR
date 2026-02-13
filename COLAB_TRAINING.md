"""
Colab Training Notebook Script
Kopiera detta till en Colab cell för att träna 3D CNN med GPU
"""

# ============================================================================
# COLAB SETUP - Run this first
# ============================================================================

# 1. Clone repository
# !git clone https://github.com/Hibbel2026/Cloud-based-SLR.git
# %cd Cloud-based-SLR

# 2. Install dependencies
# !pip install opencv-python torch torchvision tqdm

# 3. Download dataset (if needed)
# !gdown DATASET_ID -O data/

# ============================================================================
# TRAINING SCRIPT
# ============================================================================

import torch
import os

# Check GPU
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Train 3D CNN
# !python3 src/train_3dcnn.py --model 3dcnn --batch_size 8 --num_epochs 50 --device cuda

# ============================================================================
# AFTER TRAINING - Push results back to GitHub
# ============================================================================

# !git config --global user.email "your-email@example.com"
# !git config --global user.name "Your Name"
# !git add checkpoints/best_temporal3dcnn_model.pth
# !git commit -m "Add 3D CNN trained model results"
# !git push origin Baseline-ML

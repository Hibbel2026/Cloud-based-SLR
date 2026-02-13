# src/config.py
import torch
import os

# Paths
DATA_DIR = "data/processed"
MODEL_SAVE_DIR = "checkpoints"
RESULTS_DIR = "results"

# Model parameters
INPUT_FEATURES = 225  # Pose (99) + Hands (126)
NUM_CLASSES = 100     # WLASL-100
DROPOUT = 0.5
MAX_FRAMES = 48

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
PATIENCE = 15  # Early stopping

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create directories
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
#!/usr/bin/env python3
"""
Quick skeleton CNN training on a subset of data for fast testing.
Use this to verify the model works before running full training on Colab/GPU.

This trains on:
- Small batch size (4)
- Smaller frames (112x112 instead of 224x224)
- Fewer epochs (5-10)
- Should complete in 5-10 minutes on CPU

Run: python train_skeleton_quick.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import time
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset_skeleton import get_skeleton_dataloaders


class QuickSkeletonCNN(nn.Module):
    """Lightweight 2D CNN for quick testing."""
    
    def __init__(self, num_classes: int = 100, frame_size: int = 112):
        super().__init__()
        
        # Single frame CNN (lightweight version)
        self.frame_cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_frames, 3, height, width)
        """
        batch_size, num_frames, c, h, w = x.shape
        
        # Process each frame
        x = x.view(batch_size * num_frames, c, h, w)
        x = self.frame_cnn(x)
        x = x.view(batch_size * num_frames, -1)
        
        # Average across time
        x = x.view(batch_size, num_frames, -1).mean(dim=1)
        
        # Classify
        x = self.classifier(x)
        return x


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", unit="batch")
    
    for frames, labels in pbar:
        frames = frames.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{total_loss / (pbar.n + 1):.4f}',
            'acc': f'{100.0 * correct / total:.1f}%'
        })
    
    return total_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, criterion, device, split="VAL"):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=split, unit="batch")
        
        for frames, labels in pbar:
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{total_loss / (pbar.n + 1):.4f}',
                'acc': f'{100.0 * correct / total:.1f}%'
            })
    
    return total_loss / len(val_loader), 100.0 * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_size = 112  # Smaller for speed
    
    print(f"\n{'='*70}")
    print(f"Skeleton CNN - Quick Training (for testing)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Frame size: {frame_size}x{frame_size}")
    print(f"Batch size: 4 (small for speed)")
    print(f"Epochs: 10 (short run)")
    print(f"{'='*70}\n")
    
    # Load data
    print("📁 Loading datasets...")
    train_loader, val_loader, test_loader = get_skeleton_dataloaders(
        data_dir="data",
        batch_size=4,
        num_frames=16,
        frame_size=(frame_size, frame_size),
        num_workers=0,
        cache_dir=None
    )
    
    # Create model
    print("\n🏗️  Creating model...")
    model = QuickSkeletonCNN(num_classes=100, frame_size=frame_size).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: {num_params:,} parameters")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training loop
    print(f"\n🚀 Starting quick training (est. 5-10 min)...\n")
    best_val_acc = 0.0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    
    for epoch in range(1, 11):  # 10 epochs
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device, "VAL")
        test_loss, test_acc = validate(model, test_loader, criterion, device, "TEST")
        
        scheduler.step(val_acc)
        
        print(f"\n{'─'*70}")
        print(f"Epoch {epoch}/10")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc,
            }, checkpoint_dir / "best_skeleton_quick_model.pth")
            print(f"  ✓ New best model saved!")
        
        print(f"{'─'*70}")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✓ Training finished!")
    print(f"  Time: {elapsed / 60:.1f} minutes")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Final Test Acc: {test_acc:.2f}%")
    print(f"{'='*70}")
    print(f"\nNext: If results are good, run full training on Colab with GPU!")
    print(f"Run: python src/train_skeleton.py --batch_size 16 --num_epochs 50\n")


if __name__ == "__main__":
    main()

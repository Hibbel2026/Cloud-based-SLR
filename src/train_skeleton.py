"""
=============================================================================
Sign Language Recognition - Skeleton CNN Training
=============================================================================
File: src/train_skeleton.py
Description: Train a 2D CNN on skeleton-drawn frames.
             This hybrid approach combines visual information with skeleton clarity.

Usage:
    python src/train_skeleton.py --model cnn2d --batch_size 16 --num_epochs 50

=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Tuple
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset_skeleton import get_skeleton_dataloaders


class SkeletonCNN2D(nn.Module):
    """
    2D CNN optimized for skeleton frames.
    Input: (batch, num_frames, 3, 224, 224)
    Output: (batch, num_classes)
    
    Architecture: Multiple frames processed independently, then averaged
    """
    
    def __init__(self, num_classes: int = 100):
        super().__init__()
        
        # Single frame CNN
        self.frame_cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Temporal aggregation + Classification
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_frames, 3, height, width)
        
        Returns:
            logits: (batch, num_classes)
        """
        batch_size, num_frames, c, h, w = x.shape
        
        # Process each frame through CNN
        x = x.view(batch_size * num_frames, c, h, w)  # (batch*frames, 3, h, w)
        x = self.frame_cnn(x)  # (batch*frames, 256, 1, 1)
        x = x.view(batch_size * num_frames, -1)  # (batch*frames, 256)
        
        # Average features across temporal dimension
        x = x.view(batch_size, num_frames, -1).mean(dim=1)  # (batch, 256)
        
        # Classification
        x = self.classifier(x)  # (batch, num_classes)
        
        return x


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [TRAIN]")
    
    for frames, labels in pbar:
        frames = frames.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{total_loss / (pbar.n + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100.0 * correct / total


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split: str = "VAL"
) -> Tuple[float, float]:
    """Validate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"[{split}]")
        
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
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    return total_loss / len(val_loader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Train skeleton CNN for sign language recognition")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="skeleton_cnn2d", 
                       help="Model type: skeleton_cnn2d")
    parser.add_argument("--num_classes", type=int, default=100,
                       help="Number of output classes")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--num_frames", type=int, default=16,
                       help="Number of frames per video")
    parser.add_argument("--frame_size", type=int, default=224,
                       help="Frame size (224x224)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: 'auto', 'cuda', or 'cpu'")
    
    # Caching arguments
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Directory to cache skeleton frames")
    parser.add_argument("--use_cache", action="store_true",
                       help="Use cached skeleton frames if available")
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*70}")
    print(f"Sign Language Recognition - Skeleton CNN Training")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Num epochs: {args.num_epochs}")
    print(f"{'='*70}\n")
    
    # Create dataloaders
    print("📁 Loading datasets...")
    train_loader, val_loader, test_loader = get_skeleton_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        frame_size=(args.frame_size, args.frame_size),
        num_workers=args.num_workers,
        cache_dir=args.cache_dir if args.use_cache else None
    )
    print(f"✓ Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = SkeletonCNN2D(num_classes=args.num_classes).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {num_params:,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    print(f"\n🚀 Starting training...\n")
    best_val_acc = 0.0
    patience_counter = 0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    
    start_time = time.time()
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.num_epochs
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, "VAL")
        
        # Test
        test_loss, test_acc = validate(model, test_loader, criterion, device, "TEST")
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Scheduler step
        scheduler.step(val_acc)
        
        # Print epoch summary
        print(f"\n{'─'*70}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}  Test Acc:  {test_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"best_skeleton_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc,
                'history': history,
            }, checkpoint_path)
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹️  Early stopping triggered after {epoch} epochs")
                break
        
        print(f"{'─'*70}")
    
    # Training finished
    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✓ Training finished!")
    print(f"  Time elapsed: {elapsed_time / 60:.1f} minutes")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Final Test Acc: {test_acc:.2f}%")
    print(f"{'='*70}\n")
    
    # Save training history
    history_path = checkpoint_dir / "skeleton_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✓ Training history saved to {history_path}")


if __name__ == "__main__":
    main()

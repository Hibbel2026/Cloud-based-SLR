"""
3D CNN Training Script
Train on raw video frames using GPU in Colab or local machine
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os
from pathlib import Path

from models.cnn3d_model import Temporal3DCNN, Temporal3DCNNLite
from dataset_3dcnn import get_raw_video_dataloaders
import config


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=True)
    for frames, labels in pbar:
        frames, labels = frames.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix(loss=f'{loss.item():.4f}', refresh=True)
    
    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="Validation", leave=True):
            frames, labels = frames.to(device), labels.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total


def main(args):
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    
    # Create directories
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Load data
    print(f"\nLoading raw video data from {args.data_dir}...")
    train_loader, val_loader, test_loader = get_raw_video_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        frame_size=(args.frame_size, args.frame_size),
        num_workers=args.num_workers
    )
    
    # Create model
    if args.model == '3dcnn':
        model = Temporal3DCNN(
            num_classes=100,
            num_frames=args.num_frames,
            input_height=args.frame_size,
            input_width=args.frame_size
        ).to(device)
        model_name = 'temporal3dcnn'
    elif args.model == '3dcnn_lite':
        model = Temporal3DCNNLite(
            num_classes=100,
            num_frames=args.num_frames,
            input_height=args.frame_size,
            input_width=args.frame_size
        ).to(device)
        model_name = 'temporal3dcnn_lite'
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model}")
    print(f"Parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    print(f"\nStarting training ({args.num_epochs} epochs)...")
    print("="*60)
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = f"{config.MODEL_SAVE_DIR}/best_{model_name}_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"✓ Saved new best model (Val Acc: {val_acc:.2f}%) → {model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Test evaluation
    print("\n" + "="*60)
    print("Evaluating on test set...")
    model_path = f"{config.MODEL_SAVE_DIR}/best_{model_name}_model.pth"
    model.load_state_dict(torch.load(model_path))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D CNN on raw video frames")
    parser.add_argument('--model', default='3dcnn', choices=['3dcnn', '3dcnn_lite'],
                       help='Model architecture')
    parser.add_argument('--data_dir', default='data/WLASL_100',
                       help='Path to WLASL_100 dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (lower for GPU memory)')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames to sample from each video')
    parser.add_argument('--frame_size', type=int, default=224,
                       help='Target frame size (224x224 or 112x112 for lite)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    main(args)

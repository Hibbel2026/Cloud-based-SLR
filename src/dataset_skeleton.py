"""
=============================================================================
Sign Language Recognition - Skeleton Dataset Module
=============================================================================
File: src/dataset_skeleton.py
Description: PyTorch Dataset that loads skeleton-drawn frames from videos.
             Uses MediaPipe to generate skeleton frames on-the-fly or cached.

Features:
- Loads videos from WLASL-100 directory structure
- Extracts skeleton frames using SkeletonDrawer
- Data augmentation (brightness, flip, temporal jitter)
- Caching option to save skeleton frames for faster training

=============================================================================
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm
import os
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from skeleton_drawer import SkeletonDrawer, extract_skeleton_frames_from_video


class SkeletonDataset(Dataset):
    """
    Dataset that loads skeleton-drawn frames from videos.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize skeleton dataset.
        
        Args:
            data_dir: Path to WLASL_100 directory
            split: "train", "val", or "test"
            num_frames: Number of frames to extract per video
            frame_size: Size of output frames (height, width)
            augment: Whether to apply augmentation
            cache_dir: Optional directory to cache skeleton frames
        """
        self.data_dir = Path(data_dir) / "WLASL_100" / split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Scan directory structure
        self.classes = sorted([d.name for d in self.data_dir.iterdir() 
                               if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build list of (video_path, class_idx)
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.data_dir / class_name
            for video_file in class_dir.glob("*.mp4"):
                self.samples.append((str(video_file), class_idx))
        
        print(f"✓ Found {len(self.samples)} videos in {len(self.classes)} classes")
        print(f"  Split: {split}, Classes: {len(self.classes)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get skeleton frames and label for a video.
        
        Returns:
            frames: Tensor of shape (num_frames, 3, height, width), values in [0, 1]
            label: Class index
        """
        video_path, label = self.samples[idx]
        
        # Try to load from cache
        if self.cache_dir:
            cache_path = self._get_cache_path(video_path)
            if cache_path.exists():
                cached_data = np.load(cache_path)
                frames = cached_data['frames']
            else:
                frames, _ = extract_skeleton_frames_from_video(
                    video_path, self.num_frames, self.frame_size
                )
                np.savez(cache_path, frames=frames)
        else:
            frames, _ = extract_skeleton_frames_from_video(
                video_path, self.num_frames, self.frame_size
            )
        
        # Handle case where fewer frames were extracted than expected
        if len(frames) < self.num_frames:
            # Pad with white frames
            padding = self.num_frames - len(frames)
            white_frame = np.ones(
                (self.frame_size[0], self.frame_size[1], 3), 
                dtype=np.uint8
            ) * 255
            frames = np.vstack([frames, np.repeat([white_frame], padding, axis=0)])
        
        # Apply augmentation
        if self.augment:
            frames = self._augment_frames(frames)
        
        # Convert to tensor: (num_frames, height, width, 3) -> (num_frames, 3, height, width)
        frames = torch.from_numpy(frames).float() / 255.0  # Normalize to [0, 1]
        frames = frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        
        return frames, label
    
    def _augment_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to frames.
        
        Args:
            frames: Array of shape (num_frames, height, width, 3)
        
        Returns:
            Augmented frames
        """
        frames = frames.copy()
        
        # Random brightness adjustment
        if np.random.rand() < 0.3:
            brightness_factor = np.random.uniform(0.8, 1.2)
            frames = np.clip(frames * brightness_factor, 0, 255).astype(np.uint8)
        
        # Random horizontal flip
        if np.random.rand() < 0.5:
            frames = np.flip(frames, axis=2).copy()  # Flip width axis and copy to remove negative strides
        
        # Random temporal jitter (shuffle frames slightly)
        if np.random.rand() < 0.2:
            # Random shuffle a few frames
            indices = np.arange(len(frames))
            np.random.shuffle(indices[:5])  # Shuffle first 5 frames slightly
            frames = frames[indices].copy()
        
        return frames
    
    def _get_cache_path(self, video_path: str) -> Path:
        """Get cache file path for a video."""
        # Create unique filename from video path
        filename = Path(video_path).stem + ".npz"
        return self.cache_dir / filename
    
    def get_class_name(self, idx: int) -> str:
        """Get class name from index."""
        return self.classes[idx]


def get_skeleton_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for train, validation, and test sets.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        num_frames: Number of frames per video
        frame_size: Size of frames
        num_workers: Number of worker processes
        cache_dir: Optional cache directory for skeleton frames
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Training set (with augmentation)
    train_dataset = SkeletonDataset(
        data_dir=data_dir,
        split="train",
        num_frames=num_frames,
        frame_size=frame_size,
        augment=True,
        cache_dir=cache_dir
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation set (no augmentation)
    val_dataset = SkeletonDataset(
        data_dir=data_dir,
        split="val",
        num_frames=num_frames,
        frame_size=frame_size,
        augment=False,
        cache_dir=cache_dir
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Test set (no augmentation)
    test_dataset = SkeletonDataset(
        data_dir=data_dir,
        split="test",
        num_frames=num_frames,
        frame_size=frame_size,
        augment=False,
        cache_dir=cache_dir
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_skeleton_cache(
    data_dir: str,
    cache_dir: str,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (224, 224),
    split: str = "train",
):
    """
    Pre-compute and cache all skeleton frames for faster training.
    
    Args:
        data_dir: Path to data directory
        cache_dir: Path to cache directory
        num_frames: Number of frames per video
        frame_size: Size of frames
        split: Dataset split to cache
    """
    dataset = SkeletonDataset(
        data_dir=data_dir,
        split=split,
        num_frames=num_frames,
        frame_size=frame_size,
        augment=False,
        cache_dir=cache_dir
    )
    
    print(f"\n🔄 Creating skeleton cache for {split} set...")
    print(f"   Output: {cache_dir}/{split}/")
    
    for idx in tqdm(range(len(dataset)), desc=f"Caching {split} frames"):
        _ = dataset[idx]  # This triggers caching
    
    print(f"✓ Cache created successfully!")

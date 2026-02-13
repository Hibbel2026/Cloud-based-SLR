"""
3D CNN Dataset Module
Loads raw video frames directly from disk for 3D CNN training
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List
import random


class RawVideoDataset(Dataset):
    """
    Load raw video frames directly for 3D CNN
    
    Structure:
    data/WLASL_100/{split}/{class_name}/{video_files}
    """
    
    def __init__(self, root_dir: str, split: str = 'train', 
                 num_frames: int = 16, frame_size: Tuple[int, int] = (224, 224),
                 augment: bool = False):
        """
        Args:
            root_dir: Path to WLASL_100 directory
            split: 'train', 'val', or 'test'
            num_frames: Number of frames to extract per video
            frame_size: Target frame size (height, width)
            augment: Apply data augmentation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.augment = augment
        
        self.split_dir = self.root_dir / split
        
        # Get class names from directory structure
        self.class_names = sorted([d.name for d in self.split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Build file list
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.split_dir / class_name
            video_files = list(class_dir.glob('*.mp4')) + list(class_dir.glob('*.avi')) + \
                         list(class_dir.glob('*.mov')) + list(class_dir.glob('*.MOV'))
            
            for video_file in video_files:
                self.samples.append((str(video_file), self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} videos from {split} split")
        print(f"Classes: {len(self.class_names)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load video and return frames + label
        
        Returns:
            frames: [3, num_frames, height, width] - RGB video tensor
            label: class index
        """
        video_path, label = self.samples[idx]
        
        # Load video
        frames = self._load_video(video_path)
        
        # Apply augmentation if training
        if self.augment and frames is not None:
            frames = self._augment_frames(frames)
        
        # Convert to tensor [C, T, H, W]
        if frames is None:
            # Return black frames if loading failed
            frames = np.zeros((self.num_frames, *self.frame_size, 3), dtype=np.float32)
        
        # Normalize to [-1, 1]
        frames = frames.astype(np.float32) / 127.5 - 1.0
        
        # Reshape to [T, H, W, C] -> [C, T, H, W]
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2)
        
        return frames, label
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """
        Load video and sample frames uniformly
        
        Returns:
            frames: [num_frames, height, width, 3] (BGR - OpenCV format)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                return None
            
            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames < self.num_frames:
                # If video has fewer frames than needed, repeat frames
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            else:
                # Sample uniformly
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            frames = []
            for frame_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Resize to target size
                    frame = cv2.resize(frame, self.frame_size)
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    # If frame reading fails, repeat last frame
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((*self.frame_size, 3), dtype=np.uint8))
            
            cap.release()
            
            return np.array(frames, dtype=np.uint8)
        
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None
    
    def _augment_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to frames
        """
        # Random brightness
        if random.random() < 0.3:
            brightness = random.uniform(0.8, 1.2)
            frames = np.clip(frames.astype(float) * brightness, 0, 255).astype(np.uint8)
        
        # Random horizontal flip
        if random.random() < 0.3:
            frames = np.fliplr(frames)
        
        # Random temporal shuffling (drop some frames)
        if random.random() < 0.2:
            drop_idx = random.randint(0, self.num_frames - 1)
            if drop_idx > 0:
                frames[drop_idx] = frames[drop_idx - 1]
        
        return frames


def get_raw_video_dataloaders(root_dir: str, batch_size: int = 32, 
                             num_frames: int = 16, frame_size: Tuple[int, int] = (224, 224),
                             num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, val, test splits
    
    Args:
        root_dir: Path to WLASL_100 root directory
        batch_size: Batch size
        num_frames: Number of frames per video
        frame_size: Target frame dimensions
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = RawVideoDataset(root_dir, split='train', num_frames=num_frames,
                                   frame_size=frame_size, augment=True)
    val_dataset = RawVideoDataset(root_dir, split='val', num_frames=num_frames,
                                 frame_size=frame_size, augment=False)
    test_dataset = RawVideoDataset(root_dir, split='test', num_frames=num_frames,
                                  frame_size=frame_size, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

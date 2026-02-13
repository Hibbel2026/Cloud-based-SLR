"""
Quick test script to verify skeleton drawing works correctly.
Run this to see example skeleton frames before training!
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '/Users/belhajali/Desktop/Exjobb Master/SLR/Cloud-based-SLR/Cloud-based-SLR')

from src.skeleton_drawer import extract_skeleton_frames_from_video


def test_skeleton_drawing():
    """Test skeleton drawing on a sample video."""
    
    # Find a sample video
    data_dir = Path("/Users/belhajali/Desktop/Exjobb Master/SLR/Cloud-based-SLR/Cloud-based-SLR/data/WLASL_100")
    
    # Find first available video
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    videos = list(class_dir.glob("*.mp4"))
                    if videos:
                        video_path = str(videos[0])
                        class_name = class_dir.name
                        
                        print(f"\n{'='*70}")
                        print(f"Testing skeleton drawing")
                        print(f"{'='*70}")
                        print(f"Video: {video_path}")
                        print(f"Class: {class_name}")
                        print(f"Split: {split}")
                        
                        # Extract skeleton frames
                        print(f"\n🎥 Extracting skeleton frames...")
                        skeleton_frames, metadata = extract_skeleton_frames_from_video(
                            video_path,
                            num_frames=16,
                            frame_size=(224, 224)
                        )
                        
                        print(f"✓ Extracted {len(skeleton_frames)} frames")
                        print(f"  Shape: {skeleton_frames.shape}")
                        print(f"  Dtype: {skeleton_frames.dtype}")
                        
                        # Print detection stats
                        if metadata:
                            print(f"\n📊 Detection statistics:")
                            for i, meta in enumerate(metadata[:3]):  # First 3 frames
                                print(f"  Frame {i}: Hands={meta['num_hands']}, Pose={meta['pose_detected']}")
                        
                        # Save example frames
                        output_dir = Path("/tmp/skeleton_test")
                        output_dir.mkdir(exist_ok=True)
                        
                        for i in range(min(3, len(skeleton_frames))):
                            output_path = output_dir / f"skeleton_frame_{i:02d}.png"
                            cv2.imwrite(str(output_path), skeleton_frames[i])
                            print(f"\n✓ Saved example frame to {output_path}")
                        
                        print(f"\n{'='*70}")
                        print(f"✓ Test successful! Check /tmp/skeleton_test/ for output images")
                        print(f"{'='*70}\n")
                        return
    
    print("❌ No videos found in dataset")


if __name__ == "__main__":
    test_skeleton_drawing()

"""
=============================================================================
Sign Language Recognition - Preprocessing Module
=============================================================================
File: src/preprocessing.py
Description: Uses MediaPipe to extract hand and pose landmarks from images.
             Converts raw image data into feature vectors for the neural 
             network model.

Features extracted per frame:
- Hand landmarks: 21 points × 3 coordinates (x,y,z) × 2 hands = 126 features
- Pose landmarks: 33 points × 3 coordinates = 99 features
- Total: 225 features per frame

Author: Hiba
Project: Master's Thesis - AWS Deployment Comparison
Partner: Knightec AB
=============================================================================
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List
import urllib.request
import os

# Download model files if not present
def download_models():
    """Download MediaPipe model files."""
    models_dir = Path("models/mediapipe")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Hand landmarker model
    hand_model_path = models_dir / "hand_landmarker.task"
    if not hand_model_path.exists():
        print("Downloading hand landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, hand_model_path)
        print("Done!")
    
    # Pose landmarker model
    pose_model_path = models_dir / "pose_landmarker.task"
    if not pose_model_path.exists():
        print("Downloading pose landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, pose_model_path)
        print("Done!")
    
    return str(hand_model_path), str(pose_model_path)


class MediaPipeExtractor:
    """Extract hand and pose landmarks using MediaPipe Tasks API."""
    
    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.5):
        """
        Initialize MediaPipe extractors.
        """
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        # Download models
        hand_model_path, pose_model_path = download_models()
        
        # Hand landmarker options
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model_path),
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        
        # Pose landmarker options
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model_path),
            min_pose_detection_confidence=min_detection_confidence
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
        
        # Feature sizes
        # Hand: 21 landmarks × 3 coords (x,y,z) × 2 hands = 126
        # Pose: 33 landmarks × 3 coords = 99
        self.hand_feature_size = 21 * 3 * 2  # 126
        self.pose_feature_size = 33 * 3       # 99
        self.total_feature_size = self.hand_feature_size + self.pose_feature_size  # 225
        
        # Store mediapipe reference
        self.mp = mp
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract landmarks from a single image.
        """
        # Read image with MediaPipe
        mp_image = self.mp.Image.create_from_file(image_path)
        
        # Initialize feature vector with zeros
        features = np.zeros(self.total_feature_size, dtype=np.float32)
        
        # Extract hand landmarks
        hand_results = self.hand_detector.detect(mp_image)
        if hand_results.hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.hand_landmarks[:2]):
                start_idx = idx * 21 * 3
                for i, landmark in enumerate(hand_landmarks):
                    features[start_idx + i*3] = landmark.x
                    features[start_idx + i*3 + 1] = landmark.y
                    features[start_idx + i*3 + 2] = landmark.z
        
        # Extract pose landmarks
        pose_results = self.pose_detector.detect(mp_image)
        if pose_results.pose_landmarks:
            pose_start = self.hand_feature_size
            for i, landmark in enumerate(pose_results.pose_landmarks[0]):
                features[pose_start + i*3] = landmark.x
                features[pose_start + i*3 + 1] = landmark.y
                features[pose_start + i*3 + 2] = landmark.z
        
        return features
    
    def extract_sequence(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract landmarks from a sequence of images (one video).
        """
        features_list = []
        for path in image_paths:
            try:
                features = self.extract_features(path)
                features_list.append(features)
            except Exception as e:
                print(f"Warning: Could not process {path}: {e}")
                features_list.append(np.zeros(self.total_feature_size, dtype=np.float32))
        
        return np.array(features_list, dtype=np.float32)
    
    def close(self):
        """Release MediaPipe resources."""
        pass  # Tasks API handles cleanup automatically


def process_dataset(data_dir: str, output_dir: str, max_frames: int = 30):
    """
    Process entire dataset and save extracted features.
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if data directory exists
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_path}")
        return None, None, None
    
    # Initialize extractor
    print("Initializing MediaPipe extractor...")
    extractor = MediaPipeExtractor()
    
    # Get all classes (folders)
    classes = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(f"Found {len(classes)} classes")
    
    all_features = []
    all_labels = []
    class_names = []
    
    for class_idx, class_dir in enumerate(classes):
        class_name = class_dir.name
        class_names.append(class_name)
        
        # Get all video folders for this class
        video_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])
        
        print(f"Processing class {class_idx + 1}/{len(classes)}: {class_name} ({len(video_dirs)} videos)")
        
        for video_dir in video_dirs:
            # Get all frames for this video
            frames = sorted(list(video_dir.glob("*.jpg")))
            
            if len(frames) == 0:
                continue
            
            # Sample frames if too many
            if len(frames) > max_frames:
                indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
                frames = [frames[i] for i in indices]
            
            # Extract features for this sequence
            frame_paths = [str(f) for f in frames]
            sequence_features = extractor.extract_sequence(frame_paths)
            
            # Pad if necessary
            if len(sequence_features) < max_frames:
                padding = np.zeros((max_frames - len(sequence_features), extractor.total_feature_size))
                sequence_features = np.vstack([sequence_features, padding])
            
            all_features.append(sequence_features)
            all_labels.append(class_idx)
    
    extractor.close()
    
    # Convert to numpy arrays
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    # Save processed data
    np.save(output_path / "X.npy", X)
    np.save(output_path / "y.npy", y)
    np.save(output_path / "class_names.npy", np.array(class_names))
    
    print(f"\nDone!")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Saved to: {output_path}")
    
    return X, y, class_names


# =============================================================================
# Main - Run this to process the dataset
# =============================================================================
if __name__ == "__main__":
    # Process training data
    print("="*50)
    print("Processing training data...")
    print("="*50)
    process_dataset(
        data_dir="data/preprocessing/train/frames",
        output_dir="data/processed/train",
        max_frames=30
    )
    
    # Process validation data
    print("\n" + "="*50)
    print("Processing validation data...")
    print("="*50)
    process_dataset(
        data_dir="data/preprocessing/val/frames",
        output_dir="data/processed/val",
        max_frames=30
    )
    
    # Process test data
    print("\n" + "="*50)
    print("Processing test data...")
    print("="*50)
    process_dataset(
        data_dir="data/preprocessing/test/frames",
        output_dir="data/processed/test",
        max_frames=30
    )
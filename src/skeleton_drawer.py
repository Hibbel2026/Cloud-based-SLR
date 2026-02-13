"""
=============================================================================
Sign Language Recognition - Skeleton Drawing Module
=============================================================================
File: src/skeleton_drawer.py
Description: Draws MediaPipe hand and pose landmarks on video frames.
             Creates skeleton visualizations for CNN training.

Output: RGB frames with hand/pose skeletts drawn in different colors
- Hand landmarks: Green circles + lines
- Pose landmarks: Blue circles + lines
- White background for clean visualization

=============================================================================
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request


def download_models():
    """Download MediaPipe model files if not present."""
    models_dir = Path("models/mediapipe")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    hand_model_path = models_dir / "hand_landmarker.task"
    if not hand_model_path.exists():
        print("Downloading hand landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, hand_model_path)
        print("✓ Hand model downloaded")
    
    pose_model_path = models_dir / "pose_landmarker.task"
    if not pose_model_path.exists():
        print("Downloading pose landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, pose_model_path)
        print("✓ Pose model downloaded")
    
    return str(hand_model_path), str(pose_model_path)


class SkeletonDrawer:
    """Draw MediaPipe landmarks on frames."""
    
    # Hand skeleton connections (indices of landmarks to connect)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),           # Index
        (0, 9), (9, 10), (10, 11), (11, 12),      # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),    # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
        (5, 9), (9, 13), (13, 17)                 # Palm
    ]
    
    # Pose skeleton connections
    POSE_CONNECTIONS = [
        (11, 13), (13, 15),                        # Right arm
        (12, 14), (14, 16),                        # Left arm
        (11, 12),                                  # Shoulders
        (11, 23), (12, 24),                        # Torso
        (23, 25), (25, 27), (27, 29), (29, 31),   # Right leg
        (24, 26), (26, 28), (28, 30), (30, 32),   # Left leg
    ]
    
    def __init__(self, frame_width: int = 224, frame_height: int = 224, 
                 background_color: Tuple[int, int, int] = (255, 255, 255)):
        """
        Initialize skeleton drawer.
        
        Args:
            frame_width: Output frame width
            frame_height: Output frame height
            background_color: Background color (B, G, R)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.background_color = background_color
        
        # Initialize MediaPipe detectors
        hand_model_path, pose_model_path = download_models()
        
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=hand_model_path,
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            num_hands=2,
            min_hand_detection_confidence=0.5
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)
        
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=pose_model_path,
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            min_pose_detection_confidence=0.5
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)
    
    def draw_skeleton(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Draw skeleton on a frame.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            skeleton_frame: Frame with drawn skeleton (white background)
            metadata: Detection metadata (num_hands, pose_detected, etc.)
        """
        # Create white background
        skeleton_frame = np.ones((self.frame_height, self.frame_width, 3), 
                                 dtype=np.uint8) * 255
        
        # Resize input frame for consistent processing
        resized_frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        metadata = {
            'num_hands': 0,
            'pose_detected': False,
            'hand_confidence': [],
            'pose_confidence': None
        }
        
        # Detect hands
        try:
            hand_result = self.hand_detector.detect(mp_image)
            if hand_result.handedness:
                metadata['num_hands'] = len(hand_result.handedness)
                
                for hand_idx, hand_landmarks in enumerate(hand_result.landmarks):
                    metadata['hand_confidence'].append(
                        float(hand_result.handedness[hand_idx].score)
                    )
                    self._draw_hand_skeleton(skeleton_frame, hand_landmarks)
        except Exception as e:
            pass  # Silently handle detection errors
        
        # Detect pose
        try:
            pose_result = self.pose_detector.detect(mp_image)
            if pose_result.pose_landmarks:  # Use pose_landmarks instead of landmarks
                metadata['pose_detected'] = True
                metadata['pose_confidence'] = float(
                    pose_result.pose_landmarks[0][0].presence if pose_result.pose_landmarks[0] else 0
                )
                self._draw_pose_skeleton(skeleton_frame, pose_result.pose_landmarks[0])
        except Exception as e:
            pass  # Silently handle detection errors
        
        return skeleton_frame, metadata
    
    def _draw_hand_skeleton(self, frame: np.ndarray, hand_landmarks: List):
        """Draw hand skeleton with connections."""
        # Draw connections (lines)
        for start_idx, end_idx in self.HAND_CONNECTIONS:
            if start_idx >= len(hand_landmarks) or end_idx >= len(hand_landmarks):
                continue
            
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            
            start_pos = (int(start.x * self.frame_width), 
                        int(start.y * self.frame_height))
            end_pos = (int(end.x * self.frame_width), 
                      int(end.y * self.frame_height))
            
            cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)  # Green
        
        # Draw landmarks (circles)
        for landmark in hand_landmarks:
            pos = (int(landmark.x * self.frame_width), 
                  int(landmark.y * self.frame_height))
            cv2.circle(frame, pos, 3, (0, 255, 0), -1)  # Green filled circle
    
    def _draw_pose_skeleton(self, frame: np.ndarray, pose_landmarks: List):
        """Draw pose skeleton with connections."""
        # Draw connections (lines)
        for start_idx, end_idx in self.POSE_CONNECTIONS:
            if start_idx >= len(pose_landmarks) or end_idx >= len(pose_landmarks):
                continue
            
            start = pose_landmarks[start_idx]
            end = pose_landmarks[end_idx]
            
            # Skip if confidence is too low
            if start.presence < 0.3 or end.presence < 0.3:
                continue
            
            start_pos = (int(start.x * self.frame_width), 
                        int(start.y * self.frame_height))
            end_pos = (int(end.x * self.frame_width), 
                      int(end.y * self.frame_height))
            
            cv2.line(frame, start_pos, end_pos, (255, 0, 0), 2)  # Blue
        
        # Draw landmarks (circles)
        for landmark in pose_landmarks:
            if landmark.presence < 0.3:
                continue
            
            pos = (int(landmark.x * self.frame_width), 
                  int(landmark.y * self.frame_height))
            cv2.circle(frame, pos, 2, (255, 0, 0), -1)  # Blue filled circle


def extract_skeleton_frames_from_video(
    video_path: str,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (224, 224)
) -> Tuple[np.ndarray, dict]:
    """
    Extract skeleton frames from a video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (uniformly sampled)
        frame_size: Size of output frames
    
    Returns:
        skeleton_frames: Array of shape (num_frames, height, width, 3)
        metadata: Detection metadata for all frames
    """
    drawer = SkeletonDrawer(frame_width=frame_size[0], frame_height=frame_size[1])
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return np.array([]), {}
    
    # Calculate frame indices to extract (uniform sampling)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    skeleton_frames = []
    all_metadata = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Draw skeleton
        skeleton_frame, metadata = drawer.draw_skeleton(frame)
        skeleton_frames.append(skeleton_frame)
        all_metadata.append(metadata)
    
    cap.release()
    
    return np.array(skeleton_frames, dtype=np.uint8), all_metadata

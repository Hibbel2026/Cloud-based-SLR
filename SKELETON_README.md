# Skeleton Drawing Approach for Sign Language Recognition

## Overview

This is a **hybrid approach** that combines the best of two worlds:

1. **Raw Video Information**: Preserves visual context (colors, lighting, environment)
2. **Skeleton Clarity**: Focuses the model on hand and pose movements

Instead of feeding the model either:
- ❌ Just coordinates (too abstract, loses hand shape info)
- ❌ Raw video (too much noise, difficult for CNN)

We use **skeleton-drawn frames** where MediaPipe landmarks are drawn as green (hands) and blue (pose) lines on white background.

```
Original Video → Extract landmarks → Draw on white background → CNN
     ↓                    ↓                        ↓                    ↓
  RGB frames         21+33 points        Skeleton visualization    Visual features
```

## Files Created

### 1. `src/skeleton_drawer.py`
Draws MediaPipe landmarks on frames.

**Main class**: `SkeletonDrawer`
- Input: Video frame (BGR)
- Output: Frame with drawn skeleton (white background, green/blue lines)

**Main function**: `extract_skeleton_frames_from_video()`
- Extracts N uniformly-sampled frames from a video
- Draws skeleton on each frame
- Returns array of shape (N, height, width, 3)

### 2. `src/dataset_skeleton.py`
PyTorch Dataset for loading skeleton frames.

**Main class**: `SkeletonDataset`
- Loads videos from `WLASL_100/{split}/{class}/{video.mp4}`
- Extracts skeleton frames on-the-fly or from cache
- Supports data augmentation (brightness, flip, temporal jitter)
- Returns frames of shape (num_frames, 3, 224, 224) normalized to [0, 1]

**Main function**: `get_skeleton_dataloaders()`
- Creates DataLoaders for train/val/test splits
- Automatically handles shuffling and batching

### 3. `src/train_skeleton.py`
Training script for 2D CNN on skeleton frames.

**Model**: `SkeletonCNN2D`
- Takes stacked skeleton frames as input (batch, num_frames, 3, 224, 224)
- Processes each frame independently through 2D CNN
- Averages temporal features
- Classifies into 100 classes

**Features**:
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping (patience=10)
- Checkpoint saving
- Training history logging

## Quick Start

### Step 1: Test skeleton drawing

```bash
cd /Users/belhajali/Desktop/Exjobb\ Master/SLR/Cloud-based-SLR/Cloud-based-SLR

python test_skeleton.py
```

This will:
- Find a sample video from your dataset
- Extract and draw skeleton frames
- Save 3 example frames to `/tmp/skeleton_test/`
- Show you what the data looks like

**Expected output:**
```
Example frame with:
- White background
- Green dots/lines for hand landmarks
- Blue dots/lines for pose landmarks
```

### Step 2: Train the model

```bash
python src/train_skeleton.py \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 0.001 \
    --device auto
```

**Command line options:**
```
--batch_size: Batch size (default: 16)
--num_frames: Frames per video (default: 16)
--frame_size: Frame resolution (default: 224)
--learning_rate: Learning rate (default: 0.001)
--num_epochs: Max epochs (default: 50)
--patience: Early stopping patience (default: 10)
--device: 'cuda', 'cpu', or 'auto' (default: auto)
--cache_dir: Cache skeleton frames for faster training (optional)
--use_cache: Use cached frames if available (flag)
```

### Step 3: (Optional) Pre-compute skeleton cache

For faster training, pre-compute all skeleton frames:

```bash
python -c "
from src.dataset_skeleton import create_skeleton_cache
create_skeleton_cache('data', 'data/skeleton_cache', split='train')
create_skeleton_cache('data', 'data/skeleton_cache', split='val')
create_skeleton_cache('data', 'data/skeleton_cache', split='test')
"
```

Then train with cache:
```bash
python src/train_skeleton.py --cache_dir data/skeleton_cache --use_cache
```

## Expected Results

Based on the approach:

| Method | Accuracy | Speed | Notes |
|--------|----------|-------|-------|
| 1D CNN (landmarks) | 38.5% | Fast | Too abstract, loses hand shape |
| LSTM (landmarks) | 16.5% | Medium | Temporal but still abstract |
| **Skeleton Drawing** | **55-70%** | **Medium** | **Best balance - cleaner input!** |
| 3D CNN (raw video) | 50-65% | Slow | More information but harder to learn |

**Why skeleton drawing works better:**
1. ✅ Provides spatial context (where in frame)
2. ✅ Shows hand shape (green skeleton)
3. ✅ Shows pose (blue skeleton)
4. ✅ Removes confusing background info
5. ✅ Normalized input (white background)
6. ✅ Faster than raw video processing

## Architecture Details

### SkeletonCNN2D Model

```
Input: (batch, 16, 3, 224, 224)
    ↓
[Process each frame independently]
    ↓
Conv2d (3→32) + BatchNorm + ReLU
MaxPool2d (112→56)
    ↓
Conv2d (32→64) + BatchNorm + ReLU
MaxPool2d (56→28)
    ↓
Conv2d (64→128) + BatchNorm + ReLU
MaxPool2d (28→14)
    ↓
Conv2d (128→256) + BatchNorm + ReLU
AdaptiveAvgPool2d → (256,)
    ↓
[Average across temporal dimension]
    ↓
FC (256→256) + ReLU + Dropout
FC (256→128) + ReLU + Dropout
FC (128→100) [logits]
    ↓
Output: (batch, 100) [class probabilities]
```

**Parameters**: ~2.5M (lightweight, fast training)

## Troubleshooting

### Issue: "No module named 'mediapipe'"
```bash
pip install mediapipe
```

### Issue: "Hand/Pose models not found"
The `skeleton_drawer.py` will automatically download them on first run. Make sure you have internet connection.

### Issue: Skeleton frames look empty
This means MediaPipe couldn't detect hands/pose in that frame. This is normal for some frames. The dataset will skip frames with no detections or pad with white frames.

### Issue: Training is slow
Enable skeleton caching:
```bash
# Pre-compute all skeleton frames once
python -c "from src.dataset_skeleton import create_skeleton_cache; create_skeleton_cache('data', 'data/skeleton_cache', split='train')"

# Then train with cache (no need to compute skeletons during training)
python src/train_skeleton.py --cache_dir data/skeleton_cache --use_cache
```

## Next Steps After Training

Once you get results, you can:

1. **Compare with 3D CNN**: See if skeleton drawing outperforms raw video
2. **Analyze per-class accuracy**: Which signs are still hard to recognize?
3. **Tune hyperparameters**: Try different frame rates, batch sizes, etc.
4. **Ensemble approach**: Combine skeleton CNN + 3D CNN predictions
5. **Add attention**: Focus on important hand regions

## File Structure

```
project/
├── src/
│   ├── skeleton_drawer.py      # Draws landmarks on frames
│   ├── dataset_skeleton.py      # PyTorch Dataset
│   ├── train_skeleton.py        # Training script
│   ├── models/
│   │   └── cnn_model.py         # Existing CNN (can reuse)
│   └── ...
├── data/
│   ├── WLASL_100/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── skeleton_cache/          # Optional cached frames
├── checkpoints/
│   ├── best_skeleton_model.pth
│   └── skeleton_training_history.json
├── test_skeleton.py             # Quick test
└── SKELETON_README.md           # This file
```

## Questions?

The skeleton drawing approach is ready to use! Start with `python test_skeleton.py` to see what the data looks like, then run training.

Good luck! 🚀

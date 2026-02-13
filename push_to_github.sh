#!/bin/bash
# Push 3D CNN implementation to GitHub and prepare for Colab training

echo "=== Committing 3D CNN Implementation ==="
echo ""

# Add new files
echo "📝 Adding new files..."
git add src/models/cnn3d_model.py
git add src/dataset_3dcnn.py
git add src/train_3dcnn.py
git add COLAB_TRAINING.md
git add README.md

# Commit
echo "💾 Committing..."
git commit -m "feat: Add 3D CNN model for raw video processing

- Implement Temporal3DCNN for direct video frame processing
- Add RawVideoDataset for WLASL-100 video loading
- Create train_3dcnn.py for GPU-enabled training
- Support for Colab integration with GPU acceleration
- Include both full 3D CNN and lightweight variant"

# Push
echo "🚀 Pushing to GitHub..."
git push origin Baseline-ML

echo ""
echo "✅ Successfully pushed to GitHub!"
echo ""
echo "Next steps:"
echo "1. Open Colab: https://colab.research.google.com/"
echo "2. Create new notebook"
echo "3. Run cells in COLAB_TRAINING.md"
echo "4. After training, results will be in checkpoints/best_temporal3dcnn_model.pth"

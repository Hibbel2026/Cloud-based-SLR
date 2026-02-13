# Sign Language Recognition - ML Models & GPU Training

Master's thesis project on Sign Language Recognition (SLR) using deep learning models trained on raw video frames.

## Current Status

**Latest Models Implemented:**
- ✅ **1D CNN**: 38.5% accuracy on WLASL-100 (100 classes)
- ✅ **LSTM**: 16.5% accuracy on processed landmarks
- ✅ **3D CNN**: In development - training on raw video frames with GPU in Colab

## Models

### 1. MediaPipe Landmark-Based Models

#### 1D CNN (TemporalCNN)
```python
python3 src/train.py cnn
```
- **Input**: [batch, 48, 225] (225-dim landmarks: 21 hand × 3 × 2 + 33 pose × 3)
- **Architecture**: 3 Conv1d blocks with batch norm
- **Accuracy**: 38.5% on WLASL-100
- **Speed**: Fast inference (~2-5ms)

#### LSTM (TemporalLSTM)
```python
python3 src/train.py lstm
```
- **Input**: [batch, 48, 225] (temporal sequences)
- **Architecture**: 1-layer bidirectional LSTM (128 hidden units)
- **Accuracy**: 16.5% on WLASL-100 (needs hyperparameter tuning)
- **Speed**: Medium inference (~8-15ms)

### 2. Raw Video 3D CNN (NEW)

#### 3D CNN (Temporal3DCNN)
```python
# On Colab with GPU:
python3 src/train_3dcnn.py --model 3dcnn --batch_size 8 --num_frames 16 --device cuda
```
- **Input**: [batch, 3, frames, height, width] = [batch, 3, 16, 224, 224]
- **Architecture**: 4 Conv3d blocks with spatial-temporal feature extraction
- **Training**: Requires GPU (Colab recommended)
- **Expected**: Better accuracy than landmark-based models (faster convergence on raw RGB data)
│   ├── containers/
│   └── hybrid/
└── evaluation/         # Benchmarking and metrics collection
```

## Setup

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
```bash
python scripts/download_data.py
```

### 4. Extract features (MediaPipe)
```bash
python scripts/extract_features.py
```

### 5. Train baseline model
```bash
python scripts/train_model.py
```

## Methodology

**Phase 1**: Data processing and feature extraction using MediaPipe  
**Phase 2**: MLP model training and validation  
**Phase 3**: AWS deployment across three architectures  
**Phase 4**: Benchmarking and comparative evaluation

## Author
Hibatallah Belhajali  
Master's Thesis Project - KTH Royal Institute of Technology  
In partnership with Knightec AB

**Supervisor**: Julius Jensen (Knightec AB)
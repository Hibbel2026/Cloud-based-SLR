# Sign Language Recognition - AWS Infrastructure Comparison

Master's thesis project comparing AWS cloud training and inference infrastructures for deep learning-based sign language recognition.

## Project Overview

This project evaluates two AWS infrastructure approaches for training and deploying a deep learning model for American Sign Language (ASL) recognition:

- **Amazon EC2 (g5.xlarge)** — Self-managed Infrastructure-as-a-Service
- **AWS SageMaker (ml.g5.xlarge)** — Managed Platform-as-a-Service

Both platforms use identical GPU hardware (NVIDIA A10G) to isolate the effect of infrastructure management overhead on training time, cost, and inference performance.

### Research Questions

**RQ1 – Training:** What are the trade-offs in training time and cost efficiency between EC2 and SageMaker for training deep learning models for sign language recognition?

**RQ2 – Inference:** What are the trade-offs in inference latency, cost efficiency, and cold start time between EC2 and SageMaker for deploying deep learning models for real-time sign language recognition?

### Metrics Evaluated

**Training phase:**
- Total training time per run
- Training cost per run
- Training stability (variance across 10 repeated runs)
- Operational complexity (setup steps, configuration effort)

**Inference phase:**
- Inference latency (mean, median, P95)
- Cold start time
- Cost per inference request
- Throughput

## Model Architecture

The project uses a hybrid **2D CNN–LSTM** model for video-based sign language recognition:

- **CNN backbone**: ResNet50 (pretrained on ImageNet, last block fine-tuned)
- **Temporal model**: 2-layer LSTM (hidden size 512)
- **Input**: Sequence of 24 frames per video (224×224 px)
- **Output**: 100-class ASL word classification

## Dataset

- **Source**: [American Sign Language Dataset](https://huggingface.co/datasets/akasheroor/American-Sign-Language-Dataset)
- **Subset**: Top 100 ASL words by video count
- **Split**: 21 train / 5 val / 5 test videos per word
- **Total**: ~3,100 videos

## Project Structure

```
Cloud-based-SLR/
├── data_exploration/        # Dataset analysis and top-100 selection
├── models/
│   ├── cnn_lstm.py          # Model architecture
│   ├── train_cnn_lstm_SM.py # SageMaker training script
│   └── Launcher.py          # SageMaker job launcher
├── scripts/
│   ├── extract_frames.py    # Frame extraction from videos
│   └── train_cnn_lstm.py    # EC2 training script
├── src/                     # Core ML modules
├── outputs/                 # Training results and epoch logs (JSON)
└── requirements.txt
```

## Setup

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare dataset
Download the ASL dataset and run:
```bash
python data_exploration/build_top100_dataset.py
python data_exploration/split_dataset_balanced.py
python scripts/extract_frames.py
```

### 4. Train on EC2
```bash
python scripts/train_cnn_lstm.py
```

### 5. Train on SageMaker
```bash
cd models
python Launcher.py
```



## Author

Hibatallah Belhajali — Master's Thesis in Computer Science, KTH Royal Institute of Technology  
Industry partner: Knightec AB  
Supervisor: Julius Jensen (Knightec AB)

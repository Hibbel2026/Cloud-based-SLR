# Sign Language Recognition - AWS Deployment Comparison

Master's thesis project comparing AWS cloud deployment architectures for sign language recognition systems.

## Project Overview

This project evaluates three AWS deployment options for ML inference workloads:
- **Amazon SageMaker** - Managed ML platform
- **Amazon EC2** - Virtual server instances
- **AWS Fargate** - Serverless containers

### Metrics Evaluated
- Latency (inference time)
- Cost (per inference, monthly)
- Reliability (uptime, error rates)
- Operational complexity

## Tech Stack
- **Dataset**: WLASL-100 (Word-Level American Sign Language)
- **Feature Extraction**: MediaPipe
- **Model**: Neural Network Classifier
- **Cloud**: AWS (SageMaker, EC2, Fargate)

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

### 3. Download dataset
```bash
python scripts/download_data.py
```

## Author
Hibatallah Belhajali - Master's Thesis Project in partnership with Knightec AB

Supervisor: Julius Jensen (Knightec AB)
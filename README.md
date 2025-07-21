# MLOps Assignment 3 - End-to-End MLOps Pipeline

## Overview
This repository contains a complete MLOps pipeline implementation including:
- Linear Regression model training on California Housing dataset
- PyTorch neural network with manual weight transfer
- Docker containerization
- CI/CD pipeline with GitHub Actions
- Manual quantization optimization

## Repository Structure
```
├── src/
│   ├── train.py          # Model training script
│   ├── predict.py        # Model prediction/verification script
│   └── quantize.py       # Quantization implementation
├── models/               # Saved models directory
├── .github/
│   └── workflows/
│       └── ci.yml        # CI/CD pipeline
├── Dockerfile           # Container configuration
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Branches
- `main`: Initial setup and documentation
- `dev`: Model development and training
- `docker_ci`: Docker containerization and CI/CD
- `quantization`: Model optimization and quantization

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python src/train.py`
4. Run prediction: `python src/predict.py`

## Results Comparison

| Metric | Original Sklearn Model | Quantized Model |
|--------|----------------------|-----------------|
| R² Score | TBD | TBD |
| Model Size | TBD KB | TBD KB |

## Docker Usage
```bash
# Build the image
docker build -t mlops-pipeline .

# Run the container
docker run mlops-pipeline python src/predict.py
```
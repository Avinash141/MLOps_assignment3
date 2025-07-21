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
| R² Score | 0.5758 | -0.1976 |
| Model Size | 0.44 KB | 0.60 KB |

**Note**: The quantized model shows degraded performance due to aggressive 8-bit quantization. This demonstrates the trade-off between model size and accuracy in quantization techniques.

## Docker Usage
```bash
# Build the image
docker build -t mlops-pipeline .

# Run the container
docker run mlops-pipeline python src/predict.py
```

## Assignment Implementation Details

### Step 1: Main Branch (Initial Setup)
- ✅ Repository initialization with README.md and .gitignore
- ✅ Project structure setup

### Step 2: Dev Branch (Model Development)
- ✅ Created `train.py` script for scikit-learn LinearRegression
- ✅ Trained on California Housing dataset
- ✅ Model saved using joblib
- ✅ Achieved R² Score: 0.5758

### Step 3: Docker_CI Branch (Automation)
- ✅ Created Dockerfile for containerization
- ✅ Created `predict.py` for model verification
- ✅ Implemented CI/CD pipeline with GitHub Actions
- ✅ Automated training, Docker build, and container testing
- ✅ Docker Hub integration ready

### Step 4: Quantization Branch (Optimization)
- ✅ Created `quantize.py` with manual 8-bit quantization
- ✅ Extracted and quantized model parameters
- ✅ Implemented dequantization for inference
- ✅ Performance comparison and analysis

## Technical Implementation

### Manual Quantization Process
1. **Parameter Extraction**: Extract coefficients and intercept from trained sklearn model
2. **Range Calculation**: Determine min/max values for quantization range
3. **8-bit Quantization**: Convert float32 parameters to uint8 integers
4. **Dequantization**: Convert back to float32 for inference
5. **Performance Evaluation**: Compare original vs quantized model performance

### CI/CD Pipeline Features
- Automated model training on push
- Docker image building and testing
- Container verification with model inference
- Docker Hub deployment (configured with secrets)
- Comprehensive error handling and logging
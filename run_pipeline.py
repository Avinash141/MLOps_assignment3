#!/usr/bin/env python3
"""
Complete MLOps Pipeline Demonstration Script
This script demonstrates the entire pipeline from training to quantization
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    """Run the complete MLOps pipeline"""
    print("MLOps Assignment 3 - Complete Pipeline Demonstration")
    print("This script will run the entire pipeline step by step")
    
    # Ensure we're in the right directory
    if not os.path.exists("src/train.py"):
        print("ERROR: Please run this script from the project root directory")
        sys.exit(1)
    
    # Step 1: Train the model
    if not run_command("python src/train.py", "Training the Linear Regression Model"):
        print("Training failed. Exiting.")
        return
    
    # Step 2: Verify the model
    if not run_command("python src/predict.py", "Verifying the Trained Model"):
        print("Model verification failed. Exiting.")
        return
    
    # Step 3: Run quantization
    if not run_command("python src/quantize.py", "Running Manual Quantization"):
        print("Quantization failed. Exiting.")
        return
    
    # Step 4: Show model files
    print(f"\n{'='*60}")
    print("FINAL RESULTS: Generated Model Files")
    print(f"{'='*60}")
    
    if os.path.exists("models"):
        for file in os.listdir("models"):
            filepath = os.path.join("models", file)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"📁 {file}: {size_kb:.2f} KB")
    
    print(f"\n{'='*60}")
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("All components of the MLOps pipeline have been executed.")
    print("Check the models/ directory for generated files.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
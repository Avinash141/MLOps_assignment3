"""
Training script for California Housing dataset using scikit-learn LinearRegression
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def load_data():
    """Load and prepare the California Housing dataset"""
    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {housing.feature_names}")
    
    return X, y, housing.feature_names

def train_model(X_train, y_train):
    """Train a Linear Regression model"""
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"Model coefficients shape: {model.coef_.shape}")
    print(f"Model intercept: {model.intercept_}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return mse, r2

def save_model(model, filepath):
    """Save the trained model using joblib"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")

def main():
    """Main training pipeline"""
    print("=== MLOps Assignment 3: Model Training ===")
    
    # Load data
    X, y, feature_names = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    mse, r2 = evaluate_model(model, X_test, y_test)
    
    # Save model
    model_path = "models/california_housing_model.joblib"
    save_model(model, model_path)
    
    # Save test data for later use
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names
    }
    joblib.dump(test_data, "models/test_data.joblib")
    print("Test data saved to: models/test_data.joblib")
    
    print("\n=== Training Complete ===")
    print(f"Final R² Score: {r2:.4f}")

if __name__ == "__main__":
    main()
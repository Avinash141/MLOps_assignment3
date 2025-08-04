"""
Prediction script for model verification in Docker container
"""

import numpy as np
import joblib
import os
from sklearn.metrics import r2_score, mean_squared_error

def load_model_and_data():
    """Load the trained model and test data"""
    model_path = "models/california_housing_model.joblib"
    test_data_path = "models/test_data.joblib"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")
    
    print("Loading trained model...")
    model = joblib.load(model_path)
    
    print("Loading test data...")
    test_data = joblib.load(test_data_path)
    
    return model, test_data

def verify_model(model, test_data):
    """Verify the model by running predictions on test data"""
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    print(f"Running predictions on {X_test.shape[0]} test samples...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Model Verification Results:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    
    # Verify model parameters
    print(f"\nModel Parameters:")
    print(f"Coefficients shape: {model.coef_.shape}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"First 3 coefficients: {model.coef_[:3]}")
    
    return r2, mse

def main():
    """Main verification pipeline"""
    print("=== Docker Container Model Verification ===")
    
    try:
        # Load model and data
        model, test_data = load_model_and_data()
        
        # Verify model
        r2, mse = verify_model(model, test_data)
        
        # Success criteria
        if r2 > 0.5:  # Reasonable R² score for this dataset
            print(f"\n✅ Model verification PASSED!")
            print(f"R² Score ({r2:.4f}) meets minimum threshold (0.5)")
        else:
            print(f"\n❌ Model verification FAILED!")
            print(f"R² Score ({r2:.4f}) below minimum threshold (0.5)")
            exit(1)
            
    except Exception as e:
        print(f"\n❌ Model verification FAILED with error: {str(e)}")
        exit(1)
    
    print("=== Verification Complete ===")

if __name__ == "__main__":
    main()
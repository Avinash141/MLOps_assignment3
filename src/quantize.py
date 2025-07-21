"""
Manual quantization script for Linear Regression model parameters
"""

import numpy as np
import joblib
import os
from sklearn.metrics import r2_score, mean_squared_error

def load_sklearn_model(model_path):
    """Load the trained scikit-learn model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading scikit-learn model from: {model_path}")
    model = joblib.load(model_path)
    
    print(f"Model coefficients shape: {model.coef_.shape}")
    print(f"Model intercept: {model.intercept_}")
    
    return model

def extract_parameters(model):
    """Extract parameters from scikit-learn model"""
    coefficients = model.coef_.copy()
    intercept = model.intercept_
    
    print(f"Extracted coefficients: {coefficients}")
    print(f"Extracted intercept: {intercept}")
    
    return coefficients, intercept

def manual_quantization(params, num_bits=16):
    """
    Manually quantize parameters to unsigned 8-bit integers
    
    Quantization formula:
    - Find min and max values
    - Scale = (max - min) / (2^num_bits - 1)
    - Quantized = round((value - min) / scale)
    - Dequantized = quantized * scale + min
    """
    print(f"\nPerforming manual {num_bits}-bit quantization...")
    
    # Flatten all parameters for global min/max
    flat_params = params.flatten()
    
    # Find global min and max
    param_min = float(np.min(flat_params))
    param_max = float(np.max(flat_params))
    
    print(f"Parameter range: [{param_min:.6f}, {param_max:.6f}]")
    
    # Handle case where min == max (single value)
    if param_max == param_min:
        print("Single value detected, using symmetric quantization around zero")
        # For single values, use a small range around the value
        abs_val = abs(param_min)
        if abs_val == 0:
            param_min, param_max = -1e-6, 1e-6
        else:
            param_min, param_max = param_min - abs_val * 0.01, param_max + abs_val * 0.01
        print(f"Adjusted range: [{param_min:.6f}, {param_max:.6f}]")
    
    # Calculate scale and zero point
    scale = (param_max - param_min) / (2**num_bits - 1)
    zero_point = param_min
    
    print(f"Quantization scale: {scale:.8f}")
    print(f"Zero point: {zero_point:.6f}")
    
    # Quantize: convert to integers
    if num_bits <= 8:
        quantized = np.round((params - zero_point) / scale).astype(np.uint8)
    elif num_bits <= 16:
        quantized = np.round((params - zero_point) / scale).astype(np.uint16)
    else:
        quantized = np.round((params - zero_point) / scale).astype(np.uint32)
    
    print(f"Quantized values range: [{np.min(quantized)}, {np.max(quantized)}]")
    
    # Store quantization parameters
    quant_params = {
        'quantized_values': quantized,
        'scale': scale,
        'zero_point': zero_point,
        'original_shape': params.shape,
        'num_bits': num_bits
    }
    
    return quant_params

def dequantize_parameters(quant_params):
    """Dequantize parameters back to float32"""
    quantized_values = quant_params['quantized_values']
    scale = quant_params['scale']
    zero_point = quant_params['zero_point']
    original_shape = quant_params['original_shape']
    
    # Dequantize: convert back to float
    dequantized = quantized_values.astype(np.float32) * scale + zero_point
    
    # Reshape to original shape
    dequantized = dequantized.reshape(original_shape)
    
    print(f"Dequantized values range: [{np.min(dequantized):.6f}, {np.max(dequantized):.6f}]")
    
    return dequantized

def calculate_model_size(filepath):
    """Calculate model file size in KB"""
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        size_kb = size_bytes / 1024
        return size_kb
    return 0

def evaluate_quantized_model(X_test, y_test, dequant_coef, dequant_intercept):
    """Evaluate the dequantized model"""
    # Manual prediction using dequantized parameters
    y_pred = np.dot(X_test, dequant_coef) + dequant_intercept
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    return r2, mse

def main():
    """Main quantization pipeline"""
    print("=== MLOps Assignment 3: Model Quantization ===")
    
    # Ensure we have the trained model
    model_path = "models/california_housing_model.joblib"
    test_data_path = "models/test_data.joblib"
    
    # Train model if it doesn't exist
    if not os.path.exists(model_path):
        print("Model not found. Training model first...")
        os.system("python src/train.py")
    
    # Load the scikit-learn model
    model = load_sklearn_model(model_path)
    
    # Load test data
    test_data = joblib.load(test_data_path)
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    # Extract parameters
    coefficients, intercept = extract_parameters(model)
    
    # Create unquantized parameters dictionary
    unquant_params = {
        'coefficients': coefficients,
        'intercept': intercept,
        'model_type': 'linear_regression'
    }
    
    # Save unquantized parameters
    unquant_path = "models/unquant_params.joblib"
    joblib.dump(unquant_params, unquant_path)
    print(f"Unquantized parameters saved to: {unquant_path}")
    
    # Quantize coefficients and intercept separately for better precision
    print("\n--- Quantizing Coefficients ---")
    coef_quant_params = manual_quantization(coefficients, num_bits=8)
    
    print("\n--- Quantizing Intercept ---")
    # For intercept, use a small range around the value for 8-bit quantization
    intercept_array = np.array([intercept])
    intercept_range = abs(intercept) * 0.1  # 10% range around intercept
    intercept_min = intercept - intercept_range
    intercept_max = intercept + intercept_range
    
    # Manual quantization for intercept with custom range
    scale = (intercept_max - intercept_min) / (2**8 - 1)
    quantized_intercept = np.round((intercept - intercept_min) / scale).astype(np.uint8)
    
    intercept_quant_params = {
        'quantized_values': quantized_intercept.reshape(1),
        'scale': scale,
        'zero_point': intercept_min,
        'original_shape': (1,),
        'num_bits': 8
    }
    
    print(f"Intercept quantization - Scale: {scale:.8f}, Zero point: {intercept_min:.6f}")
    print(f"Quantized intercept value: {quantized_intercept.item()}")
    
    # Create quantized parameters dictionary
    quant_params = {
        'coefficients_quant': coef_quant_params,
        'intercept_quant': intercept_quant_params,
        'model_type': 'quantized_linear_regression'
    }
    
    # Save quantized parameters
    quant_path = "models/quant_params.joblib"
    joblib.dump(quant_params, quant_path)
    print(f"\nQuantized parameters saved to: {quant_path}")
    
    # Dequantize for inference
    print("\n--- Dequantizing for Inference ---")
    dequant_coef = dequantize_parameters(coef_quant_params)
    dequant_intercept = dequantize_parameters(intercept_quant_params)[0]
    
    # Evaluate original model
    y_pred_original = model.predict(X_test)
    r2_original = r2_score(y_test, y_pred_original)
    mse_original = mean_squared_error(y_test, y_pred_original)
    
    # Evaluate quantized model
    r2_quantized, mse_quantized = evaluate_quantized_model(
        X_test, y_test, dequant_coef, dequant_intercept
    )
    
    # Calculate model sizes
    unquant_size = calculate_model_size(unquant_path)
    quant_size = calculate_model_size(quant_path)
    
    # Print comparison results
    print("\n" + "="*60)
    print("QUANTIZATION RESULTS COMPARISON")
    print("="*60)
    
    print(f"{'Metric':<25} {'Original Model':<20} {'Quantized Model':<20}")
    print("-" * 65)
    print(f"{'R² Score':<25} {r2_original:<20.4f} {r2_quantized:<20.4f}")
    print(f"{'MSE':<25} {mse_original:<20.4f} {mse_quantized:<20.4f}")
    print(f"{'Model Size (KB)':<25} {unquant_size:<20.2f} {quant_size:<20.2f}")
    print(f"{'Size Reduction':<25} {'-':<20} {((unquant_size-quant_size)/unquant_size*100):<19.1f}%")
    
    # Calculate quantization error
    coef_error = np.mean(np.abs(coefficients - dequant_coef))
    intercept_error = abs(intercept - dequant_intercept)
    
    print(f"\nQuantization Errors:")
    print(f"Average coefficient error: {coef_error:.6f}")
    print(f"Intercept error: {intercept_error:.6f}")
    
    print("\n=== Quantization Complete ===")
    
    # Return results for potential use in reports
    return {
        'original_r2': r2_original,
        'quantized_r2': r2_quantized,
        'original_size_kb': unquant_size,
        'quantized_size_kb': quant_size
    }

if __name__ == "__main__":
    main()
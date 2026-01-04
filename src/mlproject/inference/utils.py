"""
Utility functions for inference operations.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

def load_model(model_path: str) -> Any:
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to model file (.pkl, .joblib, or .pth)
        
    Returns:
        Loaded model object
        
    Raises:
        ValueError: If file format is not supported
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Determine loader based on extension
    if model_path.suffix == '.pkl':
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    elif model_path.suffix == '.joblib':
        return joblib.load(model_path)
    elif model_path.suffix == '.pth':
        import torch
        return torch.load(model_path)
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")

def save_predictions(predictions: pd.DataFrame, output_path: str, 
                     include_timestamp: bool = True) -> str:
    """
    Save predictions to CSV with proper formatting.
    
    Args:
        predictions: DataFrame with predictions
        output_path: Path where to save predictions
        include_timestamp: Whether to add timestamp to filename
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to filename if it's a directory
    if output_path.is_dir() or include_timestamp:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path / f"predictions_{timestamp}.csv"
    
    # Ensure .csv extension
    if output_path.suffix != '.csv':
        output_path = output_path.with_suffix('.csv')
    
    predictions.to_csv(output_path, index=False)
    return str(output_path)

def validate_inference_input(df: pd.DataFrame, 
                            required_columns: Tuple[str, ...] = None) -> bool:
    """
    Validate input data for inference.
    
    Args:
        df: Input DataFrame
        required_columns: Tuple of required column names
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if required_columns is None:
        required_columns = (
            'pickup_datetime',
            'pickup_latitude', 'pickup_longitude',
            'dropoff_latitude', 'dropoff_longitude'
        )
    
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check for NaN values in required columns
    for col in required_columns:
        if df[col].isna().any():
            raise ValueError(f"Column {col} contains NaN values")
    
    return True

def calculate_batch_metrics(predictions: np.ndarray, 
                           actuals: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate metrics for batch predictions.
    
    Args:
        predictions: Array of predictions
        actuals: Optional array of actual values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'count': len(predictions),
        'mean': float(np.mean(predictions)),
        'std': float(np.std(predictions)),
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'median': float(np.median(predictions))
    }
    
    if actuals is not None and len(actuals) == len(predictions):
        errors = predictions - actuals
        metrics.update({
            'mae': float(np.mean(np.abs(errors))),
            'mse': float(np.mean(errors ** 2)),
            'rmse': float(np.sqrt(metrics['mse'])),
            'r2_score': float(1 - np.sum(errors ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))
        })
    
    return metrics

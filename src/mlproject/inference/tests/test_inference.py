@'
"""
Tests for inference module.
"""

import pandas as pd
import numpy as np
import tempfile
import pickle
from pathlib import Path
import sys

sys.path.append('src')

from mlproject.inference import InferencePipeline, load_model, save_predictions

def test_inference_pipeline_creation():
    """Test creating an InferencePipeline instance."""
    # Create a mock model
    class MockModel:
        def predict(self, X):
            return np.array([1000.0, 1500.0, 2000.0])
    
    model = MockModel()
    pipeline = InferencePipeline(model=model)
    
    assert pipeline is not None
    assert pipeline.model is model
    print("✅ InferencePipeline creation test passed")

def test_inference_with_mock_data():
    """Test inference with mock data."""
    # Create mock model
    class MockModel:
        def predict(self, X):
            return np.ones(X.shape[0]) * 1000
    
    # Create sample data
    data = {
        'pickup_datetime': ['2025-03-18 10:30:00', '2025-03-18 11:15:00'],
        'pickup_latitude': [40.7, 40.75],
        'pickup_longitude': [-74.0, -73.98],
        'dropoff_latitude': [40.8, 40.8],
        'dropoff_longitude': [-74.0, -73.95],
        'vendor_id': [1, 2],
        'store_and_fwd_flag': ['N', 'Y']
    }
    df = pd.DataFrame(data)
    
    model = MockModel()
    pipeline = InferencePipeline(model=model)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "predictions.csv"
        results = pipeline.run(df, save_path=str(output_path))
        
        # Check results
        assert len(results) == 2
        assert 'prediction' in results.columns
        assert 'timestamp' in results.columns
        assert output_path.exists()
        
        # Load saved predictions
        saved = pd.read_csv(output_path)
        assert len(saved) == 2
        assert 'prediction' in saved.columns
    
    print("✅ Inference with mock data test passed")

def test_load_model():
    """Test model loading functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock model
        class MockModel:
            def __init__(self):
                self.coef_ = [1.0, 2.0]
        
        model = MockModel()
        
        # Save as pickle
        pkl_path = Path(tmpdir) / "model.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Load it back
        loaded = load_model(str(pkl_path))
        assert loaded.coef_ == [1.0, 2.0]
    
    print("✅ Model loading test passed")

def test_save_predictions():
    """Test predictions saving functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        predictions = pd.DataFrame({
            'id': [1, 2, 3],
            'prediction': [1000, 1500, 2000]
        })
        
        output_dir = Path(tmpdir) / "output"
        saved_path = save_predictions(predictions, str(output_dir))
        
        assert Path(saved_path).exists()
        assert saved_path.endswith('.csv')
        
        # Load and verify
        loaded = pd.read_csv(saved_path)
        assert len(loaded) == 3
        assert list(loaded.columns) == ['id', 'prediction']
    
    print("✅ Save predictions test passed")

def run_all_tests():
    """Run all inference tests."""
    print("🧪 Running inference module tests...")
    
    test_inference_pipeline_creation()
    test_inference_with_mock_data()
    test_load_model()
    test_save_predictions()
    
    print("\n🎉 All inference tests passed!")

if __name__ == "__main__":
    run_all_tests()
'@ | Set-Content tests/test_inference.py
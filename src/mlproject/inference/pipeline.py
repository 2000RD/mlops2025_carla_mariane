import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Optional
import sys

# Add src to path for imports
sys.path.append('src')

try:
    from mlproject.features.engineer import FeatureEngineer
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEER_AVAILABLE = False
    print("??  FeatureEngineer not found - using basic feature extraction")

try:
    from mlproject.train.trainer import ModelTrainer
    MODEL_TRAINER_AVAILABLE = True
except ImportError:
    MODEL_TRAINER_AVAILABLE = False

class InferencePipeline:
    """
    Handles batch inference for a trained model:
    - Applies feature engineering via FeatureEngineer class
    - Generates predictions
    - Optionally saves predictions to CSV
    """

    def __init__(self, model=None, model_name: str = None, stage: str = "None"):
        """
        Args:
            model: trained ML model (sklearn-like API) OR
            model_name: MLflow registered model name to load (e.g., "NYC_Taxi_gb")
            stage: MLflow stage ("None", "Production", "Staging")
        """
        if FEATURE_ENGINEER_AVAILABLE:
            self.feature_engineer = FeatureEngineer()
        else:
            self.feature_engineer = None
            print("??  Using basic feature extraction (FeatureEngineer not available)")
        
        if model_name:
            try:
                # Load from MLflow Model Registry by name
                import mlflow.sklearn
                self.model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
                print(f"? Loaded MLflow model: models:/{model_name}/{stage}")
            except ImportError:
                print("??  MLflow not installed, cannot load model by name")
                self.model = None
        elif model is not None:
            self.model = model
            print("? Using provided model")
        else:
            raise ValueError("Must provide either 'model' or 'model_name'")

    def run(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
        is_train: bool = False,
        fit: bool = False,
    ) -> pd.DataFrame:
        """
        Run batch inference on a dataframe

        Args:
            df (pd.DataFrame): input raw data
            save_path (str, optional): directory or filename to save predictions
            is_train (bool): whether the df contains 'trip_duration'
            fit (bool): whether to fit the encoders/scalers (only True for training)

        Returns:
            pd.DataFrame: dataframe with predictions and timestamp
        """
        # Feature engineering
        if self.feature_engineer:
            X, _, _ = self.feature_engineer.transform(df, fit=fit, is_train=is_train)
        else:
            # Basic feature extraction as fallback
            X = self._extract_basic_features(df)

        # Predictions
        if self.model is None:
            raise ValueError("Model not loaded")
        
        preds = self.model.predict(X)
        df = df.copy()
        df['prediction'] = preds
        df['timestamp'] = datetime.now()

        # Save if path provided
        if save_path:
            self._save(df, save_path)

        return df

    def _extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic features when FeatureEngineer is not available."""
        # Make a copy to avoid modifying original
        df_features = df.copy()
        
        # Convert datetime if present
        if 'pickup_datetime' in df_features.columns:
            df_features['pickup_datetime'] = pd.to_datetime(df_features['pickup_datetime'])
            df_features['hour'] = df_features['pickup_datetime'].dt.hour
            df_features['day_of_week'] = df_features['pickup_datetime'].dt.dayofweek
            df_features['month'] = df_features['pickup_datetime'].dt.month
        
        # Calculate distance (Haversine) if coordinates present
        if all(col in df_features.columns for col in ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']):
            df_features['distance_km'] = self._haversine_km(
                df_features['pickup_latitude'], df_features['pickup_longitude'],
                df_features['dropoff_latitude'], df_features['dropoff_longitude']
            )
        
        # Encode categorical variables if present
        categorical_cols = ['vendor_id', 'store_and_fwd_flag']
        for col in categorical_cols:
            if col in df_features.columns:
                df_features[col] = pd.factorize(df_features[col])[0]
        
        # Select only numeric columns for model input
        numeric_cols = df_features.select_dtypes(include=['number']).columns
        return df_features[numeric_cols]

    def _haversine_km(self, lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points in km."""
        import numpy as np
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def _save(self, df: pd.DataFrame, save_path: str):
        """
        Save predictions dataframe to CSV
        """
        if os.path.isdir(save_path):
            date_str = datetime.now().strftime("%Y%m%d")
            save_file = os.path.join(save_path, f"{date_str}_predictions.csv")
        else:
            save_file = save_path

        # Create directory if it doesn't exist
        Path(save_file).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(save_file, index=False)
        print(f"? Predictions saved to {save_file}")

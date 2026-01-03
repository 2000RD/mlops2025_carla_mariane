import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

def load_data(filepath):
    """Load CSV data."""
    print(f"Loading data from {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean the NYC taxi dataset - handles all 4 requirements."""
    print("Cleaning data...")
    
    # Make a copy so we don't modify the original
    df_clean = df.copy()
    
    # 1. HANDLE MISSING VALUES (REQUIREMENT 1)
    print("  1. Handling missing values...")
    
    # For numeric columns, fill with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"     - Filled missing values in {col} with median: {median_val}")
    
    # For categorical columns, fill with mode (most common)
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col] = df_clean[col].fillna(mode_val)
            print(f"     - Filled missing values in {col} with mode: {mode_val}")
    
    # 2. REMOVE OR FIX INVALID ROWS (REQUIREMENT 2)
    print("  2. Removing invalid rows...")
    initial_rows = df_clean.shape[0]
    
    # Remove rows where trip duration is negative or zero
    if 'trip_duration' in df_clean.columns:
        invalid_duration = df_clean[df_clean['trip_duration'] <= 0].shape[0]
        df_clean = df_clean[df_clean['trip_duration'] > 0]
        print(f"     - Removed {invalid_duration} rows with invalid trip duration")
    
    # Remove rows with unrealistic coordinates (outside NYC)
    if all(col in df_clean.columns for col in ['pickup_latitude', 'pickup_longitude', 
                                               'dropoff_latitude', 'dropoff_longitude']):
        # NYC approximate bounds
        nyc_lat_range = (40.5, 41.0)
        nyc_lon_range = (-74.3, -73.7)
        
        valid_mask = (
            df_clean['pickup_latitude'].between(*nyc_lat_range) &
            df_clean['pickup_longitude'].between(*nyc_lon_range) &
            df_clean['dropoff_latitude'].between(*nyc_lat_range) &
            df_clean['dropoff_longitude'].between(*nyc_lon_range)
        )
        
        invalid_coords = df_clean[~valid_mask].shape[0]
        df_clean = df_clean[valid_mask]
        print(f"     - Removed {invalid_coords} rows with coordinates outside NYC")
    
    # 3. BASIC CLEANING (REQUIREMENT 3)
    print("  3. Performing basic cleaning...")
    
    # Remove duplicate rows
    duplicates = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    print(f"     - Removed {duplicates} duplicate rows")
    
    # Remove any rows where passenger count is 0 or negative
    if 'passenger_count' in df_clean.columns:
        invalid_passengers = df_clean[df_clean['passenger_count'] <= 0].shape[0]
        df_clean = df_clean[df_clean['passenger_count'] > 0]
        print(f"     - Removed {invalid_passengers} rows with invalid passenger count")
    
    # 4. FINAL REPORT
    final_rows = df_clean.shape[0]
    rows_removed = initial_rows - final_rows
    
    print(f"\nCleaning complete:")
    print(f"  - Initial rows: {initial_rows}")
    print(f"  - Rows removed: {rows_removed}")
    print(f"  - Final rows: {final_rows}")
    print(f"  - Columns: {df_clean.shape[1]}")
    
    return df_clean

def save_data(df, filepath):
    """Save cleaned data (REQUIREMENT 4)."""
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet (smaller, faster than CSV)
    df.to_parquet(filepath, index=False)
    print(f"\nSaved cleaned data to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess NYC Taxi data")
    parser.add_argument("--input", default="data/train.csv", help="Input CSV file")
    parser.add_argument("--output", default="data/cleaned_train.parquet", help="Output file")
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.input)
    
    # Clean data (handles all 4 requirements)
    df_clean = clean_data(df)
    
    # Save cleaned data
    save_data(df_clean, args.output)

if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Batch inference script for NYC Taxi Trip Duration prediction.
Uses the InferencePipeline class for end-to-end prediction.
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from mlproject.inference import InferencePipeline, load_model, validate_inference_input

def main():
    parser = argparse.ArgumentParser(
        description="Batch inference for NYC Taxi Trip Duration prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input CSV file with test data")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model file (.pkl or .joblib)")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output predictions CSV file")
    
    # Optional arguments
    parser.add_argument("--model-name", type=str, default=None,
                       help="MLflow model name (if using MLflow registry)")
    parser.add_argument("--stage", type=str, default="None",
                       choices=["None", "Staging", "Production", "Archived"],
                       help="MLflow model stage")
    parser.add_argument("--validate", action="store_true",
                       help="Validate input data before inference")
    
    args = parser.parse_args()
    
    print("?? Starting batch inference...")
    
    try:
        # Load input data
        print(f"?? Loading data from {args.input}")
        df = pd.read_csv(args.input)
        print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Validate input data if requested
        if args.validate:
            print("?? Validating input data...")
            validate_inference_input(df)
            print("? Input data validation passed")
        
        # Load or create inference pipeline
        if args.model_name:
            # Use MLflow model registry
            print(f"?? Loading MLflow model: {args.model_name} (stage: {args.stage})")
            pipeline = InferencePipeline(model_name=args.model_name, stage=args.stage)
        else:
            # Load model from file
            print(f"?? Loading model from {args.model}")
            model = load_model(args.model)
            pipeline = InferencePipeline(model=model)
        
        # Run inference
        print("? Running inference...")
        results = pipeline.run(
            df=df,
            save_path=args.output,
            is_train=False,  # This is test data
            fit=False        # Don't fit encoders on test data
        )
        
        print(f"?? Generated {len(results)} predictions")
        print(f"?? Results saved to {args.output}")
        
        # Show summary statistics
        print("\n?? Prediction Summary:")
        print(f"   Mean prediction: {results['prediction'].mean():.2f}")
        print(f"   Min prediction: {results['prediction'].min():.2f}")
        print(f"   Max prediction: {results['prediction'].max():.2f}")
        print(f"   Std prediction: {results['prediction'].std():.2f}")
        
    except Exception as e:
        print(f"? Error during inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

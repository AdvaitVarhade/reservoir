#!/usr/bin/env python3
"""
Complete ML Pipeline for River Rainfall Forecasting

Workflow:
1. Fetch historical weather data (Open-Meteo, 2020-present)
2. Feature engineering (lag, rolling averages, trends)
3. Train XGBoost model
4. Save model for real-time predictions

Usage:
    python pipeline.py --fetch      # Fetch historical data
    python pipeline.py --engineer   # Feature engineering
    python pipeline.py --train      # Train model
    python pipeline.py --all        # Run complete pipeline
"""

import sys
import subprocess
from pathlib import Path
import pandas as pd
import os

from config import HISTORICAL_DATA, TRAINING_DATA, MODEL_FILE


def run_command(script_name, description):
    """Execute a Python script and return success status"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )
        print(f"✓ {description} completed\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}\n")
        return False


def check_file_exists(filepath, description):
    """Check if a required file exists"""
    p = Path(filepath)
    if p.exists():
        file_size = p.stat().st_size / 1024 / 1024
        print(f"✓ {description}: {filepath} ({file_size:.1f} MB)")
        return True
    else:
        print(f"✗ {description}: {filepath} NOT FOUND")
        return False


def main():
    """Run the complete pipeline"""
    import argparse

    parser = argparse.ArgumentParser(
        description="River Rainfall Forecasting Pipeline"
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch historical weather data"
    )
    parser.add_argument(
        "--engineer",
        action="store_true",
        help="Run feature engineering"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train XGBoost model"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run complete pipeline"
    )

    args = parser.parse_args()

    # If no arguments, run all
    if not any([args.fetch, args.engineer, args.train, args.all]):
        args.all = True

    print("\n" + "="*70)
    print("RIVER RAINFALL FORECASTING PIPELINE")
    print("="*70)
    print("\nProject: Multi-Zone Historical Weather Analysis")
    print("Rivers: Krishna (5 zones), Narmada (4 zones), Kaveri (5 zones)")
    print("Model: XGBoost Rainfall Predictor")
    print("="*70)

    # --- STEP 1: FETCH HISTORICAL DATA ---
    if args.fetch or args.all:
        if not check_file_exists(HISTORICAL_DATA, "Checking historical data"):
            print(f"\nFetching historical weather data...\n")
            if not run_command(
                "fetch_historical_weather.py",
                "Step 1/3: Fetch Historical Weather Data"
            ):
                print("\n✗ Pipeline failed at data fetching stage")
                return False
        else:
            print("\n(Skipping fetch - historical data already exists)\n")

    # --- STEP 2: FEATURE ENGINEERING ---
    if args.engineer or args.all:
        if not check_file_exists(HISTORICAL_DATA, "Input data check"):
            print("\n✗ Cannot run feature engineering: historical_weather.csv not found")
            print("  Run with --fetch first\n")
            return False

        print("\n" + "="*70)
        print("Step 2/3: Feature Engineering")
        print("="*70)
        print("\nRunning feature engineering...")
        if not run_command(
            "feature_engineering.py",
            "Creating lag features, rolling averages, and target variables"
        ):
            print("\n✗ Pipeline failed at feature engineering stage")
            return False

    # --- STEP 3: TRAIN MODEL ---
    if args.train or args.all:
        if not check_file_exists(TRAINING_DATA, "Training data check"):
            print("\n✗ Cannot train model: training_dataset.csv not found")
            print("  Run with --engineer first\n")
            return False

        print("\n" + "="*70)
        print("Step 3/3: Train XGBoost Model")
        print("="*70)
        print("\nTraining model...")
        if not run_command(
            "train_model.py",
            "Training XGBoost rainfall predictor"
        ):
            print("\n✗ Pipeline failed at model training stage")
            return False

    # --- COMPLETION SUMMARY ---
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)

    print("\nGenerated Files:")
    if check_file_exists(HISTORICAL_DATA, "  Historical data"):
        print("    -> Use for retraining / data analysis")

    if check_file_exists(TRAINING_DATA, "  Training dataset"):
        print("    -> Engineered features ready for ML")

    if check_file_exists(MODEL_FILE, "  Trained model"):
        print("    -> Ready for real-time predictions")
        print("\n  Load with: joblib.load('rainfall_model.pkl')")
        print("  Use with: model.predict(features)")

    print("\nNext Steps:")
    print("  1. Use river_weather_predictions.py for real-time predictions")
    print("  2. Monitor model performance over time")
    print("  3. Retrain quarterly with new historical data")
    print("\n" + "="*70 + "\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

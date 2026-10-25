#!/usr/bin/env python3
"""Pipeline entry point for sediment hotspot modules.

Rainfall forecasting is intentionally not retrained here. Use the completed
rainfall prediction output as an input via --rainfall.
"""

import argparse
import os

from sediment_hotspot.preprocessing.flow import build_flow_features
from sediment_hotspot.preprocessing.landuse import landuse_erosion_probability
from sediment_hotspot.preprocessing.rainfall_adapter import rainfall_prediction_features
from sediment_hotspot.preprocessing.satellite import preprocess_multispectral_image
from sediment_hotspot.preprocessing.terrain import compute_terrain_features
from sediment_hotspot.training.train_fusion import train
from sediment_hotspot.inference.predict_hotspots import predict
from sediment_hotspot.reservoirs.pipeline import (
    run_all_multi_reservoir_pipelines,
    run_single_reservoir_pipeline,
)


def main():
    parser = argparse.ArgumentParser(description="Sediment hotspot prediction pipeline")
    parser.add_argument("--satellite-image")
    parser.add_argument("--dem")
    parser.add_argument("--landcover")
    parser.add_argument("--flow-csv")
    parser.add_argument("--rainfall", default="rainfall_predictions.csv")
    parser.add_argument("--train-manifest")
    parser.add_argument("--model", default="data/models/sediment_fusion_model.pt")
    parser.add_argument("--sample", default="data/processed/sample_0.npz")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--multi-reservoir", action="store_true", help="Run zone-wise sedimentation analysis across Krishna, Narmada, and Kaveri basins.")
    parser.add_argument("--reservoir", default="all", help="Target reservoir ID or 'all'.")
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    if args.preprocess:
        if args.satellite_image:
            print(preprocess_multispectral_image(args.satellite_image))
        if args.dem:
            print(compute_terrain_features(args.dem))
        if args.landcover:
            print(landuse_erosion_probability(args.landcover))
        if args.flow_csv:
            print(build_flow_features(args.flow_csv))
        print(rainfall_prediction_features(args.rainfall))

    if args.train:
        if args.train_manifest and os.path.exists(args.train_manifest):
            print(train(args.train_manifest, args.epochs, batch_size=4, output_dir="data/models"))
        else:
            print("[INFO] Skipping retraining: --train-manifest was not provided or found. Using pre-trained weights at data/models/sediment_fusion_model.pt")

    if args.predict:
        if not args.model or not args.sample:
            raise ValueError("--model and --sample are required with --predict")
        print(predict(args.model, args.sample, mc_samples=args.mc_samples))

    if args.multi_reservoir:
        if args.reservoir == "all":
            print("[INFO] Running Multi-Reservoir Zone-Wise Pipeline across all basins...")
            print(run_all_multi_reservoir_pipelines(model_path=args.model, mc_samples=args.mc_samples))
        else:
            print(f"[INFO] Running Multi-Reservoir Pipeline for {args.reservoir}...")
            res = run_single_reservoir_pipeline(args.reservoir, model_path=args.model, mc_samples=args.mc_samples)
            print(res["zone_risk_df"])


if __name__ == "__main__":
    main()
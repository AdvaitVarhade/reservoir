#!/usr/bin/env python3
"""CLI runner for Multi-Reservoir Zone-Wise Sedimentation Analysis.

Executes zone partitioning, forward prediction, epistemic uncertainty,
WR-GNN connectivity, and CSDN counterfactual analysis across reservoirs in
the Krishna, Narmada, and Kaveri river basins.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sediment_hotspot.reservoirs.catalog import (
    RESERVOIR_CATALOG,
    list_primary_reservoirs,
)
from sediment_hotspot.reservoirs.pipeline import (
    run_all_multi_reservoir_pipelines,
    run_single_reservoir_pipeline,
)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Reservoir Zone-Wise Sedimentation Prediction & CSDN Analysis"
    )
    parser.add_argument(
        "--reservoir",
        choices=list(RESERVOIR_CATALOG.keys()) + ["all"],
        default="all",
        help="Target reservoir ID or 'all' to run across Krishna, Narmada, and Kaveri basins.",
    )
    parser.add_argument(
        "--model",
        default="data/models/sediment_fusion_model.pt",
        help="Path to trained SedimentHotspotFusionNet weights.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/outputs/multi_reservoir",
        help="Base directory for output artifacts.",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=20,
        help="Monte Carlo Dropout samples for epistemic uncertainty.",
    )
    args = parser.parse_args()

    print("=======================================================================")
    print("Multi-Reservoir Zone-Wise Sedimentation Prediction & CSDN System")
    print(f"Basins: Krishna, Narmada, Kaveri | Target: {args.reservoir}")
    print("=======================================================================")

    if args.reservoir == "all":
        comp_df = run_all_multi_reservoir_pipelines(
            model_path=args.model,
            output_base_dir=args.output_dir,
            mc_samples=args.mc_samples,
        )
        print("\nCross-Reservoir Basin Comparison Table:")
        print(comp_df[["Reservoir Name", "River Basin", "Mean Hotspot Risk", "Peak Hotspot Risk", "Top Vulnerable Zone (+20% Rain)"]].to_string(index=False))
    else:
        res = run_single_reservoir_pipeline(
            reservoir_id=args.reservoir,
            model_path=args.model,
            output_base_dir=args.output_dir,
            mc_samples=args.mc_samples,
        )
        spec = res["spec"]
        print(f"\nCompleted pipeline for {spec.name} ({spec.basin} Basin)")
        print(res["zone_risk_df"].to_string(index=False))

    print("\n[SUCCESS] All multi-reservoir zone-wise artifacts generated.")


if __name__ == "__main__":
    main()
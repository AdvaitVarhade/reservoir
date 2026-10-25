"""Reservoir-Specific CSDN Counterfactual Scenarios and Sensitivity Analysis.

Evaluates controlled hydrological and meteorological interventions (rainfall, flow,
inflow) on reservoir spatial grids and reports zone-level impacts and sensitivity.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from sediment_hotspot.counterfactual.interventions import apply_intervention
from sediment_hotspot.counterfactual.sensitivity import compute_spatial_sensitivity
from sediment_hotspot.models.fusion import SedimentHotspotFusionNet
from sediment_hotspot.models.uncertainty import mc_dropout_predict
from sediment_hotspot.reservoirs.catalog import ReservoirSpec
from sediment_hotspot.reservoirs.zones import ReservoirGrid
from sediment_hotspot.utils.io import ensure_dir

STANDARD_RESERVOIR_SCENARIOS = [
    {"name": "baseline", "type": "rainfall", "magnitude": 0.0, "label": "Baseline State (Reference)"},
    {"name": "rainfall_plus10", "type": "rainfall", "magnitude": 0.10, "label": "Rainfall +10%"},
    {"name": "rainfall_plus20", "type": "rainfall", "magnitude": 0.20, "label": "Rainfall +20%"},
    {"name": "rainfall_plus30", "type": "rainfall", "magnitude": 0.30, "label": "Rainfall +30%"},
    {"name": "flow_minus10", "type": "flow", "magnitude": -0.10, "label": "River Flow -10%"},
    {"name": "flow_minus20", "type": "flow", "magnitude": -0.20, "label": "River Flow -20%"},
    {"name": "flow_plus10", "type": "flow", "magnitude": 0.10, "label": "River Flow +10%"},
    {"name": "inflow_minus20", "type": "inflow", "magnitude": -0.20, "label": "Reservoir Inflow -20%"},
    {"name": "inflow_plus20", "type": "inflow", "magnitude": 0.20, "label": "Reservoir Inflow +20%"},
]


def run_reservoir_counterfactuals(
    model: SedimentHotspotFusionNet,
    batch: Dict[str, torch.Tensor],
    grid: ReservoirGrid,
    output_dir: Path | str,
    mc_samples: int = 15,
) -> Dict[str, Any]:
    """Executes CSDN counterfactual analysis across all standard scenarios for a specific reservoir."""
    out_dir = ensure_dir(output_dir)
    diff_dir = ensure_dir(out_dir / "counterfactual_difference_maps")
    sens_dir = ensure_dir(out_dir / "sensitivity_maps")

    spec = grid.spec
    model.eval()

    # 1. Compute baseline prediction
    with torch.no_grad():
        base_unc = mc_dropout_predict(model, batch, samples=mc_samples)
        base_heat = base_unc["mean"].squeeze().cpu().numpy()
        base_var = base_unc["variance"].squeeze().cpu().numpy()

    np.save(out_dir / "counterfactual_baseline.npy", base_heat)

    # Calculate baseline zone statistics using real 2D zone masks
    num_zones = len(spec.zones)
    base_zone_means = []
    for z_idx in range(num_zones):
        mask = (grid.zone_id_map == z_idx) & grid.water_mask
        if mask.sum() == 0:
            mask = (grid.zone_id_map == z_idx)
        base_zone_means.append(float(base_heat[mask].mean()))

    scenario_summary_records = []
    zone_scenario_tables: Dict[str, pd.DataFrame] = {}

    for sc in STANDARD_RESERVOIR_SCENARIOS:
        name = sc["name"]
        mag = sc["magnitude"]
        itype = sc["type"]

        if mag == 0.0:
            cf_heat = base_heat.copy()
            cf_var = base_var.copy()
        else:
            cf_batch, _ = apply_intervention(batch, itype, mag)
            with torch.no_grad():
                cf_unc = mc_dropout_predict(model, cf_batch, samples=mc_samples)
                cf_heat = cf_unc["mean"].squeeze().cpu().numpy()
                cf_var = cf_unc["variance"].squeeze().cpu().numpy()

        diff_heat = (cf_heat - base_heat).astype("float32")
        sens_heat = compute_spatial_sensitivity(base_heat, cf_heat, mag)

        np.save(out_dir / f"counterfactual_{name}.npy", cf_heat)
        np.save(diff_dir / f"diff_{name}.npy", diff_heat)
        np.save(sens_dir / f"sens_{name}.npy", sens_heat)

        # Zone-level analysis using actual reservoir 2D spatial masks
        zone_rows = []
        for z_idx, z_spec in enumerate(spec.zones):
            z_mask = (grid.zone_id_map == z_idx) & grid.water_mask
            if z_mask.sum() == 0:
                z_mask = (grid.zone_id_map == z_idx)

            b_risk = float(base_heat[z_mask].mean())
            c_risk = float(cf_heat[z_mask].mean())
            d_risk = float(diff_heat[z_mask].mean())
            s_val = float(sens_heat[z_mask].mean())
            peak_r = float(cf_heat[z_mask].max())
            high_area = float((cf_heat[z_mask] >= 0.5).mean() * 100.0)

            zone_rows.append({
                "reservoir_id": spec.id,
                "zone_id": z_spec.zone_id,
                "zone_name": z_spec.name,
                "zone_type": z_spec.zone_type,
                "baseline_risk": round(b_risk, 4),
                "counterfactual_risk": round(c_risk, 4),
                "risk_change_delta": round(d_risk, 4),
                "mean_sensitivity": round(s_val, 4),
                "peak_counterfactual_risk": round(peak_r, 4),
                "high_risk_area_pct": round(high_area, 2),
            })

        z_df = pd.DataFrame(zone_rows)
        # Classify vulnerability
        q75 = z_df["mean_sensitivity"].quantile(0.75)
        q25 = z_df["mean_sensitivity"].quantile(0.25)
        z_df["vulnerability_category"] = z_df["mean_sensitivity"].apply(
            lambda s: "Highly Sensitive" if s >= q75 else ("Moderately Sensitive" if s >= q25 else "Stable")
        )
        z_df = z_df.sort_values("mean_sensitivity", ascending=False).reset_index(drop=True)
        z_df["sensitivity_rank"] = z_df.index + 1

        zone_scenario_tables[name] = z_df

        # Save primary scenario table
        if name == "rainfall_plus20":
            z_df.to_csv(out_dir / "zone_sensitivity.csv", index=False)

        top_zone = z_df.iloc[0]["zone_name"] if not z_df.empty else "N/A"
        scenario_summary_records.append({
            "scenario": name,
            "label": sc["label"],
            "intervention_type": itype,
            "magnitude": mag,
            "mean_risk": round(float(cf_heat.mean()), 4),
            "risk_change": round(float(diff_heat.mean()), 4),
            "max_risk_increase": round(float(diff_heat.max()), 4),
            "mean_sensitivity": round(float(sens_heat.mean()), 4),
            "top_vulnerable_zone": top_zone,
            "high_risk_area_pct": round(float((cf_heat >= 0.5).mean() * 100.0), 2),
            "mean_uncertainty": round(float(cf_var.mean()), 6),
        })

    summary_df = pd.DataFrame(scenario_summary_records)
    summary_df.to_csv(out_dir / "counterfactual_results.csv", index=False)

    return {
        "summary_df": summary_df,
        "zone_tables": zone_scenario_tables,
        "baseline_heat": base_heat,
        "baseline_var": base_var,
    }

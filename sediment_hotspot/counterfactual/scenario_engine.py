import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sediment_hotspot.counterfactual.interventions import apply_intervention
from sediment_hotspot.counterfactual.sensitivity import compute_spatial_sensitivity, zone_sensitivity_analysis
from sediment_hotspot.models.fusion import SedimentHotspotFusionNet
from sediment_hotspot.models.uncertainty import mc_dropout_predict
from sediment_hotspot.utils.io import ensure_dir, require_file

STANDARD_SCENARIOS = [
    {"name": "baseline", "type": "rainfall", "magnitude": 0.0, "label": "Baseline Conditions"},
    {"name": "rainfall_plus10", "type": "rainfall", "magnitude": 0.10, "label": "Rainfall +10%"},
    {"name": "rainfall_plus20", "type": "rainfall", "magnitude": 0.20, "label": "Rainfall +20%"},
    {"name": "rainfall_plus30", "type": "rainfall", "magnitude": 0.30, "label": "Rainfall +30%"},
    {"name": "flow_minus10", "type": "flow", "magnitude": -0.10, "label": "River Flow -10%"},
    {"name": "flow_minus20", "type": "flow", "magnitude": -0.20, "label": "River Flow -20%"},
    {"name": "flow_plus10", "type": "flow", "magnitude": 0.10, "label": "River Flow +10%"},
    {"name": "inflow_minus20", "type": "inflow", "magnitude": -0.20, "label": "Reservoir Inflow -20%"},
    {"name": "inflow_plus20", "type": "inflow", "magnitude": 0.20, "label": "Reservoir Inflow +20%"},
]


def run_all_scenarios(
    model_or_path: str | SedimentHotspotFusionNet,
    sample_or_path: str | dict,
    output_dir: str = "data/outputs",
    mc_samples: int = 20,
) -> pd.DataFrame:
    """Executes the standard suite of CSDN scenarios with a shared baseline and stores numerical outputs."""
    out_dir = ensure_dir(output_dir)
    diff_dir = ensure_dir(out_dir / "counterfactual_difference_maps")
    sens_dir = ensure_dir(out_dir / "sensitivity_maps")

    if isinstance(model_or_path, str):
        ckpt = torch.load(require_file(model_or_path, "model checkpoint"), map_location="cpu")
        model = SedimentHotspotFusionNet()
        model.load_state_dict(ckpt["model_state"])
    else:
        model = model_or_path
    model.eval()

    if isinstance(sample_or_path, str):
        raw = np.load(require_file(sample_or_path, "sample npz"))
        batch = {
            "satellite": torch.from_numpy(raw["satellite"].astype("float32")).unsqueeze(0),
            "terrain": torch.from_numpy(raw["terrain"].astype("float32")).unsqueeze(0),
            "landuse": torch.from_numpy(raw["landuse"].astype("float32")).unsqueeze(0),
            "flow_sequence": torch.from_numpy(raw["flow_sequence"].astype("float32")).unsqueeze(0),
            "rain_features": torch.from_numpy(raw["rain_features"].astype("float32")).unsqueeze(0),
            "dsdi_features": torch.from_numpy(raw["dsdi_features"].astype("float32")).unsqueeze(0),
            "graph_nodes": torch.from_numpy(raw["graph_nodes"].astype("float32")).unsqueeze(0),
        }
        if "dsdi_feature_map" in raw:
            batch["dsdi_feature_map"] = torch.from_numpy(raw["dsdi_feature_map"].astype("float32")).unsqueeze(0)
    else:
        batch = sample_or_path

    # Set seed for reproducible MC dropout estimation
    torch.manual_seed(42)

    # 1. Compute shared baseline once
    base_unc = mc_dropout_predict(model, batch, samples=mc_samples)
    base_heat = base_unc["mean"].squeeze().cpu().numpy()
    base_var = base_unc["variance"].squeeze().cpu().numpy()

    with torch.no_grad():
        base_out = model(batch)
        base_zones = base_out["zone_risk"].squeeze().cpu().numpy()

    np.save(out_dir / "counterfactual_baseline.npy", base_heat)

    records = []
    for sc in STANDARD_SCENARIOS:
        name = sc["name"]
        mag = sc["magnitude"]
        itype = sc["type"]

        if mag == 0.0:
            cf_heat = base_heat.copy()
            cf_var = base_var.copy()
            cf_zones = base_zones.copy()
        else:
            cf_batch, _ = apply_intervention(batch, itype, mag)
            cf_unc = mc_dropout_predict(model, cf_batch, samples=mc_samples)
            cf_heat = cf_unc["mean"].squeeze().cpu().numpy()
            cf_var = cf_unc["variance"].squeeze().cpu().numpy()
            with torch.no_grad():
                cf_out = model(cf_batch)
                cf_zones = cf_out["zone_risk"].squeeze().cpu().numpy()

        diff_heat = (cf_heat - base_heat).astype("float32")
        sens_heat = compute_spatial_sensitivity(base_heat, cf_heat, mag)
        zone_df = zone_sensitivity_analysis(base_zones, cf_zones, sens_heat, mag)

        np.save(out_dir / f"counterfactual_{name}.npy", cf_heat)
        np.save(diff_dir / f"diff_{name}.npy", diff_heat)
        np.save(sens_dir / f"sens_{name}.npy", sens_heat)

        if name == "rainfall_plus20":
            zone_df.to_csv(out_dir / "zone_sensitivity.csv", index=False)

        records.append({
            "scenario": sc["name"],
            "label": sc["label"],
            "intervention_type": sc["type"],
            "magnitude": sc["magnitude"],
            "mean_risk": float(cf_heat.mean()),
            "risk_change": float(diff_heat.mean()),
            "max_risk_increase": float(diff_heat.max()),
            "mean_sensitivity": float(sens_heat.mean()),
            "top_vulnerable_zone": str(zone_df.iloc[0]["zone"]),
            "high_risk_area_pct": float((cf_heat >= 0.5).mean() * 100),
            "mean_uncertainty": float(cf_var.mean()),
        })

    df = pd.DataFrame(records)
    results_p = out_dir / "counterfactual_results.csv"
    df.to_csv(results_p, index=False)
    print(f"Executed {len(STANDARD_SCENARIOS)} counterfactual scenarios. Saved results to {results_p}")
    return df

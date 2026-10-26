import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sediment_hotspot.models.fusion import SedimentHotspotFusionNet
from sediment_hotspot.models.uncertainty import mc_dropout_predict, classify_uncertainty_regions
from sediment_hotspot.xai.explain import compute_satellite_gradcam, compute_feature_attributions
from sediment_hotspot.utils.io import ensure_dir, require_file


def predict(model_path: str, sample_npz: str, output_dir: str = "data/outputs", mc_samples: int = 20) -> dict[str, Path]:
    checkpoint = torch.load(require_file(model_path, "fusion model"), map_location="cpu")
    sample = np.load(require_file(sample_npz, "inference sample"))

    model = SedimentHotspotFusionNet()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    batch = {
        "satellite": _batched(sample["satellite"]),
        "terrain": _batched(sample["terrain"]),
        "landuse": _batched(sample["landuse"]),
        "flow_sequence": _batched(sample["flow_sequence"]),
        "rain_features": _batched(sample["rain_features"]),
    }
    if "dsdi_features" in sample:
        batch["dsdi_features"] = _batched(sample["dsdi_features"])
    if "dsdi_feature_map" in sample:
        batch["dsdi_feature_map"] = _batched(sample["dsdi_feature_map"])
    if "graph_nodes" in sample:
        batch["graph_nodes"] = _batched(sample["graph_nodes"])
    if "graph_adjacency" in sample:
        batch["graph_adjacency"] = _batched(sample["graph_adjacency"])

    with torch.no_grad():
        output = model(batch)

    # Uncertainty prediction via MC-Dropout
    unc_res = mc_dropout_predict(model, batch, samples=mc_samples)
    heatmap_mean = unc_res["mean"].squeeze().numpy()
    heatmap_var = unc_res["variance"].squeeze().numpy()

    # DSDI heatmap
    if output["dsdi_map"] is not None:
        dsdi_map = output["dsdi_map"].squeeze().numpy()
    else:
        dsdi_val = float(output["dsdi"].squeeze().numpy())
        dsdi_map = np.clip(heatmap_mean * 0.6 + dsdi_val * 0.4, 0, 1)

    out_dir = ensure_dir(output_dir)
    heatmap_path = out_dir / "future_sediment_hotspot_heatmap.npy"
    dsdi_path = out_dir / "dsdi_heatmap.npy"
    uncertainty_path = out_dir / "uncertainty_heatmap.npy"
    zone_path = out_dir / "zone_sediment_risk.csv"
    attention_path = out_dir / "modality_attention.csv"
    causal_path = out_dir / "causal_attention.csv"
    timeline_path = out_dir / "risk_timeline.csv"
    meta_path = out_dir / "prediction_metadata.json"

    np.save(heatmap_path, heatmap_mean)
    np.save(dsdi_path, dsdi_map)
    np.save(uncertainty_path, heatmap_var)

    zone_risk = output["zone_risk"].squeeze().numpy()
    pd.DataFrame({"zone": [f"Zone {chr(65 + i)}" for i in range(len(zone_risk))], "risk": zone_risk}).to_csv(zone_path, index=False)

    modalities = ["satellite", "flow", "terrain", "landuse", "rain", "graph", "dsdi"]
    learned_weights = output["learned_attention"].squeeze().numpy()
    causal_weights = output["causal_attention"].squeeze().numpy()
    fusion_weights = output["fusion_weight"].squeeze().numpy()

    pd.DataFrame({
        "modality": modalities,
        "learned_attention": learned_weights,
        "causal_attention": causal_weights,
        "fusion_weight": fusion_weights,
        "attention_weight": fusion_weights,
    }).to_csv(attention_path, index=False)

    pd.DataFrame({
        "modality": modalities,
        "causal_score": causal_weights,
    }).to_csv(causal_path, index=False)

    _risk_timeline(heatmap_mean, zone_risk).to_csv(timeline_path, index=False)

    # Uncertainty regions summary
    unc_cats = classify_uncertainty_regions(heatmap_mean, heatmap_var)
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": "SedimentHotspotFusionNet",
        "provenance": str(sample.get("label_type", "DERIVED_FROM_SATELLITE")),
        "observation_source": str(sample.get("label_source", "Sentinel-2 / Open-Meteo")),
        "high_risk_fraction": float((heatmap_mean >= 0.5).mean()),
        "high_confidence_hotspots": float(unc_cats["high_risk_high_confidence"].mean()),
        "monitoring_priority_fraction": float(unc_cats["high_risk_low_confidence"].mean()),
        "mean_dsdi": float(np.mean(dsdi_map)),
        "mc_samples": mc_samples,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "heatmap": heatmap_path,
        "dsdi": dsdi_path,
        "uncertainty": uncertainty_path,
        "zone_risk": zone_path,
        "attention": attention_path,
        "causal": causal_path,
        "timeline": timeline_path,
        "metadata": meta_path,
    }


def _batched(array: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(array.astype("float32"))
    return t.unsqueeze(0) if t.ndim in (1, 2, 3) else t


def _risk_timeline(heatmap: np.ndarray, zone_risk: np.ndarray, days: int = 14) -> pd.DataFrame:
    base_risk = float(np.mean(zone_risk))
    high_risk_area = float(np.mean(heatmap > 0.5))
    start = datetime.now().replace(minute=0, second=0, microsecond=0)
    rows = []
    for day in range(days):
        seasonal_factor = 1 + 0.18 * np.sin(day / max(days - 1, 1) * np.pi)
        rows.append({
            "timestamp": start + timedelta(days=day),
            "mean_risk": min(base_risk * seasonal_factor, 1.0),
            "high_risk_area": min(high_risk_area * seasonal_factor, 1.0),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output-dir", default="data/outputs")
    parser.add_argument("--mc-samples", type=int, default=20)
    args = parser.parse_args()
    print(predict(args.model, args.sample, args.output_dir, args.mc_samples))


if __name__ == "__main__":
    main()

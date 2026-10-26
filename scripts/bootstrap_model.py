#!/usr/bin/env python3
"""Bootstrap starter model weights and minimal validation datasets.

Ensures that fresh clones can immediately run tests, pipelines, and the
Streamlit dashboard without encountering FileNotFoundError on gitignored artifacts.
"""

import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from sediment_hotspot.config import SEDIMENT
from sediment_hotspot.models.fusion import SedimentHotspotFusionNet
from sediment_hotspot.utils.io import ensure_dir


def ensure_rainfall_starter_artifacts(
    model_file: str = "rainfall_model.pkl",
    predictions_file: str = "rainfall_predictions.csv",
) -> dict[str, Path]:
    """Ensures rainfall XGBoost model and initial predictions table exist."""
    m_p = Path(model_file)
    p_p = Path(predictions_file)

    if not m_p.exists():
        print(f"[BOOTSTRAP] Initializing starter rainfall model at {m_p}...")
        feature_cols = [
            "temperature", "humidity", "rainfall", "pressure", "wind_speed", "cloud_cover",
            "temperature_lag_1", "temperature_lag_2", "temperature_lag_3",
            "humidity_lag_1", "humidity_lag_2", "humidity_lag_3",
            "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3",
            "pressure_lag_1", "pressure_lag_2", "pressure_lag_3",
            "wind_speed_lag_1", "wind_speed_lag_2", "wind_speed_lag_3",
            "cloud_cover_lag_1", "cloud_cover_lag_2", "cloud_cover_lag_3",
            "rainfall_avg_3", "humidity_avg_3", "temperature_avg_3",
            "pressure_change", "humidity_change", "temperature_change", "wind_change",
            "hour", "day", "month",
        ]
        # Train on realistic synthetic meteorological distributions
        rng = np.random.RandomState(42)
        n_rows = 100
        x_data = {
            col: rng.uniform(0.0, 50.0 if "rain" in col else 100.0, size=n_rows)
            for col in feature_cols
        }
        df_x = pd.DataFrame(x_data)
        y = np.clip(0.4 * df_x["rainfall"] + 0.3 * df_x["rainfall_avg_3"] + rng.normal(0, 0.5, n_rows), 0, None)

        xgb = XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
        xgb.fit(df_x, y)
        joblib.dump(xgb, m_p)
        print(f"[BOOTSTRAP] Created {m_p}")

    if not p_p.exists():
        print(f"[BOOTSTRAP] Generating starter rainfall predictions in {p_p}...")
        sample_rows = [
            {"timestamp": "2026-10-26 12:00:00", "river": "Krishna", "zone": "K1_Mahabaleshwar", "latitude": 17.9237, "longitude": 73.6586, "current_temp": 24.2, "current_humidity": 78.0, "current_rainfall": 2.1, "predicted_rainfall_next_hour": 3.4, "rainfall_class": "Moderate", "alert": ""},
            {"timestamp": "2026-10-26 12:00:00", "river": "Krishna", "zone": "K2_Almatti_Dam", "latitude": 16.3314, "longitude": 75.8869, "current_temp": 29.1, "current_humidity": 62.0, "current_rainfall": 0.5, "predicted_rainfall_next_hour": 0.8, "rainfall_class": "Light", "alert": ""},
            {"timestamp": "2026-10-26 12:00:00", "river": "Krishna", "zone": "K4_Nagarjuna_Sagar", "latitude": 16.5750, "longitude": 79.3167, "current_temp": 30.5, "current_humidity": 60.0, "current_rainfall": 0.0, "predicted_rainfall_next_hour": 0.2, "rainfall_class": "No Rain", "alert": ""},
            {"timestamp": "2026-10-26 12:00:00", "river": "Narmada", "zone": "N3_Omkareshwar", "latitude": 22.2452, "longitude": 76.1510, "current_temp": 28.0, "current_humidity": 55.0, "current_rainfall": 0.4, "predicted_rainfall_next_hour": 0.9, "rainfall_class": "Light", "alert": ""},
            {"timestamp": "2026-10-26 12:00:00", "river": "Kaveri", "zone": "C4_Mettur_Dam", "latitude": 11.7870, "longitude": 77.8008, "current_temp": 27.5, "current_humidity": 68.0, "current_rainfall": 1.2, "predicted_rainfall_next_hour": 1.8, "rainfall_class": "Light", "alert": ""},
        ]
        pd.DataFrame(sample_rows).to_csv(p_p, index=False)
        print(f"[BOOTSTRAP] Created {p_p}")

    return {"model_file": m_p, "predictions_file": p_p}


def ensure_starter_artifacts(
    model_path: str = "data/models/sediment_fusion_model.pt",
    sample_npz: str = "data/processed/sample_0.npz",
    manifest_csv: str = "data/processed/manifest.csv",
) -> dict[str, Path]:
    """Initializes starter model checkpoint and synthetic samples if absent."""
    m_path = Path(model_path)
    s_path = Path(sample_npz)
    mf_path = Path(manifest_csv)

    # 1. Ensure minimal sample dataset exists
    if not s_path.exists() or not mf_path.exists():
        print(f"[BOOTSTRAP] Generating starter samples in {s_path.parent}...")
        from scripts.generate_synthetic_data import generate_dataset
        generate_dataset(num_samples=6, output_dir=str(s_path.parent))

    # 2. Ensure starter model weights exist
    if not m_path.exists():
        print(f"[BOOTSTRAP] Initializing starter model weights at {m_path}...")
        ensure_dir(m_path.parent)
        model = SedimentHotspotFusionNet(config=SEDIMENT)
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": SEDIMENT.__dict__,
                "bootstrap": True,
                "notes": "Starter initialization weights for zero-setup execution and tests.",
            },
            m_path,
        )
        print(f"[BOOTSTRAP] Created {m_path}")

    return {
        "model_path": m_path,
        "sample_npz": s_path,
        "manifest_csv": mf_path,
    }


if __name__ == "__main__":
    res = ensure_starter_artifacts()
    res_rain = ensure_rainfall_starter_artifacts()
    res.update(res_rain)
    print("[SUCCESS] All starter artifacts verified:")
    for k, v in res.items():
        print(f"  {k}: {v} (exists: {v.exists()})")

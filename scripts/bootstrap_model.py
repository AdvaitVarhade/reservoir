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

from sediment_hotspot.config import SEDIMENT
from sediment_hotspot.models.fusion import SedimentHotspotFusionNet
from sediment_hotspot.utils.io import ensure_dir


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
    print("[SUCCESS] All starter artifacts verified:")
    for k, v in res.items():
        print(f"  {k}: {v} (exists: {v.exists()})")

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sediment_hotspot.counterfactual.scenario_engine import run_all_scenarios

if __name__ == "__main__":
    df = run_all_scenarios(
        "data/models/sediment_fusion_model.pt",
        "data/processed/sample_0.npz",
        output_dir="data/outputs",
    )
    print("\nCSDN Scenario Engine Results:")
    print(df.to_string())

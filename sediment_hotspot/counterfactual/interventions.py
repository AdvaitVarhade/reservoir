from typing import Literal
import torch

InterventionType = Literal["rainfall", "flow", "inflow", "rainfall_intensity", "water_velocity"]

INTERVENTION_INDEX_MAP = {
    "rainfall": [0],             # DSDI index 0: rainfall
    "rainfall_intensity": [1],   # DSDI index 1: rainfall_intensity
    "flow": [2, 4],              # DSDI index 2: discharge, 4: velocity
    "inflow": [3],               # DSDI index 3: inflow
    "water_velocity": [4],       # DSDI index 4: velocity
}


def apply_intervention(
    batch: dict[str, torch.Tensor],
    intervention_type: InterventionType,
    magnitude: float,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Applies a physically constrained environmental intervention to dynamic inputs.
    Strictly preserves static environmental variables (satellite imagery, DEM terrain, land-use).
    """
    cf_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            cf_batch[k] = v.clone()
        else:
            cf_batch[k] = v

    factor = 1.0 + float(magnitude)

    if intervention_type == "rainfall":
        # Modify rain features (dim: [B, 4])
        cf_batch["rain_features"] = torch.clamp(cf_batch["rain_features"] * factor, 0.0, 1.0)
        # Modify DSDI features (dim: [B, 11])
        if "dsdi_features" in cf_batch:
            cf_batch["dsdi_features"][:, 0] = torch.clamp(cf_batch["dsdi_features"][:, 0] * factor, 0.0, 1.0)
            cf_batch["dsdi_features"][:, 1] = torch.clamp(cf_batch["dsdi_features"][:, 1] * factor, 0.0, 1.0)
            # Hydrological propagation: increased rainfall increases runoff arrival
            if magnitude > 0:
                cf_batch["dsdi_features"][:, 3] = torch.clamp(cf_batch["dsdi_features"][:, 3] * (1.0 + 0.5 * magnitude), 0.0, 1.0)
        if "dsdi_feature_map" in cf_batch:
            cf_batch["dsdi_feature_map"][:, 0] = torch.clamp(cf_batch["dsdi_feature_map"][:, 0] * factor, 0.0, 1.0)
            cf_batch["dsdi_feature_map"][:, 1] = torch.clamp(cf_batch["dsdi_feature_map"][:, 1] * factor, 0.0, 1.0)
            if magnitude > 0:
                cf_batch["dsdi_feature_map"][:, 3] = torch.clamp(cf_batch["dsdi_feature_map"][:, 3] * (1.0 + 0.5 * magnitude), 0.0, 1.0)

    elif intervention_type == "flow":
        # Modify flow sequence (dim: [B, T, 8])
        cf_batch["flow_sequence"] = torch.clamp(cf_batch["flow_sequence"] * factor, 0.0, 1.0)
        if "dsdi_features" in cf_batch:
            cf_batch["dsdi_features"][:, 2] = torch.clamp(cf_batch["dsdi_features"][:, 2] * factor, 0.0, 1.0)
            cf_batch["dsdi_features"][:, 4] = torch.clamp(cf_batch["dsdi_features"][:, 4] * factor, 0.0, 1.0)
        if "dsdi_feature_map" in cf_batch:
            cf_batch["dsdi_feature_map"][:, 2] = torch.clamp(cf_batch["dsdi_feature_map"][:, 2] * factor, 0.0, 1.0)
            cf_batch["dsdi_feature_map"][:, 4] = torch.clamp(cf_batch["dsdi_feature_map"][:, 4] * factor, 0.0, 1.0)

    elif intervention_type == "inflow":
        # Modify inflow in flow sequence (channel 1) and DSDI (index 3)
        cf_batch["flow_sequence"][:, :, 1] = torch.clamp(cf_batch["flow_sequence"][:, :, 1] * factor, 0.0, 1.0)
        if "dsdi_features" in cf_batch:
            cf_batch["dsdi_features"][:, 3] = torch.clamp(cf_batch["dsdi_features"][:, 3] * factor, 0.0, 1.0)
        if "dsdi_feature_map" in cf_batch:
            cf_batch["dsdi_feature_map"][:, 3] = torch.clamp(cf_batch["dsdi_feature_map"][:, 3] * factor, 0.0, 1.0)

    elif intervention_type == "rainfall_intensity":
        cf_batch["rain_features"][:, 1] = torch.clamp(cf_batch["rain_features"][:, 1] * factor, 0.0, 1.0)
        if "dsdi_features" in cf_batch:
            cf_batch["dsdi_features"][:, 1] = torch.clamp(cf_batch["dsdi_features"][:, 1] * factor, 0.0, 1.0)
        if "dsdi_feature_map" in cf_batch:
            cf_batch["dsdi_feature_map"][:, 1] = torch.clamp(cf_batch["dsdi_feature_map"][:, 1] * factor, 0.0, 1.0)

    elif intervention_type == "water_velocity":
        cf_batch["flow_sequence"][:, :, 2] = torch.clamp(cf_batch["flow_sequence"][:, :, 2] * factor, 0.0, 1.0)
        if "dsdi_features" in cf_batch:
            cf_batch["dsdi_features"][:, 4] = torch.clamp(cf_batch["dsdi_features"][:, 4] * factor, 0.0, 1.0)
        if "dsdi_feature_map" in cf_batch:
            cf_batch["dsdi_feature_map"][:, 4] = torch.clamp(cf_batch["dsdi_feature_map"][:, 4] * factor, 0.0, 1.0)

    metadata = {
        "intervention_type": intervention_type,
        "magnitude": magnitude,
        "percentage_change": f"{magnitude * 100:+.1f}%",
        "description": f"{intervention_type.replace('_', ' ').title()} changed by {magnitude * 100:+.1f}%",
        "preserved_modalities": ["satellite", "terrain", "landuse", "graph_nodes"],
    }
    return cf_batch, metadata


def generate_counterfactual_batch(
    batch: dict[str, torch.Tensor],
    intervention_type: InterventionType = "rainfall",
    magnitude: float = 0.20,
) -> dict[str, torch.Tensor]:
    cf_batch, _ = apply_intervention(batch, intervention_type, magnitude)
    return cf_batch

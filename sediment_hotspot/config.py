from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path = Path(".")
    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    models: Path = Path("data/models")
    outputs: Path = Path("data/outputs")
    results: Path = Path("results")
    paper: Path = Path("results/paper")
    rainfall_predictions: Path = Path("rainfall_predictions.csv")


@dataclass(frozen=True)
class RasterConfig:
    target_crs: str = "EPSG:4326"
    target_resolution_m: int = 30
    cloud_probability_threshold: float = 40.0
    ndwi_threshold: float = 0.15
    fixed_turbidity_quantile: float = 0.85
    adaptive_threshold_floor: float = 0.45
    adaptive_threshold_ceiling: float = 0.97


@dataclass(frozen=True)
class ModelConfig:
    image_channels: int = 6
    terrain_channels: int = 4
    landuse_channels: int = 5
    flow_features: int = 8
    rain_features: int = 4
    hidden_dim: int = 128
    zones: int = 8
    graph_features: int = 8
    dsdi_features: int = 11
    projection_dim: int = 64
    dropout: float = 0.20
    image_size: tuple[int, int] = (128, 128)
    modality_names: tuple[str, ...] = (
        "satellite",
        "flow",
        "terrain",
        "landuse",
        "rain",
        "graph",
        "dsdi",
    )
    dsdi_feature_names: tuple[str, ...] = (
        "rainfall",
        "rainfall_intensity",
        "river_discharge",
        "reservoir_inflow",
        "water_velocity",
        "terrain_slope",
        "terrain_curvature",
        "erosion_susceptibility",
        "landuse_erosion_probability",
        "turbidity_ndti",
        "sediment_plume_intensity",
    )


@dataclass(frozen=True)
class SedimentConfig:
    use_dsdi: bool = True
    use_causal_attention: bool = True
    use_gnn: bool = True
    use_contrastive: bool = True
    use_uncertainty: bool = True
    use_adaptive_threshold: bool = True
    use_temporal_constraint: bool = True
    use_physics_loss: bool = True
    use_counterfactual_loss: bool = True
    hotspot_loss: str = "bce"
    zone_loss: str = "mse"
    lambda_zone: float = 1.0
    lambda_physics: float = 0.25
    lambda_temporal: float = 0.10
    lambda_consistency: float = 0.15
    lambda_contrastive: float = 0.05
    lambda_counterfactual: float = 0.10
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    consistency_margin: float = 0.05
    temporal_change_tolerance: float = 0.15
    counterfactual_tolerance: float = 0.08
    hotspot_probability_threshold: float = 0.50
    uncertainty_quantile: float = 0.75
    mc_dropout_samples: int = 20
    graph_layers: int = 2
    graph_use_fallback_chain: bool = True
    causal_temperature: float = 1.0
    explanation_top_k: int = 4


PATHS = Paths()
RASTER = RasterConfig()
MODEL = ModelConfig()
SEDIMENT = SedimentConfig()


def dataclass_dict(config) -> dict:
    return asdict(config)

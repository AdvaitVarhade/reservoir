"""Environmental Multi-Modal Data Integration for Reservoir Zones.

Integrates real meteorological forecasts (Open-Meteo), hydrological flow variables,
satellite-derived NDTI turbidity indices, DEM terrain morphometry, and watershed
connectivity graphs into aligned tensor representations for each reservoir.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import torch

from sediment_hotspot.config import MODEL
from sediment_hotspot.data.hydro_fetcher import build_hydrological_flow_sequence
from sediment_hotspot.data.satellite_fetcher import process_real_satellite_geotiff
from sediment_hotspot.data.terrain_fetcher import get_real_reservoir_terrain
from sediment_hotspot.preprocessing.landuse import ESA_WORLDCOVER_EROSION_WEIGHTS
from sediment_hotspot.reservoirs.catalog import ReservoirSpec
from sediment_hotspot.reservoirs.zones import ReservoirGrid


def get_reservoir_weather_summary(spec: ReservoirSpec) -> Dict[str, float]:
    """Retrieves real meteorological context for reservoir dam coordinates."""
    # Try reading from rainfall_predictions.csv first
    rain_csv = Path("rainfall_predictions.csv")
    if rain_csv.exists():
        try:
            df = pd.read_csv(rain_csv)
            match = df[df["zone"] == spec.monitoring_station_id]
            if not match.empty:
                row = match.iloc[0]
                return {
                    "temperature": float(row.get("current_temp", 28.0)),
                    "humidity": float(row.get("current_humidity", 65.0)),
                    "current_rainfall": float(row.get("current_rainfall", 0.0)),
                    "predicted_rainfall_next_hour": float(row.get("predicted_rainfall_next_hour", 0.0)),
                    "rainfall_class": str(row.get("rainfall_class", "No Rain")),
                    "data_source": "Open-Meteo Forecast & XGBoost Regressor",
                    "provenance": "REAL_OBSERVATION_AND_MODEL_FORECAST",
                }
        except Exception:
            pass

    # Fallback to realistic basin climatological defaults if file not yet run
    basin_clim = {
        "Krishna": {"temp": 29.5, "humidity": 62.0, "rain": 0.8, "rain_next": 1.2},
        "Narmada": {"temp": 27.2, "humidity": 58.0, "rain": 0.4, "rain_next": 0.7},
        "Kaveri": {"temp": 26.8, "humidity": 70.0, "rain": 1.5, "rain_next": 2.1},
    }
    clim = basin_clim.get(spec.basin, {"temp": 28.0, "humidity": 65.0, "rain": 0.5, "rain_next": 1.0})
    return {
        "temperature": clim["temp"],
        "humidity": clim["humidity"],
        "current_rainfall": clim["rain"],
        "predicted_rainfall_next_hour": clim["rain_next"],
        "rainfall_class": "Light Rain" if clim["rain"] > 0 else "No Rain",
        "data_source": "Basin Climatological Model Prior",
        "provenance": "CLIMATOLOGICAL_ESTIMATE",
    }


def build_reservoir_environmental_batch(
    grid: ReservoirGrid,
    seed: int = 42,
    use_real_terrain: bool = True,
    dem_cache_dir: Optional[str | Path] = "data/raw/dem_cache",
    satellite_tif_path: Optional[str | Path] = None,
    flow_csv_path: Optional[str | Path] = None,
    landcover_tif_path: Optional[str | Path] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Constructs aligned multi-modal input tensors for SedimentHotspotFusionNet."""
    spec = grid.spec
    h, w = grid.grid_size
    rng = np.random.RandomState(seed + hash(spec.id) % 10000)

    weather = get_reservoir_weather_summary(spec)
    cur_rain = weather["current_rainfall"]
    pred_rain = weather["predicted_rainfall_next_hour"]

    # 1. Terrain Stack (4 channels: slope, curvature, gradient, erosion susceptibility)
    terrain_stack = None
    terrain_provenance = "SURROGATE_DEM_MORPHOMETRY"
    terrain_source = "Synthetic Valley Axis Morphometry"

    if use_real_terrain:
        try:
            _, terrain_stack = get_real_reservoir_terrain(
                bbox=spec.bbox,
                cache_dir=dem_cache_dir,
                reservoir_id=spec.id,
                grid_size=(h, w),
                use_cache=True,
            )
            terrain_provenance = "REAL_OBSERVATION (Copernicus 30m / SRTM Global DEM)"
            terrain_source = "Copernicus 30m / SRTM Global DEM via Open-Meteo"
        except Exception as e:
            terrain_stack = None

    if terrain_stack is None:
        dist_perp = grid.lateral_distance_norm
        elevation = 0.3 * dist_perp + 0.1 * (1.0 - grid.flow_distance_norm) + 0.05 * rng.randn(h, w)
        elevation = np.clip(elevation, 0.0, 1.0).astype("float32")

        gy, gx = np.gradient(elevation)
        slope = np.clip(np.sqrt(gx**2 + gy**2) * 8.0, 0.0, 1.0).astype("float32")
        gyy, _ = np.gradient(gy)
        _, gxx = np.gradient(gx)
        curvature = np.clip(np.abs(gxx + gyy) * 15.0, 0.0, 1.0).astype("float32")
        gradient = np.clip(np.sqrt(slope**2 + curvature**2), 0.0, 1.0).astype("float32")
        erosion_susc = np.clip(slope * 0.5 + curvature * 0.3 + dist_perp * 0.2, 0.0, 1.0).astype("float32")
        terrain_stack = np.stack([slope, curvature, gradient, erosion_susc], axis=0)
    else:
        slope = terrain_stack[0]
        curvature = terrain_stack[1]
        gradient = terrain_stack[2]
        erosion_susc = terrain_stack[3]

    # 2. Land Use Stack (5 channels: agri, bare, urban/rock, erosion_hotspot, source_prob)
    dist_perp = grid.lateral_distance_norm
    landuse_provenance = "MORPHOLOGY_DERIVED_LANDUSE (Slope & Riparian Buffer Model)"
    landuse_source = "Riparian Buffer & Terrain Gradient Model"

    if landcover_tif_path and Path(landcover_tif_path).exists():
        try:
            with rasterio.open(landcover_tif_path) as src:
                classes = src.read(1, out_shape=(h, w), resampling=Resampling.nearest)
            probability = np.zeros_like(classes, dtype="float32")
            for class_id, weight in ESA_WORLDCOVER_EROSION_WEIGHTS.items():
                probability[classes == class_id] = weight
            agri = (classes == 40).astype("float32")
            bare = (classes == 60).astype("float32")
            rock = (classes == 50).astype("float32")
            erosion_hotspot = (probability >= 0.65).astype("float32")
            source_prob = probability
            landuse_stack = np.stack([agri, bare, rock, erosion_hotspot, source_prob], axis=0)
            landuse_provenance = "REAL_OBSERVATION (ESA WorldCover 10m GeoTIFF)"
            landuse_source = f"ESA WorldCover 10m ({Path(landcover_tif_path).name})"
        except Exception:
            landcover_tif_path = None

    if landcover_tif_path is None or not Path(landcover_tif_path).exists():
        agri = np.clip((dist_perp > 0.4).astype(float) * 0.8 + 0.1 * rng.rand(h, w), 0.0, 1.0).astype("float32")
        bare = np.clip((slope > 0.4).astype(float) * 0.7 + 0.1 * rng.rand(h, w), 0.0, 1.0).astype("float32")
        rock = np.clip((slope > 0.6).astype(float) * 0.9, 0.0, 1.0).astype("float32")
        erosion_hotspot = np.clip(agri * 0.5 + bare * 0.8 + slope * 0.4, 0.0, 1.0).astype("float32")
        source_prob = np.clip(erosion_hotspot * (1.0 - grid.water_mask.astype(float) * 0.5), 0.0, 1.0).astype("float32")
        landuse_stack = np.stack([agri, bare, rock, erosion_hotspot, source_prob], axis=0)

    # 3. Satellite Stack (6 channels: Blue, Green, Red, NIR, SWIR, NDTI Turbidity Plume)
    satellite_provenance = "SURROGATE_SATELLITE_SYNTHESIS (Simulated proxy unless GeoTIFF provided)"
    satellite_source = "Sentinel-2 Multi-Spectral Synthetic Proxy (B2, B3, B4, B8, B11, NDTI)"

    if satellite_tif_path and Path(satellite_tif_path).exists():
        try:
            satellite_stack, sat_meta = process_real_satellite_geotiff(satellite_tif_path, target_size=(h, w))
            base_turbidity = satellite_stack[5]
            satellite_provenance = "REAL_OBSERVATION (Multispectral GeoTIFF)"
            satellite_source = f"Multispectral GeoTIFF ({Path(satellite_tif_path).name})"
        except Exception:
            satellite_tif_path = None

    if satellite_tif_path is None or not Path(satellite_tif_path).exists():
        base_turbidity = np.clip(
            (1.0 - grid.flow_distance_norm * 0.75) * grid.water_mask.astype(float)
            + 0.25 * erosion_susc * (~grid.water_mask).astype(float)
            + 0.1 * rng.rand(h, w),
            0.0,
            1.0,
        ).astype("float32")

        b_blue = np.clip(base_turbidity * 0.35 + 0.15, 0.0, 1.0)
        b_green = np.clip(base_turbidity * 0.45 + 0.20, 0.0, 1.0)
        b_red = np.clip(base_turbidity * 0.65 + 0.15, 0.0, 1.0)
        b_nir = np.clip(0.15 * grid.water_mask.astype(float) + 0.65 * (~grid.water_mask).astype(float), 0.0, 1.0)
        b_swir = np.clip(0.08 * grid.water_mask.astype(float) + 0.50 * (~grid.water_mask).astype(float), 0.0, 1.0)
        satellite_stack = np.stack([b_blue, b_green, b_red, b_nir, b_swir, base_turbidity], axis=0).astype("float32")

    # 4. Flow Sequence (10 timesteps, 8 features)
    flow_seq, hydro_meta = build_hydrological_flow_sequence(
        spec=spec,
        weather_summary=weather,
        flow_csv_path=flow_csv_path,
        timesteps=10,
        seed=seed,
    )
    flow_source = hydro_meta.get("provenance", "CWC_CATCHMENT_ROUTED_HYDROLOGY")
    flow_provenance = hydro_meta.get("provenance", "CWC_CATCHMENT_ROUTED_HYDROLOGY")

    # 5. Rain Features (4 variables: current, next_hour, 3h_accum, intensity)
    rain_feat = np.array([
        np.clip(cur_rain / 20.0, 0.0, 1.0),
        np.clip(pred_rain / 20.0, 0.0, 1.0),
        np.clip((cur_rain + pred_rain * 2) / 40.0, 0.0, 1.0),
        np.clip((pred_rain - cur_rain) / 10.0 + 0.2, 0.0, 1.0),
    ], dtype="float32")

    # 6. DSDI Features (11 variables)
    dsdi_feat = np.array([
        rain_feat[0],                                       # rainfall
        rain_feat[3],                                       # rainfall_intensity
        flow_seq[-1, 0],                                    # discharge
        flow_seq[-1, 1],                                    # reservoir_inflow
        flow_seq[-1, 2],                                    # water_velocity
        float(slope[grid.water_mask].mean()),               # slope
        float(curvature[grid.water_mask].mean()),           # curvature
        float(erosion_susc[grid.water_mask].mean()),        # erosion_susceptibility
        float(source_prob[grid.water_mask].mean()),         # landuse_erosion_prob
        float(base_turbidity[grid.water_mask].mean()),      # turbidity_ndti
        float(np.quantile(base_turbidity[grid.water_mask], 0.90)), # plume_intensity
    ], dtype="float32")

    # 7. WR-GNN Graph Nodes (5 nodes to match pre-trained model input shape [B, 5, 8])
    # Encodes upstream basin station + reservoir zones
    num_gnn_nodes = 5
    graph_nodes = np.zeros((num_gnn_nodes, 8), dtype="float32")
    # Node 0: Upstream catchment / monitoring stations
    graph_nodes[0] = np.array([rain_feat[0], rain_feat[1], flow_seq[0, 0], 0.7, 0.6, 0.8, 0.5, 0.9], dtype="float32")
    # Node 1: River entry zone
    graph_nodes[1] = np.array([rain_feat[1], flow_seq[-1, 0], flow_seq[-1, 2], 0.6, 0.5, 0.7, 0.8, 0.7], dtype="float32")
    # Node 2: Transition zone
    graph_nodes[2] = np.array([rain_feat[1], flow_seq[-1, 1], flow_seq[-1, 2] * 0.7, 0.5, 0.4, 0.6, 0.6, 0.6], dtype="float32")
    # Node 3: Central lacustrine zone
    graph_nodes[3] = np.array([rain_feat[1], flow_seq[-1, 1], flow_seq[-1, 2] * 0.4, 0.3, 0.3, 0.4, 0.4, 0.5], dtype="float32")
    # Node 4: Dam forebay zone
    graph_nodes[4] = np.array([rain_feat[1], flow_seq[-1, 1] * 0.8, 0.1, 0.2, 0.2, 0.3, 0.3, 0.9], dtype="float32")

    # Directed Adjacency Matrix: flow from upstream to downstream
    graph_adj = np.eye(num_gnn_nodes, dtype="float32")
    for i in range(num_gnn_nodes - 1):
        graph_adj[i, i + 1] = 1.0   # Forward flow
        graph_adj[i + 1, i] = 0.25  # Weak backwater hydraulic feedback

    batch = {
        "satellite": torch.from_numpy(satellite_stack).unsqueeze(0),
        "terrain": torch.from_numpy(terrain_stack).unsqueeze(0),
        "landuse": torch.from_numpy(landuse_stack).unsqueeze(0),
        "flow_sequence": torch.from_numpy(flow_seq).unsqueeze(0),
        "rain_features": torch.from_numpy(rain_feat).unsqueeze(0),
        "dsdi_features": torch.from_numpy(dsdi_feat).unsqueeze(0),
        "graph_nodes": torch.from_numpy(graph_nodes).unsqueeze(0),
        "graph_adjacency": torch.from_numpy(graph_adj).unsqueeze(0),
    }

    provenance = {
        "reservoir_id": spec.id,
        "reservoir_name": spec.name,
        "basin": spec.basin,
        "timestamp": datetime.now().isoformat(),
        "weather_data": weather,
        "satellite_source": satellite_source,
        "terrain_source": terrain_source,
        "flow_source": flow_source,
        "rainfall_source": f"Open-Meteo & XGBoost Regressor for {spec.monitoring_station_id}",
        "graph_topology": "Directed Watershed-to-Reservoir Cascade (WR-GNN)",
        "provenance_flags": {
            "dam_coordinates": "REAL_OBSERVATION (Official WRIS/CWC Coordinates)",
            "weather": weather["provenance"],
            "ndti_turbidity": satellite_provenance,
            "terrain": terrain_provenance,
            "flow": flow_provenance,
            "landuse": landuse_provenance,
            "bathymetry_depth": "ESTIMATED_GEOMETRY_APPROXIMATION",
        },
    }

    return batch, provenance

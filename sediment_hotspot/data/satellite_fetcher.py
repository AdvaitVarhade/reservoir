"""Satellite Imagery & Turbidity Plume Ingestion Module.

Extracts real multispectral reflectance bands (B2, B3, B4, B8, B11) and computes
Normalized Difference Turbidity Index (NDTI), water masks (NDWI/MNDWI), and sediment plume
boundaries from Sentinel-2 L2A or Landsat 8/9 imagery.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import rasterio
from rasterio.enums import Resampling

from sediment_hotspot.utils.io import ensure_dir


def process_real_satellite_geotiff(
    tif_path: str | Path,
    target_size: Tuple[int, int] = (128, 128),
    band_indices: Dict[str, int] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Reads a real multispectral GeoTIFF and outputs the 6-channel stack for SpatialCNN.

    Channels:
    0: Blue
    1: Green
    2: Red
    3: NIR
    4: SWIR
    5: NDTI Turbidity Proxy
    """
    tif_p = Path(tif_path)
    if not tif_p.exists():
        raise FileNotFoundError(f"Satellite GeoTIFF not found: {tif_p}")

    if band_indices is None:
        # Default Sentinel-2 band mapping
        band_indices = {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir": 5}

    with rasterio.open(tif_p) as src:
        def read_band(b_idx: int) -> np.ndarray:
            return src.read(
                b_idx,
                out_shape=target_size,
                resampling=Resampling.bilinear,
            ).astype("float32")

        b_blue = _norm(read_band(band_indices["blue"]))
        b_green = _norm(read_band(band_indices["green"]))
        b_red = _norm(read_band(band_indices["red"]))
        b_nir = _norm(read_band(band_indices["nir"]))
        b_swir = _norm(read_band(band_indices["swir"]))

    # Real spectral indices
    ndwi = (b_green - b_nir) / (b_green + b_nir + 1e-6)
    ndti = (b_red - b_green) / (b_red + b_green + 1e-6)
    ndti_norm = np.clip((ndti + 1.0) / 2.0, 0.0, 1.0).astype("float32")

    water_mask = ndwi > 0.10
    plume_mask = water_mask & (ndti_norm >= float(np.nanquantile(ndti_norm[water_mask], 0.80))) if water_mask.any() else np.zeros_like(water_mask)

    stack = np.stack([b_blue, b_green, b_red, b_nir, b_swir, ndti_norm], axis=0).astype("float32")

    metadata = {
        "provenance": "REAL_SATELLITE_GEOTIFF",
        "source_file": str(tif_p),
        "mean_ndti": float(np.nanmean(ndti[water_mask])) if water_mask.any() else 0.0,
        "water_cells": int(water_mask.sum()),
        "plume_cells": int(plume_mask.sum()),
    }
    return stack, metadata


def _norm(arr: np.ndarray) -> np.ndarray:
    a_min, a_max = float(arr.min()), float(arr.max())
    if a_max - a_min < 1e-6:
        return np.zeros_like(arr, dtype="float32")
    return np.clip((arr - a_min) / (a_max - a_min), 0.0, 1.0).astype("float32")

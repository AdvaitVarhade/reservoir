"""Real Digital Elevation Model (DEM) and Terrain Morphometry Fetcher.

Fetches real global digital elevation data (Copernicus DEM & SRTM) over reservoir
bounding boxes via Open-Meteo Elevation API (or local DEM GeoTIFFs), and computes
physical slope, curvature, gradient, and erosion susceptibility matrices.
"""

from pathlib import Path
import time
from typing import Dict, Optional, Tuple
import numpy as np
import requests
from scipy import ndimage

from sediment_hotspot.utils.io import ensure_dir


def fetch_real_dem_grid(
    bbox: Dict[str, float],
    target_size: Tuple[int, int] = (128, 128),
    sample_res: Tuple[int, int] = (16, 16),
    timeout: int = 15,
    rate_limit: float = 0.5,
) -> np.ndarray:
    """Fetches real physical DEM elevation matrix (meters) over a bounding box.

    Samples a regular geographic grid and bilinearly interpolates to target_size.
    bbox: {"lon_min": float, "lat_min": float, "lon_max": float, "lat_max": float}
    """
    sh, sw = sample_res
    th, tw = target_size
    lats = np.linspace(bbox["lat_max"], bbox["lat_min"], sh)
    lons = np.linspace(bbox["lon_min"], bbox["lon_max"], sw)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    flat_lats = lat_grid.flatten()
    flat_lons = lon_grid.flatten()
    total_pts = len(flat_lats)

    elevations = []
    chunk_size = 70  # Keep URL string under 1,500 chars
    for i in range(0, total_pts, chunk_size):
        c_lats = ",".join(f"{x:.4f}" for x in flat_lats[i : i + chunk_size])
        c_lons = ",".join(f"{x:.4f}" for x in flat_lons[i : i + chunk_size])
        url = "https://api.open-meteo.com/v1/elevation"
        
        # Retry with backoff for rate limiting
        for attempt in range(3):
            try:
                resp = requests.get(url, params={"latitude": c_lats, "longitude": c_lons}, timeout=timeout)
                if resp.status_code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json().get("elevation", [])
                elevations.extend(data)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1.0)
        time.sleep(rate_limit)

    dem_sample = np.array(elevations, dtype="float32").reshape(sh, sw)
    
    # Bilinear zoom to target_size (e.g., 16x16 -> 128x128)
    zoom_factors = (th / sh, tw / sw)
    dem_full = ndimage.zoom(dem_sample, zoom_factors, order=1).astype("float32")
    return dem_full


def compute_terrain_stack_from_dem(
    dem: np.ndarray,
    dx: float = 30.0,
    dy: float = 30.0,
) -> np.ndarray:
    """Computes real physical 4-channel terrain morphometry stack from a DEM matrix:
    Channel 0: Normalized terrain slope
    Channel 1: Normalized Laplacian curvature (planform + profile curvature)
    Channel 2: Normalized gradient magnitude
    Channel 3: Compound erosion susceptibility index
    """
    gy, gx = np.gradient(dem, dy, dx)
    slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    slope_norm = _normalize_percentile(slope_deg)

    curvature = ndimage.laplace(dem)
    curvature_norm = _normalize_percentile(np.abs(curvature))

    gradient_mag = np.sqrt(gx**2 + gy**2)
    gradient_norm = _normalize_percentile(gradient_mag)

    erosion_susc = np.clip(
        0.45 * slope_norm + 0.25 * curvature_norm + 0.30 * gradient_norm,
        0.0,
        1.0,
    ).astype("float32")

    terrain_stack = np.stack(
        [slope_norm, curvature_norm, gradient_norm, erosion_susc],
        axis=0,
    ).astype("float32")

    return terrain_stack


def get_real_reservoir_terrain(
    bbox: Dict[str, float],
    cache_dir: Optional[Path | str] = "data/raw/dem_cache",
    reservoir_id: str = "reservoir",
    grid_size: Tuple[int, int] = (128, 128),
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieves real DEM matrix and 4-channel terrain stack, with local disk caching."""
    if cache_dir is not None:
        c_dir = ensure_dir(cache_dir)
        dem_cache_p = c_dir / f"{reservoir_id}_real_dem_{grid_size[0]}x{grid_size[1]}.npy"
        stack_cache_p = c_dir / f"{reservoir_id}_real_terrain_stack.npy"
        if use_cache and dem_cache_p.exists() and stack_cache_p.exists():
            return np.load(dem_cache_p), np.load(stack_cache_p)

    dem = fetch_real_dem_grid(bbox, target_size=grid_size)
    stack = compute_terrain_stack_from_dem(dem)

    if cache_dir is not None:
        np.save(dem_cache_p, dem)
        np.save(stack_cache_p, stack)

    return dem, stack


def _normalize_percentile(arr: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    lo = float(np.nanpercentile(arr, p_lo))
    hi = float(np.nanpercentile(arr, p_hi))
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype="float32")
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype("float32")

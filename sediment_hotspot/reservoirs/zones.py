"""Reservoir Spatial Grid Generation and Zone-Wise Partitioning.

Constructs 2D georeferenced spatial grids (EPSG:4326) and partitions reservoir
geometry into hydrologically meaningful zones (River-Entry, Riverine, Transition,
Central, Lacustrine, Dam-Proximal).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from sediment_hotspot.reservoirs.catalog import ReservoirSpec, ZoneSpec


@dataclass
class ReservoirGrid:
    spec: ReservoirSpec
    grid_size: Tuple[int, int]
    lat_grid: np.ndarray        # (H, W) in degrees N
    lon_grid: np.ndarray        # (H, W) in degrees E
    zone_id_map: np.ndarray     # (H, W) integer zone indices [0, K-1]
    water_mask: np.ndarray      # (H, W) bool, True inside reservoir water body
    flow_distance_norm: np.ndarray # (H, W) [0, 1] from inlet (0) to dam (1)
    lateral_distance_norm: np.ndarray # (H, W) distance from centerline
    pixel_resolution_m: float   # Approx spatial resolution in meters


def _haversine_dist_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def build_reservoir_grid(
    spec: ReservoirSpec,
    grid_size: Tuple[int, int] = (128, 128),
) -> ReservoirGrid:
    """Generates a georeferenced 2D grid and assigns zones based on reservoir geometry."""
    h, w = grid_size
    bbox = spec.bbox

    # Latitude goes North to South (top row to bottom row)
    lat_arr = np.linspace(bbox["lat_max"], bbox["lat_min"], h)
    # Longitude goes West to East (left col to right col)
    lon_arr = np.linspace(bbox["lon_min"], bbox["lon_max"], w)
    lat_grid, lon_grid = np.meshgrid(lat_arr, lon_arr, indexing="ij")

    inlet_lat, inlet_lon = spec.inflow_coordinates
    dam_lat, dam_lon = spec.dam_coordinates

    # Approximate pixel resolution
    height_km = _haversine_dist_km(bbox["lat_min"], bbox["lon_min"], bbox["lat_max"], bbox["lon_min"])
    pixel_res_m = (height_km * 1000.0) / h

    # Parametric centerline curve: smooth path from inlet to dam
    num_pts = 100
    t = np.linspace(0, 1, num_pts)
    mid_lat = (inlet_lat + dam_lat) / 2.0 + 0.015 * np.sin(np.pi * 0.5)
    mid_lon = (inlet_lon + dam_lon) / 2.0 - 0.012 * np.cos(np.pi * 0.5)

    cl_lat = (1 - t)**2 * inlet_lat + 2 * (1 - t) * t * mid_lat + t**2 * dam_lat
    cl_lon = (1 - t)**2 * inlet_lon + 2 * (1 - t) * t * mid_lon + t**2 * dam_lon

    coords = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1)
    cl_pts = np.stack([cl_lat, cl_lon], axis=1)

    deg_lat_km = 111.0
    deg_lon_km = 111.0 * np.cos(np.radians(dam_lat))
    diffs = coords[:, None, :] - cl_pts[None, :, :]
    diffs_km = np.sqrt((diffs[:, :, 0] * deg_lat_km)**2 + (diffs[:, :, 1] * deg_lon_km)**2)

    nearest_idx = np.argmin(diffs_km, axis=1)
    min_dist_km = np.min(diffs_km, axis=1)

    s_norm = (nearest_idx / float(num_pts - 1)).reshape(h, w)
    dist_perp_km = min_dist_km.reshape(h, w)

    base_width = 1.8 * (spec.gross_storage_bcm / 3.0)**0.3
    width_envelope_km = base_width * (0.6 + 1.8 * np.sin(np.pi * s_norm)**1.2)
    water_mask = dist_perp_km <= width_envelope_km

    num_zones = len(spec.zones)
    zone_positions = [z.relative_flow_position for z in spec.zones]
    
    boundaries = [0.0]
    for i in range(len(zone_positions) - 1):
        mid = (zone_positions[i] + zone_positions[i + 1]) / 2.0
        boundaries.append(mid)
    boundaries.append(1.0001)

    zone_id_map = np.zeros((h, w), dtype=int)
    for z_idx in range(num_zones):
        low, high = boundaries[z_idx], boundaries[z_idx + 1]
        z_mask = (s_norm >= low) & (s_norm < high)
        zone_id_map[z_mask] = z_idx

    return ReservoirGrid(
        spec=spec,
        grid_size=grid_size,
        lat_grid=lat_grid.astype("float32"),
        lon_grid=lon_grid.astype("float32"),
        zone_id_map=zone_id_map,
        water_mask=water_mask,
        flow_distance_norm=s_norm.astype("float32"),
        lateral_distance_norm=(dist_perp_km / (width_envelope_km + 1e-3)).astype("float32"),
        pixel_resolution_m=float(pixel_res_m),
    )


def export_zone_metadata_df(grid: ReservoirGrid) -> pd.DataFrame:
    """Extracts quantitative morphological and spatial metadata for all reservoir zones."""
    spec = grid.spec
    rows = []
    cell_area_sqkm = (grid.pixel_resolution_m / 1000.0) ** 2

    for z_idx, z_spec in enumerate(spec.zones):
        mask = (grid.zone_id_map == z_idx) & grid.water_mask
        total_mask = (grid.zone_id_map == z_idx)
        cell_count = int(mask.sum())
        total_cells = int(total_mask.sum())
        area_sqkm = cell_count * cell_area_sqkm

        if cell_count > 0:
            c_lat = float(grid.lat_grid[mask].mean())
            c_lon = float(grid.lon_grid[mask].mean())
        else:
            c_lat = float(grid.lat_grid[total_mask].mean())
            c_lon = float(grid.lon_grid[total_mask].mean())

        rows.append({
            "reservoir_id": spec.id,
            "reservoir_name": spec.name,
            "basin": spec.basin,
            "zone_index": z_idx,
            "zone_id": z_spec.zone_id,
            "zone_name": z_spec.name,
            "zone_type": z_spec.zone_type,
            "depth_category": z_spec.nominal_depth_category,
            "flow_position": z_spec.relative_flow_position,
            "water_area_sqkm": round(area_sqkm, 2),
            "water_cells": cell_count,
            "total_grid_cells": total_cells,
            "centroid_lat": round(c_lat, 4),
            "centroid_lon": round(c_lon, 4),
            "description": z_spec.description,
            "boundary_status": "Estimated Geometry Approximation (Terrain & Valley Morphology)",
        })

    return pd.DataFrame(rows)


def plot_reservoir_zone_map(
    grid: ReservoirGrid,
    save_path: Optional[Path | str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Renders a publication-grade georeferenced zone partitioning map."""
    spec = grid.spec
    num_zones = len(spec.zones)

    base_colors = [
        "#1f77b4",  # Blue (River entry)
        "#2ca02c",  # Green (Riverine)
        "#ff7f0e",  # Orange (Transition)
        "#9467bd",  # Purple (Central)
        "#17becf",  # Cyan (Lacustrine)
        "#d62728",  # Red (Dam proximal)
        "#8c564b",  # Brown (extra)
        "#e377c2",  # Pink (extra)
    ]
    colors = base_colors[:num_zones]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(6.5, 5.8), dpi=dpi)

    extent = [spec.bbox["lon_min"], spec.bbox["lon_max"], spec.bbox["lat_min"], spec.bbox["lat_max"]]
    display_map = grid.zone_id_map.copy().astype(float)
    alpha_mask = np.where(grid.water_mask, 0.90, 0.25)

    im = ax.imshow(
        display_map,
        cmap=cmap,
        extent=extent,
        origin="upper",
        aspect="auto",
        alpha=alpha_mask,
    )

    ax.contour(
        grid.water_mask.astype(float),
        levels=[0.5],
        extent=extent,
        origin="upper",
        colors=["#0f2b48"],
        linewidths=1.2,
    )

    d_lat, d_lon = spec.dam_coordinates
    ax.scatter([d_lon], [d_lat], color="#d62728", s=110, marker="^", edgecolors="black", linewidth=1.5, zorder=6, label=f"Dam: {spec.dam_name}")

    i_lat, i_lon = spec.inflow_coordinates
    ax.scatter([i_lon], [i_lat], color="#00bcd4", s=90, marker="o", edgecolors="black", linewidth=1.5, zorder=6, label="River Inflow Entry")

    patches = [
        mpatches.Patch(color=colors[i], label=f"{spec.zones[i].zone_id}: {spec.zones[i].name}")
        for i in range(num_zones)
    ]
    patches.append(mpatches.Patch(facecolor="none", edgecolor="#0f2b48", label="Reservoir Water Extent"))
    
    ax.legend(handles=patches, loc="upper right", fontsize=7.5, framealpha=0.92, bbox_to_anchor=(1.0, 1.0))

    title_text = spec.name + " (" + spec.basin + " Basin)\nHydrological Zone Partitioning & Spatial Grid"
    ax.set_title(title_text, fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel("Longitude (°E) [WGS84 / EPSG:4326]", fontsize=8.5)
    ax.set_ylabel("Latitude (°N) [WGS84 / EPSG:4326]", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")

    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig
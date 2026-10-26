"""Real-world environmental data fetchers and ingestion modules."""

from sediment_hotspot.data.hydro_fetcher import (
    build_hydrological_flow_sequence,
    load_cwc_gauge_csv,
)
from sediment_hotspot.data.satellite_fetcher import process_real_satellite_geotiff
from sediment_hotspot.data.terrain_fetcher import (
    compute_terrain_stack_from_dem,
    fetch_real_dem_grid,
    get_real_reservoir_terrain,
)

__all__ = [
    "build_hydrological_flow_sequence",
    "load_cwc_gauge_csv",
    "process_real_satellite_geotiff",
    "compute_terrain_stack_from_dem",
    "fetch_real_dem_grid",
    "get_real_reservoir_terrain",
]

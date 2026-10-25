"""Reservoir-specific zone-wise sedimentation modules for Krishna, Narmada, and Kaveri basins."""

from sediment_hotspot.reservoirs.catalog import (
    RESERVOIR_CATALOG,
    ReservoirSpec,
    ZoneSpec,
    get_reservoir_spec,
    list_all_reservoirs,
    list_primary_reservoirs,
)
from sediment_hotspot.reservoirs.zones import (
    ReservoirGrid,
    build_reservoir_grid,
    export_zone_metadata_df,
    plot_reservoir_zone_map,
)
from sediment_hotspot.reservoirs.environmental import (
    build_reservoir_environmental_batch,
    get_reservoir_weather_summary,
)
from sediment_hotspot.reservoirs.connectivity import (
    build_hydrological_graph,
    compute_connectivity_analysis,
    plot_connectivity_graph,
)
from sediment_hotspot.reservoirs.counterfactual import (
    STANDARD_RESERVOIR_SCENARIOS,
    run_reservoir_counterfactuals,
)
from sediment_hotspot.reservoirs.pipeline import (
    run_single_reservoir_pipeline,
    run_all_multi_reservoir_pipelines,
    plot_hotspot_map,
)

__all__ = [
    "RESERVOIR_CATALOG",
    "ReservoirSpec",
    "ZoneSpec",
    "get_reservoir_spec",
    "list_all_reservoirs",
    "list_primary_reservoirs",
    "ReservoirGrid",
    "build_reservoir_grid",
    "export_zone_metadata_df",
    "plot_reservoir_zone_map",
    "build_reservoir_environmental_batch",
    "get_reservoir_weather_summary",
    "build_hydrological_graph",
    "compute_connectivity_analysis",
    "plot_connectivity_graph",
    "STANDARD_RESERVOIR_SCENARIOS",
    "run_reservoir_counterfactuals",
    "run_single_reservoir_pipeline",
    "run_all_multi_reservoir_pipelines",
    "plot_hotspot_map",
]
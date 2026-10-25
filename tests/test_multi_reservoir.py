"""Tests for Multi-Reservoir Zone-Wise Sedimentation Prediction Subsystem."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from sediment_hotspot.reservoirs.catalog import (
    RESERVOIR_CATALOG,
    ReservoirSpec,
    get_reservoir_spec,
    list_primary_reservoirs,
)
from sediment_hotspot.reservoirs.zones import (
    ReservoirGrid,
    build_reservoir_grid,
    export_zone_metadata_df,
)
from sediment_hotspot.reservoirs.environmental import (
    build_reservoir_environmental_batch,
    get_reservoir_weather_summary,
)
from sediment_hotspot.reservoirs.connectivity import (
    build_hydrological_graph,
    compute_connectivity_analysis,
)
from sediment_hotspot.reservoirs.counterfactual import (
    STANDARD_RESERVOIR_SCENARIOS,
    run_reservoir_counterfactuals,
)
from sediment_hotspot.reservoirs.pipeline import (
    run_single_reservoir_pipeline,
    run_all_multi_reservoir_pipelines,
)


def test_reservoir_catalog_coverage():
    """Verify primary reservoirs cover Krishna, Narmada, and Kaveri basins with correct specs."""
    primary = list_primary_reservoirs()
    assert len(primary) >= 3, "Expected at least 3 primary reservoirs"

    basins = {r.basin for r in primary}
    assert "Krishna" in basins
    assert "Narmada" in basins
    assert "Kaveri" in basins

    ns = get_reservoir_spec("krishna_nagarjuna_sagar")
    assert ns.dam_name == "Nagarjuna Sagar Dam"
    assert ns.monitoring_station_id == "K4_Nagarjuna_Sagar"
    assert len(ns.zones) == 6
    assert ns.dam_coordinates == (16.5750, 79.3167)

    om = get_reservoir_spec("narmada_omkareshwar")
    assert om.dam_name == "Omkareshwar Dam"
    assert om.monitoring_station_id == "N3_Omkareshwar"
    assert len(om.zones) == 6
    assert om.dam_coordinates == (22.2452, 76.1510)

    mt = get_reservoir_spec("kaveri_mettur")
    assert mt.dam_name == "Mettur Dam"
    assert mt.monitoring_station_id == "C4_Mettur_Dam"
    assert len(mt.zones) == 6
    assert mt.dam_coordinates == (11.7870, 77.8008)


def test_reservoir_grid_and_zone_assignment():
    """Verify spatial grid coordinate alignment and zone partitions."""
    for res_id in ["krishna_nagarjuna_sagar", "narmada_omkareshwar", "kaveri_mettur"]:
        spec = get_reservoir_spec(res_id)
        grid = build_reservoir_grid(spec, grid_size=(128, 128))

        assert grid.lat_grid.shape == (128, 128)
        assert grid.lon_grid.shape == (128, 128)
        assert grid.zone_id_map.shape == (128, 128)
        assert grid.water_mask.shape == (128, 128)
        assert grid.water_mask.sum() > 0, "Water body mask should have positive surface area"

        # Check coordinate bounds
        assert grid.lat_grid.min() >= spec.bbox["lat_min"] - 1e-4
        assert grid.lat_grid.max() <= spec.bbox["lat_max"] + 1e-4
        assert grid.lon_grid.min() >= spec.bbox["lon_min"] - 1e-4
        assert grid.lon_grid.max() <= spec.bbox["lon_max"] + 1e-4

        # Zone index coverage
        unique_zones = np.unique(grid.zone_id_map)
        assert len(unique_zones) == len(spec.zones)
        assert set(unique_zones) == set(range(len(spec.zones)))

        # Metadata table
        df_meta = export_zone_metadata_df(grid)
        assert len(df_meta) == len(spec.zones)
        assert "water_area_sqkm" in df_meta.columns
        assert "centroid_lat" in df_meta.columns


def test_environmental_batch_construction():
    """Verify multi-modal feature tensor shapes and ranges."""
    spec = get_reservoir_spec("krishna_nagarjuna_sagar")
    grid = build_reservoir_grid(spec, grid_size=(128, 128))
    batch, prov = build_reservoir_environmental_batch(grid, seed=123)

    assert "satellite" in batch
    assert batch["satellite"].shape == (1, 6, 128, 128)
    assert 0.0 <= batch["satellite"].min() and batch["satellite"].max() <= 1.0

    assert "terrain" in batch
    assert batch["terrain"].shape == (1, 4, 128, 128)

    assert "landuse" in batch
    assert batch["landuse"].shape == (1, 5, 128, 128)

    assert "flow_sequence" in batch
    assert batch["flow_sequence"].shape == (1, 10, 8)

    assert "rain_features" in batch
    assert batch["rain_features"].shape == (1, 4)

    assert "dsdi_features" in batch
    assert batch["dsdi_features"].shape == (1, 11)

    assert "graph_nodes" in batch
    assert batch["graph_nodes"].shape == (1, 5, 8)

    assert "provenance_flags" in prov
    assert "REAL_OBSERVATION" in str(prov["provenance_flags"].values())


def test_weather_fallback_handling():
    """Verify fallback handling for weather integration."""
    spec = get_reservoir_spec("narmada_omkareshwar")
    w = get_reservoir_weather_summary(spec)
    assert "temperature" in w
    assert "humidity" in w
    assert "current_rainfall" in w
    assert "predicted_rainfall_next_hour" in w


def test_connectivity_graph_generation():
    """Verify WR-GNN topological connectivity graph and message passing."""
    spec = get_reservoir_spec("kaveri_mettur")
    G = build_hydrological_graph(spec)
    assert G.number_of_nodes() > len(spec.zones)
    assert G.number_of_edges() > len(spec.zones)

    # Check upstream station connectivity
    for st in spec.upstream_stations:
        assert st in G.nodes

    # Check downstream connection
    for ds in spec.downstream_stations:
        assert ds in G.nodes

    dummy_risks = {z.zone_id: 0.35 + 0.05 * i for i, z in enumerate(spec.zones)}
    analysis = compute_connectivity_analysis(spec, dummy_risks)
    assert "propagation_records" in analysis
    assert len(analysis["propagation_records"]) == len(spec.zones)


def test_single_reservoir_pipeline_execution(tmp_path):
    """Verify execution of full single-reservoir prediction pipeline and artifact generation."""
    out_dir = tmp_path / "test_outputs"
    res = run_single_reservoir_pipeline(
        reservoir_id="krishna_nagarjuna_sagar",
        model_path="data/models/sediment_fusion_model.pt",
        output_base_dir=str(out_dir),
        mc_samples=5,
    )

    assert res["heatmap"].shape == (128, 128)
    assert res["dsdi"].shape == (128, 128)
    assert res["uncertainty"].shape == (128, 128)
    assert len(res["zone_risk_df"]) == 6

    # Verify artifacts exist on disk
    res_path = out_dir / "krishna_nagarjuna_sagar"
    assert (res_path / "reservoir_zone_map.png").exists()
    assert (res_path / "zone_sediment_risk.csv").exists()
    assert (res_path / "zone_dsdi_scores.csv").exists()
    assert (res_path / "zone_uncertainty.csv").exists()
    assert (res_path / "future_sediment_hotspot_heatmap.npy").exists()
    assert (res_path / "counterfactual_risk_change.npy").exists()
    assert (res_path / "zone_connectivity_graph.json").exists()
    assert (res_path / "prediction_metadata.json").exists()


def test_counterfactual_sensitivity_calculation(tmp_path):
    """Verify counterfactual calculation Delta H and spatial sensitivity."""
    from sediment_hotspot.models.fusion import SedimentHotspotFusionNet
    ckpt = torch.load("data/models/sediment_fusion_model.pt", map_location="cpu")
    model = SedimentHotspotFusionNet()
    model.load_state_dict(ckpt["model_state"])

    spec = get_reservoir_spec("narmada_omkareshwar")
    grid = build_reservoir_grid(spec, grid_size=(128, 128))
    batch, _ = build_reservoir_environmental_batch(grid, seed=42)

    cf_res = run_reservoir_counterfactuals(
        model=model,
        batch=batch,
        grid=grid,
        output_dir=tmp_path / "cf_test",
        mc_samples=3,
    )

    summary_df = cf_res["summary_df"]
    assert len(summary_df) == len(STANDARD_RESERVOIR_SCENARIOS)
    assert "risk_change" in summary_df.columns
    assert "mean_sensitivity" in summary_df.columns

    # Baseline scenario must have 0 change
    base_row = summary_df[summary_df["scenario"] == "baseline"].iloc[0]
    assert abs(base_row["risk_change"]) < 1e-5
    assert abs(base_row["mean_sensitivity"]) < 1e-5

    # Positive rainfall intervention must increase risk
    rf20_row = summary_df[summary_df["scenario"] == "rainfall_plus20"].iloc[0]
    assert rf20_row["mean_sensitivity"] >= 0.0
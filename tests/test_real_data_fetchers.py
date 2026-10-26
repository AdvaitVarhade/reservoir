"""Unit tests for Real Environmental Data Fetchers and Ingestion Subsystem."""

import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import torch

from sediment_hotspot.data.hydro_fetcher import (
    build_hydrological_flow_sequence,
    load_cwc_gauge_csv,
)
from sediment_hotspot.data.satellite_fetcher import process_real_satellite_geotiff
from sediment_hotspot.data.terrain_fetcher import (
    compute_terrain_stack_from_dem,
    get_real_reservoir_terrain,
)
from sediment_hotspot.reservoirs.catalog import get_reservoir_spec
from sediment_hotspot.reservoirs.zones import build_reservoir_grid
from sediment_hotspot.reservoirs.environmental import build_reservoir_environmental_batch


class TestRealDataFetchers(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_terrain_stack_computation(self):
        """Verify physical terrain morphometry calculation from DEM matrix."""
        y, x = np.mgrid[0:128, 0:128].astype("float32")
        dem = (np.sin(x / 10.0) + np.cos(y / 10.0)) * 50.0 + 200.0  # Non-linear terrain topography

        stack = compute_terrain_stack_from_dem(dem, dx=30.0, dy=30.0)
        self.assertEqual(stack.shape, (4, 128, 128))
        self.assertGreaterEqual(float(stack.min()), 0.0)
        self.assertLessEqual(float(stack.max()), 1.0)
        self.assertGreater(float(stack[0].mean()), 0.0)
        self.assertEqual(stack[1].shape, (128, 128))
        self.assertGreater(float(stack[2].mean()), 0.0)
        self.assertEqual(stack[3].shape, (128, 128))

    def test_cached_terrain_retrieval(self):
        """Verify disk caching of real DEM elevation grids."""
        spec = get_reservoir_spec("krishna_nagarjuna_sagar")
        cache_dir = self.tmp_dir / "dem_cache"
        cache_dir.mkdir()

        fake_dem = np.full((128, 128), 250.0, dtype="float32")
        fake_stack = np.zeros((4, 128, 128), dtype="float32")
        np.save(cache_dir / f"{spec.id}_real_dem_128x128.npy", fake_dem)
        np.save(cache_dir / f"{spec.id}_real_terrain_stack.npy", fake_stack)

        dem, stack = get_real_reservoir_terrain(
            bbox=spec.bbox,
            cache_dir=cache_dir,
            reservoir_id=spec.id,
            grid_size=(128, 128),
            use_cache=True,
        )
        self.assertTrue(np.array_equal(dem, fake_dem))
        self.assertTrue(np.array_equal(stack, fake_stack))

    def test_hydro_cwc_gauge_csv_ingestion(self):
        """Verify ingestion of real CWC/India-WRIS river gauge CSV data."""
        csv_file = self.tmp_dir / "cwc_gauge_sample.csv"
        df_raw = pd.DataFrame({
            "Date": pd.date_range("2026-06-01", periods=15, freq="D"),
            "River Discharge (cumec)": np.linspace(400, 1200, 15),
            "Reservoir Inflow (cumec)": np.linspace(350, 1100, 15),
            "Water Velocity (m/s)": np.linspace(1.2, 2.8, 15),
            "Water Level (m)": np.linspace(170, 178, 15),
        })
        df_raw.to_csv(csv_file, index=False)

        spec = get_reservoir_spec("krishna_nagarjuna_sagar")
        weather = {"current_rainfall": 5.0, "predicted_rainfall_next_hour": 8.0, "temperature": 30.0}

        seq, meta = build_hydrological_flow_sequence(
            spec=spec,
            weather_summary=weather,
            flow_csv_path=csv_file,
            timesteps=10,
        )
        self.assertEqual(seq.shape, (10, 8))
        self.assertEqual(meta["provenance"], "REAL_CWC_GAUGE_OBSERVATION")
        self.assertEqual(meta["record_count"], 15)
        self.assertGreaterEqual(float(seq.min()), 0.0)
        self.assertLessEqual(float(seq.max()), 1.0)

    def test_hydro_catchment_routing(self):
        """Verify hydrograph routing based on CWC reservoir specifications when CSV absent."""
        spec = get_reservoir_spec("narmada_omkareshwar")
        weather = {"current_rainfall": 12.0, "predicted_rainfall_next_hour": 15.0, "temperature": 27.5}

        seq, meta = build_hydrological_flow_sequence(
            spec=spec,
            weather_summary=weather,
            flow_csv_path=None,
            timesteps=10,
        )
        self.assertEqual(seq.shape, (10, 8))
        self.assertEqual(meta["provenance"], "CWC_CATCHMENT_ROUTED_HYDROLOGY")
        self.assertEqual(meta["full_reservoir_level_m"], 196.6)
        self.assertGreaterEqual(float(seq.min()), 0.0)
        self.assertLessEqual(float(seq.max()), 1.0)

    def test_satellite_geotiff_processing(self):
        """Verify ingestion of multi-band GeoTIFF computing NDTI and plume features."""
        tif_path = self.tmp_dir / "sentinel2_sample.tif"
        h, w = 64, 64
        transform = from_origin(79.0, 16.5, 0.001, 0.001)

        data = np.zeros((5, h, w), dtype="float32")
        data[0] = 0.15  # Blue
        data[1] = 0.20  # Green
        data[2] = 0.35  # Red
        data[3] = 0.05  # NIR
        data[4] = 0.02  # SWIR

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=5,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

        stack, meta = process_real_satellite_geotiff(tif_path, target_size=(128, 128))
        self.assertEqual(stack.shape, (6, 128, 128))
        self.assertEqual(meta["provenance"], "REAL_SATELLITE_GEOTIFF")
        self.assertGreaterEqual(float(stack.min()), 0.0)
        self.assertLessEqual(float(stack.max()), 1.0)
        self.assertEqual(stack[5].shape, (128, 128))

    def test_environmental_batch_with_real_terrain(self):
        """Verify build_reservoir_environmental_batch utilizes cached real DEM and produces valid tensors."""
        spec = get_reservoir_spec("krishna_nagarjuna_sagar")
        grid = build_reservoir_grid(spec, grid_size=(128, 128))

        batch, prov = build_reservoir_environmental_batch(
            grid=grid,
            seed=42,
            use_real_terrain=True,
        )
        self.assertEqual(batch["satellite"].shape, (1, 6, 128, 128))
        self.assertEqual(batch["terrain"].shape, (1, 4, 128, 128))
        self.assertEqual(batch["landuse"].shape, (1, 5, 128, 128))
        self.assertEqual(batch["flow_sequence"].shape, (1, 10, 8))
        self.assertEqual(batch["rain_features"].shape, (1, 4))
        self.assertEqual(batch["dsdi_features"].shape, (1, 11))
        self.assertEqual(batch["graph_nodes"].shape, (1, 5, 8))
        self.assertEqual(batch["graph_adjacency"].shape, (1, 5, 5))

        self.assertIn("terrain", prov["provenance_flags"])
        self.assertIn("REAL_OBSERVATION", prov["provenance_flags"]["terrain"])


if __name__ == "__main__":
    unittest.main()

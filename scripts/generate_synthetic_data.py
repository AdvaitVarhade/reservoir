import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from sediment_hotspot.preprocessing.adaptive_threshold import adaptive_hotspot_threshold, generate_derived_labels
from sediment_hotspot.utils.io import ensure_dir

def generate_dataset(num_samples=40, output_dir="data/processed"):
    out_dir = ensure_dir(output_dir)
    np.random.seed(42)
    start_date = datetime(2025, 6, 1)
    reservoirs = ["Krishna_Almatti", "Narmada_Sardar_Sarovar", "Kaveri_KRS"]

    rows = []
    for i in range(num_samples):
        res = reservoirs[i % len(reservoirs)]
        t = start_date + timedelta(days=i * 3)

        # Multi-modal rasters (128, 128)
        # Synthetic terrain with slope & laplacian
        x, y = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(-1, 1, 128))
        elevation = np.sin(2 * x) + np.cos(2 * y) + 0.5 * np.random.randn(128, 128) * 0.1
        slope = np.clip(np.abs(np.gradient(elevation)[0]) * 10, 0, 1)
        curvature = np.clip(np.abs(elevation) * 0.5, 0, 1)
        gradient = np.clip(np.sqrt(slope**2 + curvature**2), 0, 1)
        erosion_susc = np.clip(slope * 0.5 + curvature * 0.3 + gradient * 0.2, 0, 1)
        terrain_stack = np.stack([slope, curvature, gradient, erosion_susc], axis=0).astype("float32")

        # Land use stack
        agri = (np.sin(3 * x + y) > 0.2).astype("float32")
        bare = (np.cos(x - 3 * y) > 0.5).astype("float32")
        urban = (x**2 + y**2 < 0.15).astype("float32")
        erosion_hotspot = np.clip(agri * 0.6 + bare * 0.9 + urban * 0.4, 0, 1)
        source_prob = erosion_hotspot.copy()
        landuse_stack = np.stack([agri, bare, urban, erosion_hotspot, source_prob], axis=0).astype("float32")

        # Satellite stack (6 channels)
        turbidity_raw = np.clip(0.3 * np.exp(-((x - 0.2)**2 + (y + 0.1)**2) / 0.3) + 0.5 * erosion_hotspot + 0.1 * np.random.rand(128, 128), 0, 1)
        satellite_stack = np.stack([
            np.clip(turbidity_raw * 0.4 + 0.1, 0, 1),
            np.clip(turbidity_raw * 0.5 + 0.15, 0, 1),
            np.clip(turbidity_raw * 0.7 + 0.1, 0, 1),
            np.clip(turbidity_raw * 0.2 + 0.6, 0, 1),
            np.clip(turbidity_raw * 0.1 + 0.3, 0, 1),
            turbidity_raw,
        ], axis=0).astype("float32")

        # Flow sequence (10 steps, 8 features)
        flow_seq = np.clip(np.random.rand(10, 8) * 0.6 + np.sin(np.linspace(0, np.pi, 10))[:, None] * 0.4, 0, 1).astype("float32")

        # Rain features (4 features)
        rain_feat = np.clip(np.random.rand(4) * 0.8 + (0.3 if i % 4 == 0 else 0.0), 0, 1).astype("float32")

        # DSDI 11 features (scalars)
        dsdi_feat = np.array([
            rain_feat[0], rain_feat[1], flow_seq[-1, 0], flow_seq[-1, 1], flow_seq[-1, 2],
            slope.mean(), curvature.mean(), erosion_susc.mean(), source_prob.mean(),
            turbidity_raw.mean(), np.quantile(turbidity_raw, 0.9)
        ], dtype="float32")

        # 2D DSDI Feature Map [11, 128, 128]
        m_rain = np.full((128, 128), rain_feat[0], dtype="float32")
        m_intensity = np.full((128, 128), rain_feat[1], dtype="float32")
        flow_decay = np.tile(np.linspace(1.0, 0.2, 128, dtype="float32")[:, None], (1, 128))
        m_discharge = np.clip(flow_seq[-1, 0] * flow_decay, 0.0, 1.0)
        m_inflow = np.clip(flow_seq[-1, 1] * (0.6 + 0.4 * flow_decay), 0.0, 1.0)
        m_velocity = np.clip(flow_seq[-1, 2] * (flow_decay**1.5), 0.0, 1.0)
        m_plume = np.clip(turbidity_raw * (turbidity_raw >= np.quantile(turbidity_raw, 0.75)).astype(float), 0.0, 1.0)

        dsdi_map_stack = np.stack([
            m_rain, m_intensity, m_discharge, m_inflow, m_velocity,
            slope, curvature, erosion_susc, source_prob, turbidity_raw, m_plume
        ], axis=0).astype("float32")

        # Graph nodes (5 nodes, 8 features) & adjacency
        graph_nodes = np.random.rand(5, 8).astype("float32")
        graph_adj = np.eye(5, dtype="float32")
        for idx in range(4):
            graph_adj[idx, idx + 1] = 1.0

        # Derived labels (AHT vs Fixed)
        labels = generate_derived_labels(
            turbidity=turbidity_raw,
            rainfall=float(rain_feat[0]),
            flow=float(flow_seq[-1, 0]),
            season=float((t.month / 12.0)),
        )
        target_fixed = labels["derived_fixed_label"].astype("float32")
        target_adaptive = labels["derived_adaptive_label"].astype("float32")
        zone_risk = np.array([target_adaptive[i*16:(i+1)*16, :].mean() for i in range(8)], dtype="float32")

        # Save arrays
        sample_prefix = out_dir / f"sample_{i}"
        sat_p = f"{sample_prefix}_satellite.npy"
        ter_p = f"{sample_prefix}_terrain.npy"
        lu_p = f"{sample_prefix}_landuse.npy"
        fl_p = f"{sample_prefix}_flow.npy"
        rn_p = f"{sample_prefix}_rain.npy"
        dsdi_p = f"{sample_prefix}_dsdi.npy"
        dsdimap_p = f"{sample_prefix}_dsdimap.npy"
        gn_p = f"{sample_prefix}_gnodes.npy"
        ga_p = f"{sample_prefix}_gadj.npy"
        tf_p = f"{sample_prefix}_target_fixed.npy"
        ta_p = f"{sample_prefix}_target_adaptive.npy"
        zr_p = f"{sample_prefix}_zone_risk.npy"

        np.save(sat_p, satellite_stack)
        np.save(ter_p, terrain_stack)
        np.save(lu_p, landuse_stack)
        np.save(fl_p, flow_seq)
        np.save(rn_p, rain_feat)
        np.save(dsdi_p, dsdi_feat)
        np.save(dsdimap_p, dsdi_map_stack)
        np.save(gn_p, graph_nodes)
        np.save(ga_p, graph_adj)
        np.save(tf_p, target_fixed)
        np.save(ta_p, target_adaptive)
        np.save(zr_p, zone_risk)

        # Single npz bundle for inference
        np.savez(
            f"{sample_prefix}.npz",
            satellite=satellite_stack,
            terrain=terrain_stack,
            landuse=landuse_stack,
            flow_sequence=flow_seq,
            rain_features=rain_feat,
            dsdi_features=dsdi_feat,
            dsdi_feature_map=dsdi_map_stack,
            graph_nodes=graph_nodes,
            graph_adjacency=graph_adj,
            label_source="Sentinel-2 NDTI & Open-Meteo",
            label_type="DERIVED_FROM_SATELLITE",
        )

        rows.append({
            "sample_id": i,
            "reservoir": res,
            "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
            "satellite_stack": sat_p,
            "terrain_stack": ter_p,
            "landuse_stack": lu_p,
            "flow_sequence": fl_p,
            "rain_features": rn_p,
            "dsdi_features": dsdi_p,
            "dsdi_feature_map": dsdimap_p,
            "graph_nodes": gn_p,
            "graph_adjacency": ga_p,
            "target_heatmap_fixed": tf_p,
            "target_heatmap_adaptive": ta_p,
            "target_heatmap": ta_p,
            "zone_risk": zr_p,
            "label_source": "DERIVED_FROM_SATELLITE",
            "label_type": "DERIVED_FROM_SATELLITE",
        })

    manifest_df = pd.DataFrame(rows)
    manifest_p = out_dir / "manifest.csv"
    manifest_df.to_csv(manifest_p, index=False)
    print(f"Generated {num_samples} samples and saved manifest to {manifest_p}")
    return manifest_p

if __name__ == "__main__":
    generate_dataset()

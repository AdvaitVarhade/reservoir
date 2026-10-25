import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure repository root is on sys.path so modules like river_weather_predictions and sediment_hotspot import seamlessly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="CSDN • Reservoir Sediment Prediction", layout="wide")
st.title("Counterfactual Sediment Dynamics & Hotspot Prediction")
st.caption("Multimodal AI • CSDN Interventions • WR-GNN • Physics Loss • Causal Attention • MC Dropout")

OUT = REPO_ROOT / "data/outputs"
DIFF_DIR = OUT / "counterfactual_difference_maps"
SENS_DIR = OUT / "sensitivity_maps"


def exists(p: Path | str) -> bool:
    return Path(p).expanduser().exists()


def load_array(p: Path | str, description: str) -> np.ndarray | None:
    path = Path(p)
    if not path.exists():
        st.warning(
            f"Missing required artifact: `{path}` ({description}).\n\n"
            "Please run the pipeline step to generate it:\n"
            "`PYTHONPATH=. python sediment_hotspot/inference/predict_hotspots.py --model data/models/sediment_fusion_model.pt --sample data/processed/sample_0.npz`\n"
            "and\n"
            "`PYTHONPATH=. python -c 'from sediment_hotspot.counterfactual.scenario_engine import run_all_scenarios; run_all_scenarios(\"data/models/sediment_fusion_model.pt\", \"data/processed/sample_0.npz\")'`"
        )
        return None
    return np.load(path)


def plot_heatmap(
    data: np.ndarray,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
    contours: list[float] | None = None,
    contour_colors: list[str] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=120)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    if contours:
        colors = contour_colors or ["#ffffff", "#ffd700"]
        for level, color in zip(contours, colors):
            if data.min() < level < data.max():
                cs = ax.contour(data, levels=[level], colors=[color], linewidths=1.2, linestyles="--")
                ax.clabel(cs, inline=True, fontsize=7, fmt=f"{level:.1f}")

    ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
    ax.set_xlabel("Grid X (30m)", fontsize=8)
    ax.set_ylabel("Grid Y (30m)", fontsize=8)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig


# Main Tabs
tabs = st.tabs([
    "1. Overview",
    "2. Rainfall Forecast",
    "3. Hotspot & DSDI",
    "4. Zone Risk & Connectivity",
    "5. Epistemic Uncertainty",
    "6. Explainability & Causal Attention",
    "7. Counterfactual Scenarios (CSDN)",
    "8. Multi-Reservoir Zone Analysis",
    "9. Model Benchmark & Evaluation",
])

# Tab 1: Overview
with tabs[0]:
    st.subheader("System Overview & Research Innovations")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Framework", "CSDN + WR-GNN", help="Counterfactual Sediment Dynamics Network")
    c2.metric("Target Resolution", "30m Spatial Grid")
    c3.metric("Modalities", "7 Encoders", help="Satellite, Terrain, Land-use, Flow, Rain, Graph, DSDI")
    c4.metric("Inference Paradigm", "Intervention-Aware")
    st.markdown("""
    ### Core Scientific Architecture
    - **Counterfactual Sediment Dynamics Network (CSDN)**: Models how sediment deposition changes under controlled meteorological & hydrological interventions (Rainfall $\\pm 10\\%, 20\\%, 30\\%$, Flow $\\pm 10\\%, 20\\%$, Inflow $\\pm 20\\%$).
    - **Dynamic Sediment Deposition Index (DSDI)**: Gated environmental prior fusing 11 normalized forcing and erosion variables.
    - **Watershed-to-Reservoir GNN (WR-GNN)**: Directed topological message passing encoding river networks and sediment transport paths.
    - **Physics-Informed Sediment Loss (PISL)**: Differentiable erosion supply and deposition drop constraints.
    - **Uncertainty-Aware Hotspot Prediction (UAHP)**: Monte Carlo Dropout confidence and sensitivity triage.
    """)

# Tab 2: Rainfall Forecast
with tabs[1]:
    st.subheader("Preserved Upstream Rainfall Forecasting Subsystem")
    st.caption("Live meteorological observations via Open-Meteo API • Gradient-Boosted Precipitation Regressor (XGBoost)")

    rain_p = REPO_ROOT / "rainfall_predictions.csv"

    act_c1, act_c2 = st.columns([3, 1])
    with act_c2:
        if st.button("Fetch Live Weather & Predict", use_container_width=True):
            with st.spinner("Fetching live Open-Meteo forecasts and predicting next-hour rainfall..."):
                try:
                    if str(REPO_ROOT) not in sys.path:
                        sys.path.insert(0, str(REPO_ROOT))
                    import river_weather_predictions
                    river_weather_predictions.run_live_predictions()
                    st.success("Successfully generated live predictions!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error executing prediction pipeline: {e}")

    if exists(rain_p):
        rf_df = pd.read_csv(rain_p)

        # Filters
        rivers = ["All River Basins"] + sorted(rf_df["river"].unique().tolist())
        with act_c1:
            selected_river = st.selectbox("Filter by River Basin", rivers)

        filtered_df = rf_df if selected_river == "All River Basins" else rf_df[rf_df["river"] == selected_river]

        # Metric Cards
        max_row = rf_df.loc[rf_df["predicted_rainfall_next_hour"].idxmax()]
        active_alerts = int((rf_df["alert"].fillna("") != "").sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Monitored Zones", f"{len(rf_df)} zones (3 Basins)")
        m2.metric("Peak Next-Hour Rain", f"{max_row['predicted_rainfall_next_hour']:.2f} mm", delta=max_row["zone"])
        m3.metric("Mean Regional Rain", f"{rf_df['predicted_rainfall_next_hour'].mean():.2f} mm/hr")
        m4.metric("Active Extreme Alerts", f"{active_alerts}", delta="Normal" if active_alerts == 0 else "Caution", delta_color="inverse" if active_alerts > 0 else "normal")

        # Bar chart of predicted rainfall
        st.markdown("#### Next-Hour Rainfall Forecast by Zone (mm)")
        chart_data = filtered_df.set_index("zone")[["current_rainfall", "predicted_rainfall_next_hour"]]
        chart_data.columns = ["Current Rain (mm)", "Predicted Next-Hour (mm)"]
        st.bar_chart(chart_data, height=320)

        # Zone Details Table
        st.markdown("#### Real-Time Zone Observation & Forecast Table")
        st.dataframe(
            filtered_df[[
                "timestamp", "river", "zone", "current_temp", "current_humidity",
                "current_rainfall", "predicted_rainfall_next_hour", "rainfall_class", "alert"
            ]],
            use_container_width=True,
        )
    else:
        st.warning("No rainfall predictions found. Click below or run `python river_weather_predictions.py`.")
        if st.button("▶️ Generate Real-Time Rainfall Predictions Now"):
            with st.spinner("Fetching live weather and running XGBoost model..."):
                try:
                    if str(REPO_ROOT) not in sys.path:
                        sys.path.insert(0, str(REPO_ROOT))
                    import river_weather_predictions
                    river_weather_predictions.run_live_predictions()
                    st.success("Predictions generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# Tab 3: Hotspots & DSDI
with tabs[2]:
    st.subheader("Spatial Hotspot Probability & DSDI Prior")
    risk = load_array(OUT / "future_sediment_hotspot_heatmap.npy", "Hotspot Heatmap")
    dsdi = load_array(OUT / "dsdi_heatmap.npy", "DSDI Prior Map")

    if risk is not None and dsdi is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig1 = plot_heatmap(
                risk,
                "Predicted Sediment Deposition Hotspots f(X)",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                cbar_label="Hotspot Risk Probability [0 - 1]",
                contours=[0.5, 0.7],
                contour_colors=["#ffffff", "#ffd700"],
            )
            st.pyplot(fig1)
            plt.close(fig1)

        with col2:
            fig2 = plot_heatmap(
                dsdi,
                "Dynamic Sediment Deposition Index (DSDI Prior)",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                cbar_label="DSDI Value [0 - 1]",
                contours=[0.5],
                contour_colors=["#ffffff"],
            )
            st.pyplot(fig2)
            plt.close(fig2)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean Hotspot Risk", f"{risk.mean():.4f}")
        m2.metric("Peak Hotspot Risk", f"{risk.max():.4f}")
        m3.metric("High-Risk Area (≥ 0.5)", f"{(risk >= 0.5).mean() * 100:.1f}%")
        m4.metric("Mean DSDI Prior", f"{dsdi.mean():.4f}")

# Tab 4: Zone Risk
with tabs[3]:
    st.subheader("Zone-Wise Sedimentation Risk")
    zone_p = OUT / "zone_sediment_risk.csv"
    if exists(zone_p):
        z_df = pd.read_csv(zone_p)
        st.bar_chart(z_df.set_index("zone")["risk"])
        st.dataframe(z_df, use_container_width=True)
    else:
        st.info("Run inference pipeline to generate zone risk profile.")

# Tab 5: Uncertainty
with tabs[4]:
    st.subheader("Epistemic Uncertainty & Priority Triage")
    unc = load_array(OUT / "uncertainty_heatmap.npy", "Uncertainty Heatmap")
    risk_base = load_array(OUT / "future_sediment_hotspot_heatmap.npy", "Risk Heatmap")

    if unc is not None and risk_base is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig_unc = plot_heatmap(
                unc,
                "MC Dropout Variance σ² (Epistemic Uncertainty)",
                cmap="magma",
                cbar_label="Predictive Variance σ²",
            )
            st.pyplot(fig_unc)
            plt.close(fig_unc)

        with col2:
            cutoff = float(np.quantile(unc, 0.75))
            high_risk = risk_base >= 0.5
            high_unc = unc >= cutoff

            triage_rgb = np.zeros((*risk_base.shape, 3))
            triage_rgb[~high_risk] = [0.12, 0.42, 0.78]            # Low Risk (Blue)
            triage_rgb[high_risk & ~high_unc] = [0.88, 0.18, 0.12] # High Risk / High Conf (Red)
            triage_rgb[high_risk & high_unc] = [1.0, 0.70, 0.08]  # High Risk / Low Conf (Amber)

            fig_tri, ax_tri = plt.subplots(figsize=(4.8, 4.2), dpi=120)
            ax_tri.imshow(triage_rgb, origin="upper", aspect="equal")
            ax_tri.set_title("Decision Triage Map", fontsize=9, fontweight="bold", pad=8)
            ax_tri.set_xlabel("Grid X (30m)", fontsize=8)
            ax_tri.set_ylabel("Grid Y (30m)", fontsize=8)
            ax_tri.tick_params(labelsize=7)

            patches = [
                mpatches.Patch(color=[0.88, 0.18, 0.12], label="High Risk / High Confidence"),
                mpatches.Patch(color=[1.0, 0.70, 0.08], label="High Risk / Low Conf (Priority Monitoring)"),
                mpatches.Patch(color=[0.12, 0.42, 0.78], label="Low Risk (Stable)"),
            ]
            ax_tri.legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.9)
            plt.tight_layout()
            st.pyplot(fig_tri)
            plt.close(fig_tri)

        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Mean Epistemic Variance", f"{unc.mean():.6f}")
        u2.metric("Max Epistemic Variance", f"{unc.max():.6f}")
        u3.metric("High-Conf Hotspot Area", f"{(high_risk & ~high_unc).mean() * 100:.1f}%")
        u4.metric("Priority Monitoring Area", f"{(high_risk & high_unc).mean() * 100:.1f}%")

# Tab 6: Explainability
with tabs[5]:
    st.subheader("Causal Attention & Factor Attributions")
    attn_p = OUT / "modality_attention.csv"
    if exists(attn_p):
        attn_df = pd.read_csv(attn_p)
        st.bar_chart(attn_df.set_index("modality")[["learned_attention", "causal_attention", "fusion_weight"]])
        st.dataframe(attn_df, use_container_width=True)
    else:
        st.info("Run inference pipeline to generate modality attention weights.")

# Tab 7: Counterfactual Scenarios (CSDN)
with tabs[6]:
    st.subheader("Counterfactual Sediment Dynamics Network (CSDN)")
    st.markdown("Evaluate spatial sediment risk response under controlled hydrological and meteorological interventions:")

    SCENARIO_OPTIONS = [
        ("Rainfall +10%", "rainfall_plus10", 0.10, "Rainfall"),
        ("Rainfall +20%", "rainfall_plus20", 0.20, "Rainfall"),
        ("Rainfall +30%", "rainfall_plus30", 0.30, "Rainfall"),
        ("River Flow +10%", "flow_plus10", 0.10, "River Flow"),
        ("River Flow -10%", "flow_minus10", -0.10, "River Flow"),
        ("River Flow -20%", "flow_minus20", -0.20, "River Flow"),
        ("Reservoir Inflow +20%", "inflow_plus20", 0.20, "Reservoir Inflow"),
        ("Reservoir Inflow -20%", "inflow_minus20", -0.20, "Reservoir Inflow"),
    ]

    selected_label = st.selectbox(
        "Select Intervention Scenario",
        [opt[0] for opt in SCENARIO_OPTIONS],
        index=1,
    )
    sel_opt = next(opt for opt in SCENARIO_OPTIONS if opt[0] == selected_label)
    sc_name, sc_mag, sc_type = sel_opt[1], sel_opt[2], sel_opt[3]

    base_p = OUT / "counterfactual_baseline.npy"
    if not exists(base_p):
        base_p = OUT / "future_sediment_hotspot_heatmap.npy"

    cf_p = OUT / f"counterfactual_{sc_name}.npy"
    diff_p = DIFF_DIR / f"diff_{sc_name}.npy"
    sens_p = SENS_DIR / f"sens_{sc_name}.npy"

    base_map = load_array(base_p, "Baseline Heatmap")
    cf_map = load_array(cf_p, f"Counterfactual Map ({selected_label})")
    diff_map = load_array(diff_p, f"Difference Map ({selected_label})")
    sens_map = load_array(sens_p, f"Sensitivity Map ({selected_label})")

    if all(m is not None for m in [base_map, cf_map, diff_map, sens_map]):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            fig_b = plot_heatmap(
                base_map,
                "1. Baseline Heatmap H = f(X)",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                cbar_label="Risk Score [0 - 1]",
                contours=[0.5, 0.7],
                contour_colors=["#ffffff", "#ffd700"],
            )
            st.pyplot(fig_b)
            plt.close(fig_b)

        with c2:
            fig_cf = plot_heatmap(
                cf_map,
                f"2. Counterfactual H_cf ({selected_label})",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                cbar_label="Risk Score [0 - 1]",
                contours=[0.5, 0.7],
                contour_colors=["#ffffff", "#ffd700"],
            )
            st.pyplot(fig_cf)
            plt.close(fig_cf)

        with c3:
            diff_abs_max = max(0.02, float(max(abs(diff_map.min()), abs(diff_map.max()))))
            fig_d = plot_heatmap(
                diff_map,
                "3. Risk Change ΔH = H_cf − H",
                cmap="RdBu_r",
                vmin=-diff_abs_max,
                vmax=diff_abs_max,
                cbar_label="Δ Risk (Red: +Risk, Blue: -Risk)",
            )
            st.pyplot(fig_d)
            plt.close(fig_d)

        with c4:
            fig_s = plot_heatmap(
                sens_map,
                "4. Spatial Sensitivity |ΔH| / |Δx|",
                cmap="magma",
                vmin=0.0,
                vmax=float(sens_map.max()) if sens_map.max() > 0 else 1.0,
                cbar_label="Sensitivity |ΔH| / |Δx|",
            )
            st.pyplot(fig_s)
            plt.close(fig_s)

        st.markdown("#### Scenario Quantitative Diagnostics")
        sum1, sum2, sum3, sum4 = st.columns(4)
        with sum1:
            st.markdown("**Baseline State**")
            st.write(f"• Mean Risk: `{base_map.mean():.4f}`")
            st.write(f"• Peak Risk: `{base_map.max():.4f}`")
            st.write(f"• Area ≥ 0.5: `{(base_map >= 0.5).mean() * 100:.2f}%`")
        with sum2:
            st.markdown(f"**Counterfactual State ({selected_label})**")
            st.write(f"• Mean Risk: `{cf_map.mean():.4f}`")
            st.write(f"• Peak Risk: `{cf_map.max():.4f}`")
            st.write(f"• Area ≥ 0.5: `{(cf_map >= 0.5).mean() * 100:.2f}%`")
        with sum3:
            st.markdown("**Risk Change (ΔH)**")
            st.write(f"• Mean ΔH: `{diff_map.mean():+.4f}`")
            st.write(f"• Max Increase: `{diff_map.max():+.4f}`")
            st.write(f"• Max Decrease: `{diff_map.min():+.4f}`")
        with sum4:
            st.markdown("**Spatial Sensitivity**")
            st.write(f"• Mean Sensitivity: `{sens_map.mean():.4f}`")
            st.write(f"• Max Sensitivity: `{sens_map.max():.4f}`")
            st.write(f"• Intervention Mag |Δx|: `{abs(sc_mag):.2f}`")

        st.markdown("#### Zone Vulnerability & Sensitivity Ranking")
        num_zones = 8
        strip_h = base_map.shape[0] // num_zones
        zone_rows = []
        for i in range(num_zones):
            z_name = f"Zone {chr(65 + i)}"
            z_slice = slice(i * strip_h, (i + 1) * strip_h)
            b_risk = float(base_map[z_slice, :].mean())
            c_risk = float(cf_map[z_slice, :].mean())
            d_risk = float(diff_map[z_slice, :].mean())
            s_val = float(sens_map[z_slice, :].mean())
            zone_rows.append({
                "Zone": z_name,
                "Baseline Risk": round(b_risk, 4),
                "Counterfactual Risk": round(c_risk, 4),
                "Risk Change (ΔH)": round(d_risk, 4),
                "Mean Sensitivity": round(s_val, 4),
            })
        z_ranking_df = pd.DataFrame(zone_rows).sort_values("Mean Sensitivity", ascending=False).reset_index(drop=True)
        z_ranking_df["Rank"] = z_ranking_df.index + 1
        q75 = z_ranking_df["Mean Sensitivity"].quantile(0.75)
        q25 = z_ranking_df["Mean Sensitivity"].quantile(0.25)
        z_ranking_df["Vulnerability Category"] = z_ranking_df["Mean Sensitivity"].apply(
            lambda s: "Highly Sensitive" if s >= q75 else ("Moderately Sensitive" if s >= q25 else "Stable")
        )
        st.dataframe(
            z_ranking_df[["Rank", "Zone", "Baseline Risk", "Counterfactual Risk", "Risk Change (ΔH)", "Mean Sensitivity", "Vulnerability Category"]],
            use_container_width=True,
        )

    cf_res_p = OUT / "counterfactual_results.csv"
    if exists(cf_res_p):
        st.markdown("#### Multi-Scenario Systematic Benchmark")
        st.dataframe(pd.read_csv(cf_res_p), use_container_width=True)


# Tab 8: Multi-Reservoir Zone Analysis
with tabs[7]:
    st.subheader("Multi-Reservoir Zone-Wise Sedimentation Analysis")
    st.caption("Zone Partitioning • Spatial Hotspot Risk • WR-GNN Topological Connectivity • Epistemic Uncertainty • CSDN Scenarios for Krishna, Narmada & Kaveri Basins")

    from sediment_hotspot.reservoirs.catalog import RESERVOIR_CATALOG, list_primary_reservoirs, get_reservoir_spec
    from sediment_hotspot.reservoirs.pipeline import run_single_reservoir_pipeline, run_all_multi_reservoir_pipelines

    MULTI_OUT = REPO_ROOT / "data/outputs/multi_reservoir"

    RESERVOIR_OPTIONS = [
        ("Nagarjuna Sagar Reservoir (Krishna Basin) [Prioritized]", "krishna_nagarjuna_sagar"),
        ("Omkareshwar Reservoir (Narmada Basin) [Prioritized]", "narmada_omkareshwar"),
        ("Mettur Dam - Stanley Reservoir (Kaveri Basin) [Prioritized]", "kaveri_mettur"),
        ("Krishna Raja Sagara - KRS (Kaveri Basin)", "kaveri_krs"),
        ("Almatti Reservoir (Krishna Basin)", "krishna_almatti"),
    ]

    top_c1, top_c2, top_c3 = st.columns([3, 1.2, 1.2])
    with top_c1:
        selected_res_label = st.selectbox(
            "Select Reservoir for Zone-Wise Analysis",
            [opt[0] for opt in RESERVOIR_OPTIONS],
            index=0,
        )
        selected_res_id = next(opt[1] for opt in RESERVOIR_OPTIONS if opt[0] == selected_res_label)

    with top_c2:
        st.write("")
        if st.button("Run Live Analysis", use_container_width=True, help="Re-runs zone pipeline and MC Dropout inference for selected reservoir"):
            with st.spinner(f"Running zone-wise prediction pipeline for {selected_res_label}..."):
                try:
                    run_single_reservoir_pipeline(selected_res_id, model_path="data/models/sediment_fusion_model.pt", output_base_dir="data/outputs/multi_reservoir", mc_samples=15)
                    st.success("Successfully generated zone-wise predictions!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error running pipeline: {e}")

    with top_c3:
        st.write("")
        if st.button("Benchmark All 3 Basins", use_container_width=True, help="Executes multi-reservoir analysis across Krishna, Narmada, and Kaveri basins"):
            with st.spinner("Executing multi-reservoir analysis across all basins..."):
                try:
                    run_all_multi_reservoir_pipelines(model_path="data/models/sediment_fusion_model.pt", output_base_dir="data/outputs/multi_reservoir", mc_samples=15)
                    st.success("Completed cross-basin benchmark!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error executing benchmark: {e}")

    res_spec = get_reservoir_spec(selected_res_id)
    res_dir = MULTI_OUT / selected_res_id

    # Check if artifacts exist; if not, automatically run or advise
    if not res_dir.exists() or not (res_dir / "future_sediment_hotspot_heatmap.npy").exists():
        st.info(f"Artifacts for **{res_spec.name}** have not been generated yet. Running one-time analysis...")
        with st.spinner(f"Initializing {res_spec.name} spatial grid, zones, and predictions..."):
            run_single_reservoir_pipeline(selected_res_id, model_path="data/models/sediment_fusion_model.pt", output_base_dir="data/outputs/multi_reservoir", mc_samples=15)
        st.rerun()

    # --- 1. Reservoir Metadata Header ---
    meta_path = res_dir / "prediction_metadata.json"
    meta_dict = {}
    if meta_path.exists():
        import json
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

    w_info = meta_dict.get("weather_data", {})
    stats = meta_dict.get("summary_statistics", {})

    st.markdown("---")
    hdr_c1, hdr_c2, hdr_c3, hdr_c4, hdr_c5 = st.columns(5)
    hdr_c1.metric("River Basin", res_spec.basin, help=f"River: {res_spec.river}")
    hdr_c2.metric("Dam Coordinates", f"{res_spec.dam_coordinates[0]:.4f}°N, {res_spec.dam_coordinates[1]:.4f}°E", help="CWC / WRIS Station Coordinates")
    hdr_c3.metric("Gross Storage", f"{res_spec.gross_storage_bcm:.3f} BCM", help=f"Full Reservoir Level: {res_spec.full_reservoir_level_m} m")
    hdr_c4.metric("Monitored Zones", f"{len(res_spec.zones)} Spatial Zones", help="Hydrologically partitioned reach zones")
    hdr_c5.metric("Live Rainfall", f"{w_info.get('current_rainfall', 0.0):.1f} mm", delta=f"1h Fcst: {w_info.get('predicted_rainfall_next_hour', 0.0):.2f} mm")

    st.markdown("---")

    # Sub-tabs for clean organization
    m_subtabs = st.tabs([
        "Zone Boundaries & Hotspots",
        "Zone-Wise Risk & DSDI",
        "Uncertainty & Connectivity",
        "Counterfactual Scenarios (CSDN)",
        "Cross-Basin Reservoir Comparison",
        "Data Provenance & Validation",
    ])

    # Sub-tab 1: Zone Boundaries & Hotspots
    with m_subtabs[0]:
        st.subheader("1. Reservoir Location, Zone Boundaries & Predicted Sediment Hotspots")
        c_map1, c_map2 = st.columns(2)

        zone_png = res_dir / "reservoir_zone_map.png"
        hotspot_npy = res_dir / "future_sediment_hotspot_heatmap.npy"
        hotspot_png = res_dir / "hotspot_heatmap.png"

        with c_map1:
            st.markdown("#### Hydrological Zone Partitioning Map")
            if zone_png.exists():
                st.image(str(zone_png), use_container_width=True)
            else:
                st.info("Zone map image generating...")

        with c_map2:
            st.markdown("#### Predicted Sediment Hotspots Heatmap f(X)")
            if hotspot_npy.exists():
                h_data = np.load(hotspot_npy)
                fig_h = plot_heatmap(
                    h_data,
                    f"{res_spec.name}: Sediment Accumulation Hotspots",
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    cbar_label="Risk Probability [0 - 1]",
                    contours=[0.5, 0.7],
                    contour_colors=["#ffffff", "#ffd700"],
                )
                st.pyplot(fig_h)
                plt.close(fig_h)
            elif hotspot_png.exists():
                st.image(str(hotspot_png), use_container_width=True)

        zone_meta_csv = res_dir / "reservoir_zone_map.csv"
        if zone_meta_csv.exists():
            st.markdown("#### Zone Geographic Specifications & Morphology Table")
            z_mdf = pd.read_csv(zone_meta_csv)
            st.dataframe(
                z_mdf[["zone_id", "zone_name", "zone_type", "depth_category", "water_area_sqkm", "centroid_lat", "centroid_lon", "description"]],
                use_container_width=True,
            )

    # Sub-tab 2: Zone Risk & DSDI
    with m_subtabs[1]:
        st.subheader("2. Zone-Wise Sediment Risk & Dynamic Sediment Deposition Index (DSDI)")
        dsdi_npy = res_dir / "dsdi_heatmap.npy"
        risk_csv = res_dir / "zone_sediment_risk.csv"
        dsdi_csv = res_dir / "zone_dsdi_scores.csv"

        z_c1, z_c2 = st.columns(2)
        with z_c1:
            if dsdi_npy.exists():
                dsdi_arr = np.load(dsdi_npy)
                fig_dsdi = plot_heatmap(
                    dsdi_arr,
                    f"{res_spec.name}: Dynamic Sediment Deposition Index (DSDI)",
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    cbar_label="DSDI Gated Prior [0 - 1]",
                    contours=[0.5],
                    contour_colors=["#ffffff"],
                )
                st.pyplot(fig_dsdi)
                plt.close(fig_dsdi)

        with z_c2:
            if risk_csv.exists():
                r_df = pd.read_csv(risk_csv)
                st.markdown("#### Mean Predicted Risk by Reservoir Zone")
                chart_df = r_df.set_index("zone_id")[["mean_risk", "peak_risk"]]
                chart_df.columns = ["Mean Risk [0-1]", "Peak Risk [0-1]"]
                st.bar_chart(chart_df, height=340)

        if risk_csv.exists() and dsdi_csv.exists():
            r_df = pd.read_csv(risk_csv)
            d_df = pd.read_csv(dsdi_csv)
            merged_zone = pd.merge(r_df, d_df[["zone_id", "dsdi_score", "erosion_forcing_category"]], on="zone_id")
            st.markdown("#### Detailed Zone Risk & Forcing Statistics")
            st.dataframe(
                merged_zone[[
                    "zone_id", "zone_name", "zone_type", "mean_risk", "peak_risk",
                    "high_risk_cells_pct", "risk_category", "dsdi_score", "erosion_forcing_category"
                ]],
                use_container_width=True,
            )

    # Sub-tab 3: Uncertainty & Connectivity
    with m_subtabs[2]:
        st.subheader("3. Epistemic Uncertainty & WR-GNN Upstream-to-Downstream Connectivity")
        unc_npy = res_dir / "uncertainty_heatmap.npy"
        unc_csv = res_dir / "zone_uncertainty.csv"
        conn_png = res_dir / "zone_connectivity_graph.png"
        conn_json = res_dir / "zone_connectivity_graph.json"

        u_c1, u_c2 = st.columns(2)
        with u_c1:
            st.markdown("#### MC Dropout Epistemic Variance (σ²)")
            if unc_npy.exists():
                u_arr = np.load(unc_npy)
                fig_u = plot_heatmap(
                    u_arr,
                    "Predictive Variance σ² (Epistemic Uncertainty)",
                    cmap="magma",
                    cbar_label="MC Dropout Variance σ²",
                )
                st.pyplot(fig_u)
                plt.close(fig_u)

        with u_c2:
            st.markdown("#### Watershed-to-Reservoir Connectivity Graph (WR-GNN)")
            if conn_png.exists():
                st.image(str(conn_png), use_container_width=True)
            else:
                st.info("Connectivity graph rendering...")

        if unc_csv.exists():
            st.markdown("#### Zone Decision Triage & Priority Monitoring Status")
            st.dataframe(pd.read_csv(unc_csv), use_container_width=True)

        if conn_json.exists():
            with open(conn_json, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            st.markdown("#### Topological Advection & Sediment Attenuation Records")
            st.dataframe(pd.DataFrame(c_data.get("propagation_records", [])), use_container_width=True)
            st.info(f"**Topological Analysis Insight**: {c_data.get('scientific_interpretation', '')}")

    # Sub-tab 4: Counterfactual Scenarios (CSDN)
    with m_subtabs[3]:
        st.subheader("4. CSDN Counterfactual Environmental Intervention Scenarios")
        st.markdown("Evaluate spatial risk shifts under controlled rainfall, river flow, and inflow changes:")

        CF_OPTS = [
            ("Baseline Reference", "baseline", 0.0),
            ("Rainfall +10%", "rainfall_plus10", 0.10),
            ("Rainfall +20%", "rainfall_plus20", 0.20),
            ("Rainfall +30%", "rainfall_plus30", 0.30),
            ("River Flow +10%", "flow_plus10", 0.10),
            ("River Flow -10%", "flow_minus10", -0.10),
            ("River Flow -20%", "flow_minus20", -0.20),
            ("Reservoir Inflow +20%", "inflow_plus20", 0.20),
            ("Reservoir Inflow -20%", "inflow_minus20", -0.20),
        ]

        sc_label = st.selectbox("Select Counterfactual Intervention", [o[0] for o in CF_OPTS], index=2)
        sel_sc = next(o for o in CF_OPTS if o[0] == sc_label)
        sc_file_key, sc_mag = sel_sc[1], sel_sc[2]

        b_map_p = res_dir / "counterfactual_baseline.npy"
        cf_map_p = res_dir / f"counterfactual_{sc_file_key}.npy"
        diff_map_p = res_dir / f"counterfactual_difference_maps/diff_{sc_file_key}.npy"
        sens_map_p = res_dir / f"sensitivity_maps/sens_{sc_file_key}.npy"

        if all(p.exists() for p in [b_map_p, cf_map_p, diff_map_p, sens_map_p]):
            b_m = np.load(b_map_p)
            cf_m = np.load(cf_map_p)
            d_m = np.load(diff_map_p)
            s_m = np.load(sens_map_p)

            p1, p2, p3, p4 = st.columns(4)
            with p1:
                f1 = plot_heatmap(b_m, "1. Baseline Risk H", cmap="viridis", vmin=0.0, vmax=1.0, cbar_label="Risk [0-1]")
                st.pyplot(f1)
                plt.close(f1)
            with p2:
                f2 = plot_heatmap(cf_m, f"2. Counterfactual H_cf ({sc_label})", cmap="viridis", vmin=0.0, vmax=1.0, cbar_label="Risk [0-1]")
                st.pyplot(f2)
                plt.close(f2)
            with p3:
                lim_d = max(0.02, float(max(abs(d_m.min()), abs(d_m.max()))))
                f3 = plot_heatmap(d_m, "3. Risk Shift ΔH = H_cf − H", cmap="RdBu_r", vmin=-lim_d, vmax=lim_d, cbar_label="Δ Risk (+Red / -Blue)")
                st.pyplot(f3)
                plt.close(f3)
            with p4:
                lim_s = max(0.05, float(s_m.max()))
                f4 = plot_heatmap(s_m, "4. Spatial Sensitivity |ΔH| / |Δx|", cmap="magma", vmin=0.0, vmax=lim_s, cbar_label="Sensitivity")
                st.pyplot(f4)
                plt.close(f4)

            # Metrics
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("Mean Shift ΔH", f"{d_m.mean():+.4f}")
            m_c2.metric("Max Shift ΔH", f"{d_m.max():+.4f}")
            m_c3.metric("High-Risk Area (≥0.5)", f"{(cf_m >= 0.5).mean() * 100:.1f}%", delta=f"{((cf_m >= 0.5).mean() - (b_m >= 0.5).mean()) * 100:+.1f}%")
            m_c4.metric("Mean Sensitivity", f"{s_m.mean():.4f}")

        # Zone vulnerability table
        z_sens_p = res_dir / "zone_sensitivity.csv"
        if z_sens_p.exists():
            st.markdown("#### Zone Vulnerability & Sensitivity Ranking (+20% Rainfall Benchmark)")
            st.dataframe(pd.read_csv(z_sens_p), use_container_width=True)

        cf_all_p = res_dir / "counterfactual_results.csv"
        if cf_all_p.exists():
            st.markdown("#### Multi-Scenario Systematic Benchmark Table")
            st.dataframe(pd.read_csv(cf_all_p), use_container_width=True)

    # Sub-tab 5: Cross-Basin Reservoir Comparison
    with m_subtabs[4]:
        st.subheader("5. Cross-Basin Multi-Reservoir Comparative Benchmark")
        st.caption("Comparison across Krishna (Nagarjuna Sagar), Narmada (Omkareshwar), and Kaveri (Mettur Dam)")

        comp_csv = MULTI_OUT / "reservoir_comparison.csv"
        if comp_csv.exists():
            cmp_df = pd.read_csv(comp_csv)
            st.dataframe(cmp_df, use_container_width=True)

            ch_c1, ch_c2 = st.columns(2)
            with ch_c1:
                st.markdown("#### Mean vs Peak Hotspot Risk Comparison")
                st.bar_chart(cmp_df.set_index("Reservoir Name")[["Mean Hotspot Risk", "Peak Hotspot Risk"]])

            with ch_c2:
                st.markdown("#### Mean Dynamic Sediment Deposition Index (DSDI)")
                st.bar_chart(cmp_df.set_index("Reservoir Name")["Mean DSDI Score"])
        else:
            st.info("Run the cross-basin benchmark to view comparison table.")

    # Sub-tab 6: Provenance & Validation
    with m_subtabs[5]:
        st.subheader("6. Evaluation, Data Provenance & Scientific Disclaimers")
        eval_p = res_dir / "evaluation_metrics.json"
        if eval_p.exists():
            with open(eval_p, "r", encoding="utf-8") as f:
                ev_data = json.load(f)

            st.markdown("#### Quantitative Model Evaluation against Remote-Sensing Derived Reference Labels")
            ev_c1, ev_c2, ev_c3, ev_c4, ev_c5 = st.columns(5)
            ev_c1.metric("Detection IoU", f"{ev_data.get('iou', 0.0):.4f}")
            ev_c2.metric("Dice Score", f"{ev_data.get('dice', 0.0):.4f}")
            ev_c3.metric("Precision", f"{ev_data.get('precision', 0.0):.4f}")
            ev_c4.metric("Recall / Sensitivity", f"{ev_data.get('recall', 0.0):.4f}")
            ev_c5.metric("ROC-AUC", f"{ev_data.get('auc', 0.5):.4f}")

            st.json(ev_data)

        st.markdown("""
        ### Label Provenance & Methodological Limitations
        - **Label Provenance**: All sediment hotspot masks are derived from optical and multispectral remote-sensing proxies (Sentinel-2 NDTI, turbid plume thresholds, and SRTM DEM erosion susceptibility). They are explicitly designated as `DERIVED_FROM_SATELLITE_INDEX`.
        - **Absence of Bathymetric Surveys**: Because periodic hydrographic/bathymetric echo-sounding surveys are not publicly published in real time for these reservoirs, **predicted values represent relative sediment accumulation risk probabilities $[0, 1]$**. They must **NOT** be interpreted as verified physical sediment volumes (million cubic meters) or bed elevation deposition thickness (meters).
        - **Spatial Grid Alignment**: Each reservoir grid is georeferenced to WGS84 (`EPSG:4326`) with an approximate cell resolution tailored to the reservoir's surface geometry.
        """)


# Tab 9: Model Benchmark & Evaluation
with tabs[8]:
    st.subheader("9. Comparative Model Benchmarks & Empirical Evaluation")
    st.caption("Rigorous performance evaluation across classical baselines, deep learning architectures, and ablation configurations")

    eval_subtabs = st.tabs([
        "Baseline Model Comparison",
        "Ablation Analysis",
        "Evaluation Figures & Publication Artifacts",
        "Statistical Metrics Summary",
    ])

    results_dir = REPO_ROOT / "results"
    paper_dir = results_dir / "paper"

    with eval_subtabs[0]:
        st.markdown("### Algorithm Performance Benchmark")
        st.markdown("Empirical comparison of the proposed multimodal architecture against classical and single-modality baseline algorithms evaluated on the identical test split.")

        baseline_csv = results_dir / "baseline_results.csv"
        if not baseline_csv.exists() and (paper_dir / "table2_baseline_comparison.csv").exists():
            baseline_csv = paper_dir / "table2_baseline_comparison.csv"

        if baseline_csv.exists():
            b_df = pd.read_csv(baseline_csv)
            prop_row = b_df[b_df["model"].str.contains("Proposed|Full Model", case=False, na=False)]
            if not prop_row.empty:
                best_m = prop_row.iloc[0]
                m_c1, m_c2, m_c3, m_c4, m_c5, m_c6 = st.columns(6)
                m_c1.metric("Peak Test IoU", f"{best_m.get('iou', 0.0)*100:.2f}%")
                m_c2.metric("Dice / F1 Score", f"{best_m.get('dice', 0.0)*100:.2f}%")
                m_c3.metric("Precision", f"{best_m.get('precision', 0.0)*100:.2f}%")
                m_c4.metric("Recall / Sensitivity", f"{best_m.get('recall', 0.0)*100:.2f}%")
                m_c5.metric("ROC-AUC", f"{best_m.get('auc', 0.0):.4f}")
                m_c6.metric("PR-AUC", f"{best_m.get('pr_auc', 0.0):.4f}")

            st.markdown("#### Comparative Benchmark Table")
            display_cols = [c for c in ["model", "iou", "dice", "precision", "recall", "auc", "pr_auc", "false_hotspot_rate", "zone_mae", "centroid_displacement"] if c in b_df.columns]
            styled_df = b_df[display_cols].copy()
            if "iou" in styled_df:
                styled_df["iou"] = styled_df["iou"].map(lambda x: f"{x*100:.2f}%")
            if "dice" in styled_df:
                styled_df["dice"] = styled_df["dice"].map(lambda x: f"{x*100:.2f}%")
            if "precision" in styled_df:
                styled_df["precision"] = styled_df["precision"].map(lambda x: f"{x*100:.2f}%")
            if "recall" in styled_df:
                styled_df["recall"] = styled_df["recall"].map(lambda x: f"{x*100:.2f}%")
            if "false_hotspot_rate" in styled_df:
                styled_df["false_hotspot_rate"] = styled_df["false_hotspot_rate"].map(lambda x: f"{x:.5f}")
            if "zone_mae" in styled_df:
                styled_df["zone_mae"] = styled_df["zone_mae"].map(lambda x: f"{x:.4f}")
            if "centroid_displacement" in styled_df:
                styled_df["centroid_displacement"] = styled_df["centroid_displacement"].map(lambda x: f"{x:.2f} px" if pd.notna(x) else "N/A")

            st.dataframe(styled_df, use_container_width=True)

            st.markdown("#### Comparative Visualizations")
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown("**Intersection over Union (IoU) Comparison (%)**")
                chart_iou = b_df.set_index("model")["iou"] * 100
                st.bar_chart(chart_iou)
            with ch2:
                st.markdown("**Area Under Precision-Recall Curve (PR-AUC)**")
                if "pr_auc" in b_df.columns:
                    chart_pr = b_df.set_index("model")["pr_auc"]
                    st.bar_chart(chart_pr)
        else:
            st.warning("Baseline results not found at results/baseline_results.csv")

    with eval_subtabs[1]:
        st.markdown("### Ablation Study Analysis")
        st.markdown("Quantitative evaluation examining the impact of individual architectural components, loss terms, and physics constraints.")

        ablation_csv = results_dir / "ablation_results.csv"
        if not ablation_csv.exists() and (paper_dir / "table3_ablation_study.csv").exists():
            ablation_csv = paper_dir / "table3_ablation_study.csv"

        if ablation_csv.exists():
            ab_df = pd.read_csv(ablation_csv)
            st.dataframe(ab_df, use_container_width=True)

            ab_c1, ab_c2 = st.columns(2)
            with ab_c1:
                st.markdown("**IoU (%) by Ablated Component**")
                if "ablation_stage" in ab_df.columns and "iou" in ab_df.columns:
                    st.bar_chart(ab_df.set_index("ablation_stage")["iou"] * 100)
            with ab_c2:
                st.markdown("**Centroid Displacement Error (Pixels)**")
                if "ablation_stage" in ab_df.columns and "centroid_displacement" in ab_df.columns:
                    st.bar_chart(ab_df.set_index("ablation_stage")["centroid_displacement"])
        else:
            st.warning("Ablation results not found at results/ablation_results.csv")

    with eval_subtabs[2]:
        st.markdown("### Research Figures & Visual Diagnostic Panels")
        st.caption("High-resolution diagnostic visualizations generated during empirical benchmarking")

        fig_cols = st.columns(2)
        figs = [
            ("fig3_predicted_hotspots.png", "Spatial Sediment Hotspot Prediction Map"),
            ("fig2_dsdi_heatmap.png", "Dynamic Sediment Delivery Index (DSDI) Heatmap"),
            ("fig7_uncertainty_map.png", "Epistemic Uncertainty via Monte Carlo Dropout"),
            ("fig5_counterfactual_diff_rainfall20.png", "Counterfactual Response Map (+20% Rainfall Shift)"),
            ("fig6_spatial_sensitivity.png", "Spatial Parameter Sensitivity Distribution"),
        ]

        for i, (fname, label) in enumerate(figs):
            fpath = paper_dir / fname
            col = fig_cols[i % 2]
            with col:
                if fpath.exists():
                    st.markdown(f"**{label}** (`{fname}`)")
                    st.image(str(fpath), use_container_width=True)
                else:
                    alt_path = OUT / fname
                    if alt_path.exists():
                        st.markdown(f"**{label}**")
                        st.image(str(alt_path), use_container_width=True)

    with eval_subtabs[3]:
        st.markdown("### Complete Experimental Artifacts & LaTeX Tables")
        st.markdown("Pre-compiled LaTeX tables formatted for academic publication:")

        tex_files = list(paper_dir.glob("*.tex")) if paper_dir.exists() else []
        if tex_files:
            for tfile in sorted(tex_files):
                with st.expander(f"LaTeX Source: {tfile.name}"):
                    with open(tfile, "r", encoding="utf-8") as f:
                        st.code(f.read(), language="latex")
        else:
            st.info("No LaTeX files found in results/paper/")

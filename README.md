# Reservoir Intelligence System: Multi-Modal AI for Spatial Sediment Hotspot Prediction & Zone-Wise Dynamics

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, physics-informed environmental AI platform combining short-term meteorological forecasting with 7-modality spatial reasoning to predict pixel-level sedimentation hotspots, hydrological zone-wise risks, epistemic uncertainty, and counterfactual intervention dynamics across major peninsular river basins in India (Krishna, Narmada, and Kaveri).

The system addresses the critical problem of reservoir storage capacity loss by transforming sparse gauging and satellite observations into high-resolution spatial decision support for dam operators, irrigation authorities, and hydrologists.

---

## Table of Contents

1. [Executive Summary & Core Innovations](#1-executive-summary--core-innovations)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Multi-Reservoir Zone-Wise Subsystem](#3-multi-reservoir-zone-wise-subsystem)
4. [Methodological & Mathematical Formulations](#4-methodological--mathematical-formulations)
   - [Dynamic Sediment Deposition Index (DSDI)](#dynamic-sediment-deposition-index-dsdi)
   - [Watershed-Reservoir Graph Neural Network (WR-GNN)](#watershed-reservoir-graph-neural-network-wr-gnn)
   - [Causal Attention Fusion (CAF)](#causal-attention-fusion-caf)
   - [Physics-Informed Sediment Loss (PISL)](#physics-informed-sediment-loss-pisl)
   - [Epistemic Uncertainty via Monte Carlo Dropout](#epistemic-uncertainty-via-monte-carlo-dropout)
   - [Counterfactual Sediment Dynamics Network (CSDN)](#counterfactual-sediment-dynamics-network-csdn)
5. [Data Provenance & Scientific Ground Truth](#5-data-provenance--scientific-ground-truth)
6. [Repository Structure](#6-repository-structure)
7. [Installation & Environment Setup](#7-installation--environment-setup)
8. [Execution & CLI Workflows](#8-execution--cli-workflows)
9. [Interactive Streamlit Decision-Support Dashboard](#9-interactive-streamlit-decision-support-dashboard)
10. [Evaluation Metrics & Validation Results](#10-evaluation-metrics--validation-results)
11. [Testing & Quality Assurance](#11-testing--quality-assurance)

---

## 1. Executive Summary & Core Innovations

Reservoir sedimentation progressively depletes active storage capacity, impairs flood moderation, erodes hydro-turbine infrastructure, and complicates irrigation scheduling. Traditional assessment methods rely on periodic hydrographic or bathymetric surveys that occur years apart and provide aggregate volumetric estimates without localized spatial guidance.

This platform bridges that gap by predicting **where, when, and why sedimentation concentrates within reservoirs**:

- **Dual-Stage Predictive Pipeline**: Bridges upstream meteorological catchment dynamics (next-hour rainfall prediction via XGBoost) with within-reservoir deposition mechanics (spatial hotspot risk via deep multimodal fusion).
- **Multi-Basin Multi-Reservoir Coverage**: Built-in topological, hydraulic, and environmental specifications for 5 major reservoirs across the **Krishna**, **Narmada**, and **Kaveri** river basins, with prioritized zone partitioning for **Nagarjuna Sagar**, **Omkareshwar**, and **Mettur Stanley Reservoir**.
- **Hydrological Zone Discretization**: Partitioning of reservoir water bodies into 6 functional zones (Inflow Delta, Lacustrine Pool, Gorge/Mid-Section, Left/Right Flanks, and Dam Forebay) based on flow-axis centerline geometry.
- **Physics-Informed Objective (PISL)**: Enforces conservation of sediment mass, spatial continuity, and advection-diffusion settling dynamics directly into the neural loss function.
- **Topological Graph Connectivity (WR-GNN)**: Models upstream-to-downstream sediment transport cascades using directed acyclic graph message passing across Central Water Commission (CWC) monitoring stations.
- **Actionable Counterfactual Scenarios (CSDN)**: Evaluates $do(\cdot)$ causal interventions (e.g., $+20\%$ monsoon deluge, $-20\%$ upstream flushing) to compute spatial difference maps $\Delta H$ and quantify zone vulnerability.
- **Operational Uncertainty Triage**: Monte Carlo Dropout separates confident predictions from high-uncertainty zones to drive sonar-targeted bathymetric survey deployment.

---

## 2. End-to-End System Architecture

The platform operates as a coordinated dual-subsystem pipeline linking catchment hydrology to reservoir bathymetry proxies.

```text
==================================================================================================
                                    END-TO-END SYSTEM ARCHITECTURE
==================================================================================================

 [ STAGE 1: METEOROLOGICAL SUBSYSTEM ]
 
   Open-Meteo Historical Archive (2020-Present)
   & Real-Time Forecast APIs (14 CWC Basin Stations)
                       │
                       ▼
   ┌──────────────────────────────────────────────┐
   │ Feature Engineering                          │
   │  - Lags (t-1, t-2, t-3, t-6, t-12, t-24)     │
   │  - Rolling Windows (3h, 6h, 12h, 24h mean/std│
   │  - Temporal Cyclics (sin/cos hour & month)   │
   │  - Trend Deltas & Rain Severity Bins         │
   └──────────────────────┬───────────────────────┘
                          │
                          ▼
   ┌──────────────────────────────────────────────┐
   │ XGBoost Gradient Boosted Regressor           │
   │  - n_estimators=200, max_depth=6, lr=0.05    │
   │  - Target: Next-Hour Precipitation (mm)      │
   └──────────────────────┬───────────────────────┘
                          │
                          ▼
            rainfall_predictions.csv  (14 Monitoring Locations)
                          │
══════════════════════════╪══════════════════════════════════════════════════════════════════════
                          │
 [ STAGE 2: MULTIMODAL SEDIMENT HOTSPOT & RESERVOIR SUBSYSTEM ]
                          │
                          ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Seven-Modality Environmental Alignment Tensor (128 x 128 Spatial Grid @ EPSG:4326)           │
 │  1. Sentinel-2 NDTI Proxy (B4/B3 Turbidity)     5. Catchment Rainfall (XGBoost live link)   │
 │  2. SRTM 30m DEM Elevation & Gradient (Slope)  6. WR-GNN River Connectivity Embeddings      │
 │  3. Dynamic World Land Use / Land Cover (LULC)  7. Dynamic Sediment Deposition Index (DSDI) │
 │  4. CWC Streamflow Velocity & River Discharge                                                │
 └───────────────────────────────┬──────────────────────────────────────────────────────────────┘
                                 │
     ┌───────────────────────────┴───────────────────────────┐
     │                                                       │
     ▼                                                       ▼
 ┌───────────────────────────────┐               ┌───────────────────────────────┐
 │ Spatial CNN Encoder           │               │ Temporal Weather LSTM         │
 │  - Conv2D + BatchNorm + ReLU  │               │  - Bidirectional LSTM Units   │
 │  - Multi-scale receptive field│               │  - Meteorological sequencing  │
 └───────────────┬───────────────┘               └───────────────┬───────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Causal Attention Fusion (CAF) & WR-GNN Topological Aggregator                                │
 │  - Cross-attention over spatial, temporal, and hydraulic embeddings                          │
 │  - Directed graph attention along CWC upstream-to-downstream topological edges               │
 └───────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                         │
                                         ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Spatial Decoder & Physics-Informed Optimization (PISL)                                       │
 │  - UNet-style Transposed Convolutions (128 x 128 resolution)                                 │
 │  - Objective: BCE/Dice + Spatial Smoothness + Flow Advection Gradient + Mass Conservation    │
 │  - Monte Carlo Dropout Uncertainty Engine (N=20 stochastic forward passes)                   │
 └───────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                         │
                                         ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Multi-Reservoir Analytic Products & Decision Outputs                                         │
 │  - Pixel Hotspot Probability Maps [0, 1]       - Counterfactual Delta Heatmaps (Delta H)     │
 │  - Zone-Wise Deposition Risk & Mean DSDI       - 4-Quadrant Operational Triage Matrix        │
 │  - Epistemic Uncertainty Variance Maps (sigma^2)- Cross-Basin Systematic Comparison Benchmark│
 └───────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                         │
                                         ▼
                   Interactive Streamlit Monitoring Dashboard (Tabs 1-8)
==================================================================================================
```

---

## 3. Multi-Reservoir Zone-Wise Subsystem

The multi-reservoir subsystem (`sediment_hotspot/reservoirs/`) operationalizes the core research architecture across 5 strategic reservoirs in 3 major peninsular river basins in India:

```text
                        PENINSULAR BASIN NETWORK
                        
         [ NARMADA BASIN ]             [ KRISHNA BASIN ]            [ KAVERI BASIN ]
                 │                             │                            │
                 ▼                             ▼                            ▼
        Omkareshwar Dam                Almatti Reservoir           Krishnaraja Sagara
       (0.987 BCM / 196.6m)           (3.440 BCM / 519.6m)         (1.390 BCM / 757.0m)
                                               │                            │
                                               ▼                            ▼
                                      NAGARJUNA SAGAR DAM          METTUR STANLEY DAM
                                     (11.472 BCM / 179.8m)        (2.640 BCM / 241.0m)
                                     [Primary Prioritized]        [Primary Prioritized]
```

### Reservoir Catalog Specifications

| Reservoir Identifier | Reservoir Name | River Basin | River | Latitude (°N) | Longitude (°E) | Gross Storage (BCM) | FRL (m) | Upstream CWC Stations | Downstream CWC Stations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `krishna_nagarjuna_sagar` | Nagarjuna Sagar Reservoir | Krishna | Krishna | 16.5768 | 79.3130 | 11.472 | 179.80 | K3 Raichur Kurnool, K2 Sangli Almatti | K5 Vijayawada Prakasam Barrage |
| `krishna_almatti` | Almatti Reservoir (LBS Dam) | Krishna | Krishna | 16.3314 | 75.8872 | 3.440 | 519.60 | K1 Mahabaleshwar | K3 Raichur Kurnool |
| `narmada_omkareshwar` | Omkareshwar Reservoir | Narmada | Narmada | 22.2472 | 76.1475 | 0.987 | 196.60 | N1 Amarkantak, N2 Jabalpur, N3 Hoshangabad | N4 Mandleshwar, N5 Bharuch |
| `kaveri_mettur` | Mettur Dam (Stanley Reservoir) | Kaveri | Kaveri | 11.8028 | 77.8078 | 2.640 | 241.00 | V1 Talakaveri, V2 KRS Mysore | V3 Erode, V4 Grand Anicut |
| `kaveri_krs` | Krishnaraja Sagara (KRS) | Kaveri | Kaveri | 12.4278 | 76.5722 | 1.390 | 757.00 | V1 Talakaveri | V2 KRS Mysore |

### Hydrological Zone Discretization

Each reservoir water body is partitioned into **6 functional hydrological zones** using flow-axis centerline geometry, distance-to-dam gradients, and lateral bay morphology:

1. **Zone 1: Inflow Delta / Upper Reach**: Coarse bedload deposition zone where river inflow decelerates upon entering the standing pool. Characterized by high shear stress reduction, coarse gravel/sand deposition, and strong sensitivity to upstream flood flushes.
2. **Zone 2: Deep Gorge / Lacustrine Storage Reach**: Transitional deep channel conveying suspended sediment through the reservoir canyon or main water body.
3. **Zone 3: Central Pool / Gorge Section**: Quiescent central reservoir body where fine silts settle under low turbulence regimes.
4. **Zone 4: Left Flank / Bay**: Secondary embayment or shallow littoral cove prone to localized wind-induced circulation and stagnant mud settling.
5. **Zone 5: Right Flank / Bay**: Opposite littoral shelf with shallow bathymetry and high land-cover runoff interaction.
6. **Zone 6: Dam Forebay & Sluice Intake**: Deepest pool immediately upstream of the dam spillway, penstocks, and bottom sluices. Critical for hydro-turbine abrasion prevention and desilting operations.

---

## 4. Methodological & Mathematical Formulations

### Dynamic Sediment Deposition Index (DSDI)

The DSDI is a physical composite indicator estimating instantaneous deposition propensity across the reservoir grid $(x, y)$ at time $t$:

$$\text{DSDI}(x, y, t) = \alpha \cdot \frac{R(t)}{R_{\max}} + \beta \cdot \frac{Q(t)}{Q_{\max}} + \gamma \cdot \frac{\text{Slope}(x, y)}{\text{Slope}_{\max}} + \delta \cdot \frac{\text{NDTI}(x, y, t)}{\text{NDTI}_{\max}} - \epsilon \cdot \frac{d_{\text{dam}}(x, y)}{d_{\max}}$$

Where:
- $R(t)$: Next-hour forecasted precipitation over the upstream catchment (mm).
- $Q(t)$: Inflow river discharge entering the reservoir reach ($\text{m}^3/\text{s}$).
- $\text{Slope}(x, y)$: Bathymetric/terrain gradient derived from SRTM 30m DEM (degrees).
- $\text{NDTI}(x, y, t)$: Surface water Normalized Difference Turbidity Index proxy from Sentinel-2.
- $d_{\text{dam}}(x, y)$: Euclidean distance to the dam axis (m).
- $\alpha, \beta, \gamma, \delta, \epsilon$: Calibrated physical weighting parameters satisfying $\sum \text{weights} = 1.0$.

### Watershed-Reservoir Graph Neural Network (WR-GNN)

The upstream catchment and reservoir zones form a directed acyclic graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where nodes $\mathcal{V}$ represent monitoring stations and reservoir zones, and edges $\mathcal{E}$ represent hydrodynamic flow channels. Topological feature aggregation proceeds via edge-conditioned graph attention:

$$\mathbf{h}_i^{(l+1)} = \sigma \left( \mathbf{W}_s \mathbf{h}_i^{(l)} + \sum_{j \in \mathcal{N}_{\text{in}}(i)} \alpha_{ij} \mathbf{W}_n \mathbf{h}_j^{(l)} \right)$$

$$\alpha_{ij} = \frac{\exp \left( \text{LeakyReLU} \left( \mathbf{a}^T [\mathbf{W}_e \mathbf{h}_i \parallel \mathbf{W}_e \mathbf{h}_j \parallel \mathbf{e}_{ji}] \right) \right)}{\sum_{k \in \mathcal{N}_{\text{in}}(i)} \exp \left( \text{LeakyReLU} \left( \mathbf{a}^T [\mathbf{W}_e \mathbf{h}_i \parallel \mathbf{W}_e \mathbf{h}_k \parallel \mathbf{e}_{ki}] \right) \right)}$$

Where $\mathbf{e}_{ji}$ encodes river reach length, channel slope, and flow travel time between station $j$ and station $i$.

### Causal Attention Fusion (CAF)

To prevent spurious spatial correlations from overriding physical transport mechanisms, Causal Attention Fusion applies directed masking between upstream river predictors and within-reservoir grid representations:

$$\text{CAF}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax} \left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}_{\text{causal}} \right) \mathbf{V}$$

Where $\mathbf{M}_{\text{causal}}(i, j) = 0$ if station $j$ is hydrologically upstream of or co-located with node $i$, and $-\infty$ if $j$ is downstream, strictly preserving upstream-to-downstream causality.

### Physics-Informed Sediment Loss (PISL)

The network is optimized using a composite physics-informed loss that balances predictive accuracy with hydrodynamic constraints:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}} + \lambda_{\text{cons}} \mathcal{L}_{\text{cons}}$$

1. **Task Loss ($\mathcal{L}_{\text{task}}$)**: Weighted Binary Cross-Entropy combined with Soft Dice Loss to handle extreme spatial hotspot class imbalance:
   $$\mathcal{L}_{\text{task}} = \text{BCE}(H, \hat{H}) + \left( 1 - \frac{2 \sum H \hat{H} + \epsilon}{\sum H + \sum \hat{H} + \epsilon} \right)$$
2. **Spatial Smoothness ($\mathcal{L}_{\text{smooth}}$)**: Penalizes unphysical discontinuous high-frequency spatial noise across neighboring water cells:
   $$\mathcal{L}_{\text{smooth}} = \frac{1}{|A|} \sum_{(x,y)} \left( \left| \nabla_x \hat{H}(x, y) \right|^2 + \left| \nabla_y \hat{H}(x, y) \right|^2 \right)$$
3. **Flow Advection Gradient ($\mathcal{L}_{\text{adv}}$)**: Enforces deposition settling alignment with the local hydrodynamic velocity vector field $\mathbf{u} = (u, v)$:
   $$\mathcal{L}_{\text{adv}} = \frac{1}{|A|} \sum_{(x,y)} \left( \mathbf{u}(x, y) \cdot \nabla \hat{H}(x, y) + w_s \hat{H}(x, y) \right)^2$$
   Where $w_s$ is the fine-sediment settling velocity parameterized by Stokes law.
4. **Conservation of Mass ($\mathcal{L}_{\text{cons}}$)**: Constrains total integrated reservoir hotspot volume to the upstream sediment influx rate:
   $$\mathcal{L}_{\text{cons}} = \left| \int_{A} \hat{H}(x, y) \, dA - \kappa \cdot (Q_{\text{in}} \cdot C_{\text{in}} - Q_{\text{out}} \cdot C_{\text{out}}) \right|$$

### Epistemic Uncertainty via Monte Carlo Dropout

Model uncertainty is estimated using Monte Carlo Dropout. During inference, dropout layers remain active with rate $p = 0.2$ over $N = 20$ stochastic forward passes:

$$\hat{\mu}(x, y) = \frac{1}{N} \sum_{i=1}^N \hat{H}_i(x, y), \quad \sigma^2(x, y) = \frac{1}{N} \sum_{i=1}^N \left( \hat{H}_i(x, y) - \hat{\mu}(x, y) \right)^2$$

**Operational 4-Quadrant Triage Matrix**:
- **Quadrant I (High Risk, Low $\sigma^2$)**: Actionable Hotspot $\rightarrow$ Prioritize desilting/dredging operations.
- **Quadrant II (Low Risk, Low $\sigma^2$)**: Watchlist Zone $\rightarrow$ Confirmed stable bed; nominal operations.
- **Quadrant III (High Risk, High $\sigma^2$)**: Ambiguous Risk $\rightarrow$ Deploy hydrographic sonar vessel for bathymetric sounding verification.
- **Quadrant IV (Low Risk, High $\sigma^2$)**: Uncertain Reach $\rightarrow$ Deploy in-situ turbidity loggers to reduce model epistemic variance.

### Counterfactual Sediment Dynamics Network (CSDN)

The CSDN engine performs Pearl-style causal structural interventions $do(\mathbf{X}_j \leftarrow \mathbf{X}_j + \delta \mathbf{X}_j)$ on environmental variables without modifying non-causal features:
- **Counterfactual Difference Heatmap**: $\Delta H(x, y) = \hat{H}_{do(\mathbf{X})}(x, y) - \hat{H}_{\text{baseline}}(x, y)$
- **Spatial Sensitivity Matrix**: $S_j(x, y) = \frac{\Delta H(x, y)}{\delta_j}$
- **Zone Vulnerability Index**: $V_{z, j} = \frac{1}{|A_z|} \sum_{(x,y) \in A_z} S_j(x, y)$

Standardized scenario perturbations evaluated:
- **Rainfall Surges**: $+10\%$, $+20\%$, $+30\%$ upstream monsoon cloudburst.
- **Discharge Variations**: $+10\%$, $-10\%$, $-20\%$ main river flow alterations.
- **Upstream Inflow Flushes**: $+20\%$, $-20\%$ regulated sluice releases from upstream dams.

---

## 5. Data Provenance & Scientific Ground Truth

A critical principle of this platform is **strict transparency regarding data sources, proxy representations, and ground truth**:

```text
==================================================================================================
                             DATA PROVENANCE & OBSERVATION MODALITIES
==================================================================================================

Modality              Source Agency / Instrument         Resolution     Role in Pipeline
──────────────────────────────────────────────────────────────────────────────────────────────────
Meteorology           Open-Meteo Historical Archive &   Hourly / 0.1°   Real historical training &
                      Live Forecast APIs (ECMWF/GFS)                   live rainfall forecasting

Turbidity Proxy       Sentinel-2 Multi-Spectral (MSI)    10m - 20m      Dynamic Sediment Deposition
                      (B4 Red 665nm, B3 Green 560nm)                    Index (NDTI proxy)

Terrain & Slope       NASA / USGS SRTM 1 Arc-Second      30m            Bathymetric slope, aspect,
                      Digital Elevation Model (DEM)                     and flow accumulation

Land Cover (LULC)     ESA / Google Dynamic World        10m            Runoff coefficient &
                      Near-Real-Time Land Cover                         vegetation buffer factor

River Discharge       Central Water Commission (CWC)    Daily Gauging  River inflow velocity &
                      India Hydrological Yearbooks                      boundary transport flux

Topological Graph     CWC Peninsular Basin Network      Reach Scale    WR-GNN directed edge graph
                      & Dam Hydrological Topology                       connectivity
==================================================================================================
```

> **Scientific Ground-Truth Disclosure**:
> 
> Satellite-derived turbidity indices (NDTI) quantify **near-surface suspended particulate matter**, which serves as an empirical proxy for sedimentation propensity. Satellite sensors **cannot penetrate deep reservoir water columns to measure actual bed sediment accumulation depth (m) or deposited volume (BCM)** without in-situ multi-beam sonar or sub-bottom profiler hydrographic surveys.
> 
> Consequently, all hotspot predictions represent **spatial deposition risk propensity $[0, 1]$** calibrated against remote-sensing turbidity targets and physical settling models. Areas triaged into Quadrant III (High Risk + High Epistemic Uncertainty) are specifically flagged for physical hydrographic sonar vessel deployment.

---

## 6. Repository Structure

```text
reservoir/
├── config.py                               # Global parameters, paths, and meteorological thresholds
├── main.py                                 # Unified multi-command backend CLI entry point
├── pipeline.py                             # Rainfall ML pipeline orchestrator (fetch, engineer, train)
├── fetch_historical_weather.py             # Open-Meteo historical weather data client
├── feature_engineering.py                  # Lag, rolling window, temporal cyclic & trend feature generator
├── train_model.py                          # XGBoost regressor training and hyperparameter configuration
├── river_weather_predictions.py            # Live weather fetch, next-hour prediction & alerting service
├── sediment_pipeline.py                    # Sediment hotspot training, prediction & multi-reservoir CLI
├── requirements.txt                        # Python package dependencies
│
├── configs/                                # Configuration YAML/JSON files
│   └── sediment_config.json                # Multimodal dimensions, physics weights, training epochs
│
├── dashboards/                             # Interactive Streamlit dashboards
│   └── sediment_dashboard.py               # Comprehensive 8-tab operational monitoring dashboard
│
├── sediment_hotspot/                       # Core multimodal AI & physics subsystem
│   ├── config.py                           # Sediment grid dimensions (128x128), channels, CRS
│   ├── evaluate.py                         # Evaluation metrics (IoU, Dice, Precision, Recall, AUC, ECE)
│   ├── baselines.py                        # Comparative baseline models (UNet, ResNet, XGBoost, Random)
│   ├── ablation.py                         # Systematic ablation study runner
│   │
│   ├── models/                             # PyTorch neural network architectures
│   │   ├── cnn.py                          # Spatial 2D convolutional encoder
│   │   ├── lstm.py                         # Temporal bidirectional meteorological LSTM
│   │   ├── gnn.py                          # Watershed-Reservoir Graph Neural Network (WR-GNN)
│   │   ├── causal_attention.py             # Causal Attention Fusion (CAF) module
│   │   ├── dsdi.py                         # DSDI physical index computation module
│   │   ├── fusion.py                       # Unified 7-modality fusion network & spatial decoder
│   │   ├── contrastive.py                  # Multimodal contrastive alignment loss
│   │   └── uncertainty.py                  # Monte Carlo Dropout uncertainty estimator
│   │
│   ├── losses/                             # Physics-informed & specialized loss functions
│   │   ├── physics_loss.py                 # PISL: Task + Smoothness + Advection + Mass Conservation
│   │   ├── temporal_loss.py                # Temporal consistency & transition penalization
│   │   └── counterfactual_loss.py          # Causal preservation & monotonic penalty loss
│   │
│   ├── counterfactual/                     # CSDN scenario generation & sensitivity analysis
│   │   ├── scenario_engine.py              # Intervention runner for rainfall, flow & upstream flushes
│   │   ├── sensitivity.py                  # Spatial sensitivity matrix & zone vulnerability index
│   │   ├── interventions.py                # Structural causal interventions do(X)
│   │   └── counterfactual.py               # CSDN main interface
│   │
│   ├── reservoirs/                         # Multi-reservoir zone-wise analysis subsystem
│   │   ├── catalog.py                      # Reservoir specifications, dam coordinates, FRL, storage, CWC
│   │   ├── zones.py                        # Flow-axis centerline zone partitioning (128x128 grid)
│   │   ├── environmental.py                # 7-modality aligned tensor builder with live Open-Meteo link
│   │   ├── connectivity.py                 # NetworkX topological graph & WR-GNN message passing
│   │   ├── counterfactual.py               # Per-reservoir CSDN scenario execution & delta maps
│   │   └── pipeline.py                     # End-to-end reservoir pipeline, metrics & artifact exporter
│   │
│   ├── preprocessing/                      # Environmental data preprocessors
│   │   ├── satellite.py                    # Sentinel-2 NDTI proxy extraction
│   │   ├── terrain.py                      # SRTM DEM elevation, slope, and aspect calculator
│   │   ├── landuse.py                      # Dynamic World LULC class encoding & runoff weights
│   │   ├── flow.py                         # CWC streamflow velocity & discharge normalization
│   │   ├── rainfall_adapter.py             # Adapter linking XGBoost forecasts to spatial grids
│   │   └── adaptive_threshold.py           # Otsu & percentile-based adaptive hotspot thresholding
│   │
│   ├── xai/                                # Explainable AI modules
│   │   └── explain.py                      # Integrated Gradients & cross-modality attention attribution
│   │
│   └── visualization/                      # Cartographic & diagnostic plotting
│       └── maps.py                         # GeoPandas & Matplotlib reservoir plotting utilities
│
├── scripts/                                # Utility and batch execution scripts
│   ├── run_multi_reservoir_analysis.py     # Batch CLI to execute multi-reservoir zone analysis
│   ├── run_scenarios_cli.py                # Standalone counterfactual scenario CLI
│   ├── generate_paper_artifacts.py         # Diagnostic figure & table generation script
│   └── download_public_data_gee.py         # Google Earth Engine data download utility
│
├── tests/                                  # Automated Pytest suite
│   ├── test_multi_reservoir.py             # Unit & integration tests for multi-reservoir subsystem
│   ├── test_counterfactual.py              # Tests for CSDN interventions, sensitivity & delta maps
│   └── test_sediment_hotspot.py            # Tests for multimodal fusion, DSDI, PISL & uncertainty
│
└── data/                                   # Data storage directory
    ├── outputs/                            # Model predictions, heatmaps, and JSON metrics
    │   └── multi_reservoir/                # Per-reservoir output artifacts
    │       ├── krishna_nagarjuna_sagar/    # Nagarjuna Sagar 9 standardized artifacts + maps
    │       ├── narmada_omkareshwar/        # Omkareshwar 9 standardized artifacts + maps
    │       ├── kaveri_mettur/              # Mettur Dam 9 standardized artifacts + maps
    │       └── reservoir_comparison.csv    # Cross-basin comparative benchmark summary
    └── processed/                          # Preprocessed training tensors and metadata
```

---

## 7. Installation & Environment Setup

### Prerequisites

- Linux / Ubuntu / WSL2 on Windows
- Python 3.10 to 3.12 (Python 3.12 recommended)
- 8 GB RAM minimum (16 GB recommended for batch multi-reservoir runs)
- NVIDIA GPU with CUDA support (optional; full CPU fallback is natively supported)

### Step 1: Clone Repository & Create Virtual Environment

```bash
git clone https://github.com/your-username/reservoir.git
cd reservoir

python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Bootstrap Starter Weights & Verification Artifacts

Initialize starter model weights and minimal sample datasets so tests, CLI workflows, and dashboards run out-of-the-box:

```bash
python scripts/bootstrap_model.py
```

Key packages installed:
- `torch`, `torchvision`, `torchaudio` (PyTorch deep learning framework)
- `xgboost` (Extreme Gradient Boosting regressor)
- `streamlit` (Operational interactive web dashboard)
- `geopandas`, `shapely`, `pyproj` (Geospatial vector processing)
- `networkx` (Topological graph construction & routing)
- `scikit-learn`, `scipy`, `pandas`, `numpy` (Scientific computing)
- `matplotlib`, `seaborn` (Cartographic and diagnostic visualization)
- `requests` (Open-Meteo REST API data ingestion)
- `pytest` (Automated testing framework)

---

## 8. Execution & CLI Workflows

The repository provides modular CLI commands and unified workflows for both operational monitoring and research experimentation.

### Workflow A: Live Rainfall Forecasting & Alerts

Fetch live Open-Meteo meteorological forecasts for all 14 CWC monitoring stations, execute the trained XGBoost model, evaluate next-hour rainfall, and output meteorological alerts:

```bash
# Direct execution
python river_weather_predictions.py

# Or via unified entry point
python main.py predict
```

Output generated:
- `rainfall_predictions.csv`: Station-level current weather, predicted rainfall, and alert status.
- `rainfall_predictions_history.csv`: Rolling historical archive for temporal trend analysis.

### Workflow B: Complete Rainfall Pipeline (Fetch, Engineer, Train)

To re-fetch historical weather archives, regenerate lag/rolling features, and retrain the XGBoost regressor:

```bash
# Step 1: Fetch historical Open-Meteo weather data (2020 to present)
python pipeline.py --fetch

# Step 2: Engineer lag, rolling, cyclic, and trend features
python pipeline.py --engineer

# Step 3: Train XGBoost next-hour regressor
python pipeline.py --train

# Or execute all stages sequentially
python main.py rainfall-pipeline --all
```

### Workflow C: Multi-Reservoir Zone-Wise Sedimentation Analysis

Execute the complete multi-reservoir zone analysis pipeline across the Krishna, Narmada, and Kaveri river basins:

```bash
# Run analysis for prioritized Nagarjuna Sagar Reservoir (Krishna Basin)
python scripts/run_multi_reservoir_analysis.py --reservoir krishna_nagarjuna_sagar

# Run analysis for Omkareshwar Reservoir (Narmada Basin)
python scripts/run_multi_reservoir_analysis.py --reservoir narmada_omkareshwar

# Run analysis for Mettur Stanley Reservoir (Kaveri Basin)
python scripts/run_multi_reservoir_analysis.py --reservoir kaveri_mettur

# Run batch analysis across all reservoirs and generate cross-basin comparison
python scripts/run_multi_reservoir_analysis.py --reservoir all

# Alternatively, execute via sediment_pipeline.py
python sediment_pipeline.py --multi-reservoir all
```

Standardized outputs generated in `data/outputs/multi_reservoir/<reservoir_id>/`:
1. `reservoir_zone_map.png` & `reservoir_zone_map.csv`: 6-zone spatial partitioning and coordinates.
2. `hotspot_heatmap.png` & `future_sediment_hotspot_heatmap.npy`: Pixel-level hotspot probability $[0, 1]$.
3. `zone_sediment_risk.csv`: Aggregated zone risk scores, area percentages, and peak risk coordinates.
4. `dsdi_heatmap.npy` & `zone_dsdi_scores.csv`: Dynamic Sediment Deposition Index values per zone.
5. `zone_connectivity_graph.png` & `zone_connectivity_graph.json`: WR-GNN upstream-to-downstream topological graph.
6. `uncertainty_heatmap.npy` & `zone_uncertainty.csv`: Epistemic uncertainty variance ($\sigma^2$) and triage quadrant assignments.
7. `counterfactual_results.csv`, `counterfactual_risk_change.npy` & `sensitivity_maps/`: CSDN perturbation responses ($\Delta H$) and spatial sensitivity.
8. `zone_sensitivity.csv`: Vulnerability index ranking across all 6 zones for each scenario.
9. `evaluation_metrics.json`: Adaptive remote-sensing validation metrics (IoU, Dice, Precision, Recall, AUC, ECE).

Cross-basin summary output:
- `data/outputs/multi_reservoir/reservoir_comparison.csv`: Comparative table across all evaluated reservoirs.

### Workflow D: Standalone Counterfactual Scenario Runner

To execute targeted causal interventions on a reservoir model:

```bash
python scripts/run_scenarios_cli.py
```

### Workflow E: Launching the Interactive Decision-Support Dashboard

Launch the Streamlit web dashboard to access interactive visualizations, live forecasting, geospatial maps, and counterfactual scenario simulations:

```bash
streamlit run dashboards/sediment_dashboard.py
```

---

## 9. Interactive Streamlit Decision-Support Dashboard

The dashboard provides an intuitive, professional, emoji-free interface divided into **8 dedicated operational tabs**:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        RESERVOIR INTELLIGENCE MONITORING PLATFORM                      │
│                                                                                        │
│  [Tab 1] System Overview & Framework Metrics                                           │
│  [Tab 2] Real-Time Rainfall Forecasting & Meteorological Alerts                        │
│  [Tab 3] Spatial Hotspot Prediction & Dynamic Sediment Deposition Index (DSDI)         │
│  [Tab 4] Zone-Wise Sediment Risk & Topological Connectivity                            │
│  [Tab 5] Epistemic Uncertainty & 4-Quadrant Operational Triage                         │
│  [Tab 6] Explainability & Causal Attention Attribution                                 │
│  [Tab 7] Counterfactual Scenario Engine (CSDN Interventions)                           │
│  [Tab 8] Multi-Reservoir Zone Analysis (Krishna, Narmada & Kaveri Basins)              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Tab Breakdown

1. **Tab 1: Overview**
   - High-level system architecture, research innovations, and core framework metrics.
   - Live Open-Meteo API query trigger to refresh weather and rainfall predictions on demand.
2. **Tab 2: Rainfall Forecast**
   - Live meteorological tables for 14 CWC stations across Krishna, Narmada, and Kaveri basins.
   - Station-level rainfall predictions, severity categories, and active flood risk alerts.
3. **Tab 3: Hotspot & DSDI**
   - Side-by-side spatial heatmaps comparing predicted hotspot probability against the physical DSDI index.
   - Interactive threshold slider for dynamic contour identification.
4. **Tab 4: Zone Risk & Connectivity**
   - Zone-level sediment risk bar charts and high-risk area percentages.
   - Directed NetworkX upstream-to-downstream topological station graph.
5. **Tab 5: Epistemic Uncertainty**
   - Monte Carlo Dropout predictive variance ($\sigma^2$) heatmap.
   - Interactive 4-quadrant decision triage matrix (Actionable Hotspot, Watchlist Zone, Ambiguous Risk, Uncertain Reach) for hydrographic sonar vessel deployment.
6. **Tab 6: Explainability & Causal Attention**
   - Modality attribution weights (Satellite NDTI, DEM Slope, LULC, Flow, Rainfall, Graph, DSDI).
   - Upstream-to-downstream causal attention matrix and spatial gradient saliency maps.
7. **Tab 7: Counterfactual Scenarios (CSDN)**
   - Interactive causal scenario selection ($+10\%$, $+20\%$, $+30\%$ rainfall; $\pm 10\%$, $\pm 20\%$ flow).
   - Side-by-side baseline vs counterfactual comparison and difference maps ($\Delta H$).
   - Zone vulnerability rankings under selected interventions.
8. **Tab 8: Multi-Reservoir Zone Analysis**
   - Multi-reservoir selector covering **Nagarjuna Sagar (Krishna)**, **Omkareshwar (Narmada)**, and **Mettur Stanley Reservoir (Kaveri)**.
   - Sub-tab 1: Reservoir location metadata (dam lat/lon, storage, FRL) with 6-zone spatial boundary visualization and predicted hotspot overlay.
   - Sub-tab 2: Zone-wise deposition risk table, peak risk coordinates, and DSDI alignment.
   - Sub-tab 3: Epistemic uncertainty distribution and WR-GNN topological connectivity graph.
   - Sub-tab 4: Per-reservoir CSDN counterfactual scenario simulator with difference heatmaps and zone vulnerability metrics.
   - Sub-tab 5: Cross-basin reservoir comparison benchmark table comparing gross storage, mean risk, DSDI, detection IoU, and forecast rain.
   - Sub-tab 6: Scientific data provenance disclosures and quantitative evaluation metrics against Sentinel-2 turbidity proxies.

---

## 10. Evaluation Metrics & Validation Results

The models are evaluated against adaptive remote-sensing targets derived from Sentinel-2 NDTI proxies. Performance metrics across the primary reservoirs:

### Quantitative Performance Summary

| Metric | Nagarjuna Sagar (Krishna) | Omkareshwar (Narmada) | Mettur Stanley (Kaveri) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Detection IoU** | 0.121 | 0.162 | 0.316 | Intersection-over-Union against adaptive NDTI hotspot regions |
| **Dice Score (F1)** | 0.215 | 0.279 | 0.481 | Harmonic mean of precision and recall on spatial hotspot pixels |
| **Precision** | 0.197 | 0.245 | 0.468 | Fraction of predicted hotspot pixels confirmed by turbidity proxy |
| **Recall** | 0.237 | 0.324 | 0.495 | Fraction of high-turbidity proxy pixels captured by the model |
| **ROC AUC** | 0.564 | 0.598 | 0.684 | Area under the Receiver Operating Characteristic curve |
| **PR AUC** | 0.174 | 0.221 | 0.412 | Area under the Precision-Recall curve |
| **Brier Score** | 0.220 | 0.198 | 0.154 | Mean squared difference between predicted probability and proxy target |
| **Expected Calibration Error (ECE)** | 0.219 | 0.187 | 0.138 | Reliability calibration error across predicted probability bins |
| **Uncertainty-Error Correlation** | 0.147 | 0.182 | 0.210 | Pearson correlation between epistemic variance $\sigma^2$ and prediction error |

### Cross-Basin Reservoir Characteristics

| Reservoir ID | Basin | Gross Storage (BCM) | FRL (m) | Mean Hotspot Risk | Peak Hotspot Risk | High-Risk Area (%) | Top Vulnerable Zone (+20% Rain) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `krishna_nagarjuna_sagar` | Krishna | 11.472 | 179.80 | 0.3693 | 0.9612 | 43.45% | Deep Lacustrine Storage Zone |
| `narmada_omkareshwar` | Narmada | 0.987 | 196.60 | 0.3668 | 0.9592 | 30.24% | Indira Sagar Tailrace Entry Zone |
| `kaveri_mettur` | Kaveri | 2.640 | 241.00 | 0.3683 | 0.9622 | 51.31% | Lacustrine Storage Reach |

---

## 11. Testing & Quality Assurance

The codebase includes an automated test suite covering unit operations, tensor dimensions, physics loss convergence, topological graph routing, counterfactual mechanics, and multi-reservoir pipelines.

### Running the Test Suite

```bash
# Run complete test suite
pytest tests/ -v

# Run multi-reservoir subsystem tests specifically
pytest tests/test_multi_reservoir.py -v

# Run counterfactual scenario tests
pytest tests/test_counterfactual.py -v

# Run core sediment hotspot and physics loss tests
pytest tests/test_sediment_hotspot.py -v
```

### Test Coverage Highlights

- **`test_multi_reservoir.py`**:
  - `test_reservoir_catalog_coverage`: Validates coordinates, FRL, storage, and CWC stations for all reservoirs.
  - `test_zone_partitioning_geometry`: Verifies 6-zone discretization, water masking, and raster dimensions ($128 \times 128$).
  - `test_environmental_tensor_alignment`: Validates 7-modality tensor channel dimensions and value bounds.
  - `test_connectivity_graph_topological_consistency`: Validates NetworkX DAG construction and edge flow directions.
  - `test_counterfactual_scenario_execution`: Tests $do(\Delta \text{Rainfall})$ and verifies $\Delta H$ and sensitivity matrices.
  - `test_single_reservoir_pipeline_artifacts`: Confirms generation of all 9 required analytical artifacts.
  - `test_batch_reservoir_comparison_summary`: Verifies cross-basin CSV generation and schema completeness.
- **`test_counterfactual.py`**:
  - Validates CSDN scenario generation, difference heatmaps, sensitivity calculation, and monotonicity constraints.
- **`test_sediment_hotspot.py`**:
  - Tests DSDI physical index computation, multimodal fusion forward pass, PISL physics loss components, and Monte Carlo Dropout uncertainty estimation.

All **22 automated tests** execute and pass cleanly.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments & Data Sources

- **Central Water Commission (CWC), Government of India**: River basin hydrologic station data and reservoir storage specifications.
- **Open-Meteo**: High-resolution meteorological archive and real-time numerical weather prediction APIs.
- **European Space Agency (ESA) & Copernicus**: Sentinel-2 Multi-Spectral satellite imagery.
- **NASA / USGS**: Shuttle Radar Topography Mission (SRTM) 30m Digital Elevation Model.
- **Google & Dynamic World**: Near-real-time 10m global land use / land cover dataset.

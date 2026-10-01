# Reservoir Rainfall and Sediment Hotspot Prediction

## A Multi-Modal AI System for Spatial Sediment Risk Analysis

An AI-based environmental monitoring and decision-support system that combines short-term rainfall forecasting with multimodal sediment hotspot prediction inside reservoirs.

The system integrates meteorological data, satellite imagery, terrain information, land-use information, river-flow observations, graph connectivity, and derived physical indicators. Its primary outputs are spatial sediment-risk heatmaps, zone-wise risk scores, uncertainty estimates, explainability outputs, and counterfactual scenario analysis.

The project is designed as a research-oriented prototype with an emphasis on reproducibility, multimodal learning, spatial reasoning, uncertainty awareness, and interpretable environmental decision support.

---

## 1. Project Overview

Reservoir sedimentation progressively reduces usable storage capacity and can affect irrigation, hydropower, water supply, flood management, and long-term reservoir operation.

Conventional sediment assessment is frequently based on periodic surveys or aggregate sediment estimates. This project instead focuses on predicting **where sedimentation risk is likely to be concentrated** and how that risk may change under different environmental conditions.

The system has two connected stages:

1. **Rainfall Forecasting**
   - Historical Open-Meteo weather data is used to train an XGBoost model for next-hour rainfall prediction.
   - Live weather information is converted into rainfall predictions for 14 river zones across 3 major river basins (Krishna, Narmada, Kaveri).

2. **Sediment Hotspot Prediction**
   - Predicted rainfall is combined with satellite, terrain, land-use, flow, graph, and physical features.
   - A multimodal PyTorch model produces pixel-level hotspot probabilities and zone-level sediment risk.
   - Additional modules provide physical constraints, causal structure, graph propagation, contrastive alignment, uncertainty estimation, explainability, and counterfactual analysis.

---

## 2. Key Capabilities

### Rainfall Forecasting

- Open-Meteo historical and forecast data integration
- Next-hour rainfall prediction using XGBoost
- Lag, rolling, trend, temporal, and rainfall-severity features
- Live prediction pipeline with real-time Open-Meteo forecast API fetching
- Rainfall alerts based on standardized meteorological severity categories

### Multimodal Sediment Prediction

The sediment model integrates seven information streams:

1. Satellite imagery
2. Terrain
3. Land use
4. River flow
5. Rainfall
6. Watershed/reservoir graph information
7. Dynamic Sediment Deposition Index (DSDI)

### Research-Oriented Components

- Dynamic Sediment Deposition Index (DSDI)
- Physics-Informed Sediment Loss (PISL)
- Causal Attention Fusion
- Watershed/Reservoir Graph Neural Network
- Multi-Modal Contrastive Alignment
- Temporal Consistency Loss
- Adaptive Hotspot Thresholding
- Uncertainty-Aware Hotspot Prediction
- Counterfactual Sediment Dynamics / intervention analysis
- Explainability through attention and gradient-based methods

---

## 3. End-to-End Architecture

```text
                         RAINFALL SUBSYSTEM

┌───────────────────────┐
│ Open-Meteo Historical │
│ Weather Data          │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ Feature Engineering   │
│ Lags / Rolling /      │
│ Trends / Time         │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ XGBoost Regressor     │
│ Next-Hour Rainfall    │
└──────────┬────────────┘
           │
           ├──────────────► rainfall_model.pkl
           │
           ▼
┌───────────────────────┐
│ Live Open-Meteo       │
│ Forecast Inference    │
└──────────┬────────────┘
           │
           ▼
    rainfall_predictions.csv
           │
           │
           ▼

                    SEDIMENT SUBSYSTEM

 ┌────────────┐  ┌──────────┐  ┌────────────┐
 │ Satellite  │  │   DEM    │  │ Land Cover │
 └─────┬──────┘  └────┬─────┘  └─────┬──────┘
       │              │              │
       ▼              ▼              ▼
 ┌────────────┐  ┌──────────┐  ┌────────────┐
 │ Satellite  │  │ Terrain  │  │ Land-Use   │
 │ Features   │  │ Features │  │ Features   │
 └─────┬──────┘  └────┬─────┘  └─────┬──────┘
       │              │              │
       └──────────────┼──────────────┘
                      │
 ┌────────────┐  ┌────────────┐  ┌────────────┐
 │ River Flow │  │ Rainfall   │  │ DSDI       │
 └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
       │               │               │
       └───────────────┼───────────────┘
                       ▼
              ┌─────────────────────┐
              │ Multimodal Fusion   │
              │ CNN + LSTM + GNN +  │
              │ DSDI + Attention    │
              └──────────┬──────────┘
                         │
          ┌──────────────┼───────────────┐
          │              │               │
          ▼              ▼               ▼
   Hotspot Heatmap   Zone Risk      DSDI / Attention
          │              │               │
          └──────────────┼───────────────┘
                         ▼
              ┌─────────────────────┐
              │ Uncertainty / XAI   │
              │ / Counterfactuals   │
              └──────────┬──────────┘
                         ▼
                Streamlit Dashboard
```

---

## 4. Centralized Configuration (`config.py`)

All global parameters, file paths, thresholds, and hyperparameters are centralized in [`config.py`](config.py) to prevent hardcoding across individual pipeline scripts.

Key parameters configured in `config.py`:
- **Paths**: Absolute path resolution for model artifacts (`rainfall_model.pkl`), training data (`training_dataset.csv`), historical data (`historical_weather.csv`), and output predictions (`rainfall_predictions.csv`, `rainfall_predictions_history.csv`).
- **Dynamic Fetch Window**: `HISTORY_START_DATE` (`2020-01-01`) to `HISTORY_END_DATE` (dynamically set to current date `datetime.today()`).
- **Rainfall Thresholds**: Meteorological boundaries (`LIGHT_RAIN_MM = 2.5`, `MODERATE_RAIN_MM = 10.0`) shared across feature engineering and prediction alerting.
- **Alert Threshold**: High rainfall alert triggering threshold (`ALERT_THRESHOLD_MM = 10.0`).
- **Model Hyperparameters**: XGBoost configuration (`N_ESTIMATORS = 200`, `LEARNING_RATE = 0.05`, `MAX_DEPTH = 6`, `RANDOM_STATE = 42`, `TEST_SIZE = 0.2`).
- **API Settings**: Request rate limits and timeouts for Open-Meteo Archive & Forecast APIs.

---

## 5. Repository Structure

```text
reservoir/
├── config.py                       # Centralized project configuration & parameters
├── main.py                         # Unified backend CLI entry point
├── pipeline.py                     # Rainfall ML pipeline orchestrator
├── fetch_historical_weather.py     # Open-Meteo historical weather data fetcher
├── feature_engineering.py          # Lag, rolling window, trend & time feature generation
├── train_model.py                  # XGBoost model training & evaluation
├── river_weather_predictions.py    # Real-time rainfall prediction & alerting service
├── sediment_pipeline.py            # Sediment hotspot prediction pipeline orchestrator
├── sediment_hotspot/               # Multimodal PyTorch models, graph neural networks, loss functions
├── dashboards/                     # Interactive Streamlit dashboards
├── configs/                        # Model & data configuration files
└── requirements.txt                # Python dependencies
```

---

## 6. Execution & Usage

### Unified CLI Entry Point (`main.py`)

Run subsystems directly via `main.py`:

```bash
# Run complete rainfall pipeline (fetch historical data, engineer features, train model)
python main.py rainfall-pipeline --all

# Run real-time rainfall prediction on active river zones
python main.py predict

# Run sediment hotspot pipeline
python main.py sediment-pipeline --preprocess --train --predict

# Launch interactive Streamlit monitoring dashboard
python main.py dashboard

# Run end-to-end rainfall pipeline + real-time predictions
python main.py all
```

### Modular Pipeline Execution

You can also run individual pipeline stages separately:

```bash
# 1. Fetch historical weather data (2020 to present)
python pipeline.py --fetch

# 2. Engineer features
python pipeline.py --engineer

# 3. Train XGBoost model
python pipeline.py --train

# 4. Run real-time live predictions
python river_weather_predictions.py
```

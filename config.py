"""
Centralized configuration for the Reservoir Intelligence System.

All scripts import from this module so that constants and file paths
are defined in exactly one place.
"""

from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Root directory (always the directory that contains this file)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
MODEL_FILE       = ROOT_DIR / "rainfall_model.pkl"
PREDICTIONS_FILE = ROOT_DIR / "rainfall_predictions.csv"
HISTORY_FILE     = ROOT_DIR / "rainfall_predictions_history.csv"
HISTORICAL_DATA  = ROOT_DIR / "historical_weather.csv"
TRAINING_DATA    = ROOT_DIR / "training_dataset.csv"

# ---------------------------------------------------------------------------
# Historical fetch window
# ---------------------------------------------------------------------------
HISTORY_START_DATE = "2020-01-01"
HISTORY_END_DATE   = datetime.today().strftime("%Y-%m-%d")   # always current

# ---------------------------------------------------------------------------
# Rainfall classification thresholds  (mm / hr)
# Shared by feature_engineering.py AND river_weather_predictions.py
# ---------------------------------------------------------------------------
LIGHT_RAIN_MM    = 2.5
MODERATE_RAIN_MM = 10.0

# ---------------------------------------------------------------------------
# Alert threshold
# ---------------------------------------------------------------------------
ALERT_THRESHOLD_MM = 10.0

# ---------------------------------------------------------------------------
# XGBoost hyper-parameters
# ---------------------------------------------------------------------------
N_ESTIMATORS  = 200
LEARNING_RATE = 0.05
MAX_DEPTH     = 6
RANDOM_STATE  = 42

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------
FORECAST_API_TIMEOUT   = 20    # seconds  - Open-Meteo forecast endpoint
HISTORICAL_API_TIMEOUT = 30    # seconds  - Open-Meteo archive endpoint
FORECAST_RATE_LIMIT    = 0.1   # seconds between zone requests (forecast)
HISTORICAL_RATE_LIMIT  = 1.0   # seconds between zone fetches  (archive)

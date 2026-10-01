"""
Real-Time River Rainfall Prediction System

This script uses the trained XGBoost model to make rainfall predictions
for all river zones in real-time using Open-Meteo Forecast API.

NOTE: This script does NOT collect training data anymore.
      Training data comes from fetch_historical_weather.py (2020-2025 historical data)
      
      This script is for LIVE PREDICTION ONLY.

Workflow:
  1. Fetch current weather from Open-Meteo Forecast API
  2. Create features (lag, rolling avg - from recent historical context)
  3. Predict next hour rainfall using trained model
  4. Alert if high rainfall probability detected
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
from pathlib import Path
import joblib
import warnings

from config import (
    MODEL_FILE,
    PREDICTIONS_FILE,
    HISTORY_FILE,
    LIGHT_RAIN_MM,
    MODERATE_RAIN_MM,
    ALERT_THRESHOLD_MM,
    FORECAST_API_TIMEOUT,
    FORECAST_RATE_LIMIT,
)

warnings.filterwarnings('ignore')

RIVER_ZONES = {
    "Krishna": {
        "K1_Mahabaleshwar": (17.9237, 73.6586),
        "K2_Sangli_Almatti": (16.8544, 74.5642),
        "K3_Raichur_Kurnool": (16.2076, 77.3463),
        "K4_Nagarjuna_Sagar": (16.5750, 79.3167),
        "K5_Vijayawada_Delta": (16.5062, 80.6480),
    },

    "Narmada": {
        "N1_Amarkantak": (22.6747, 81.7590),
        "N2_Jabalpur": (23.1815, 79.9864),
        "N3_Omkareshwar": (22.2452, 76.1510),
        "N4_Bharuch": (21.7051, 72.9959),
    },

    "Kaveri": {
        "C1_Talakaveri": (12.3855, 75.4894),
        "C2_Kodagu": (12.3375, 75.8069),
        "C3_Mysuru_KRS": (12.2958, 76.6394),
        "C4_Mettur_Dam": (11.7870, 77.8008),
        "C5_Thanjavur_Delta": (10.7867, 79.1378),
    }
}


def get_weather(lat, lon):
    """Fetch current weather from Open-Meteo Forecast API"""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,"
        f"relative_humidity_2m,"
        f"precipitation,"
        f"surface_pressure,"
        f"wind_speed_10m,"
        f"cloud_cover"
    )

    try:
        response = requests.get(url, timeout=FORECAST_API_TIMEOUT)
        data = response.json()
        current = data.get("current", {})

        return {
            "temperature": current.get("temperature_2m", 0.0),
            "humidity": current.get("relative_humidity_2m", 0.0),
            "rainfall": current.get("precipitation", 0.0),
            "pressure": current.get("surface_pressure", 1013.0),
            "wind_speed": current.get("wind_speed_10m", 0.0),
            "cloud_cover": current.get("cloud_cover", 0.0),
        }

    except Exception as e:
        print(f"Error fetching weather for ({lat}, {lon}): {e}")
        return None


def create_prediction_features(weather_data, lat=0.0, lon=0.0):
    """
    Create features for model prediction.
    Features order strictly aligns with rainfall_model.pkl training dataset.
    """
    temp  = float(weather_data.get("temperature", 0.0) or 0.0)
    hum   = float(weather_data.get("humidity", 0.0) or 0.0)
    rain  = float(weather_data.get("rainfall", 0.0) or 0.0)
    press = float(weather_data.get("pressure", 1013.0) or 1013.0)
    wind  = float(weather_data.get("wind_speed", 0.0) or 0.0)
    cloud = float(weather_data.get("cloud_cover", 0.0) or 0.0)

    now = datetime.now()

    features = {
        "latitude": float(lat),
        "longitude": float(lon),
        "temperature": temp,
        "humidity": hum,
        "rainfall": rain,
        "pressure": press,
        "wind_speed": wind,
        "cloud_cover": cloud,
        
        # Lag features (contextual approximations when streaming single points)
        "temperature_lag_1": temp,
        "temperature_lag_2": temp,
        "temperature_lag_3": temp,
        "humidity_lag_1": hum,
        "humidity_lag_2": hum,
        "humidity_lag_3": hum,
        "rainfall_lag_1": rain,
        "rainfall_lag_2": rain,
        "rainfall_lag_3": rain,
        "pressure_lag_1": press,
        "pressure_lag_2": press,
        "pressure_lag_3": press,
        "wind_speed_lag_1": wind,
        "wind_speed_lag_2": wind,
        "wind_speed_lag_3": wind,
        "cloud_cover_lag_1": cloud,
        "cloud_cover_lag_2": cloud,
        "cloud_cover_lag_3": cloud,
        
        # Rolling averages
        "rainfall_avg_3": rain,
        "humidity_avg_3": hum,
        "temperature_avg_3": temp,
        
        # Change features
        "pressure_change": 0.0,
        "humidity_change": 0.0,
        "temperature_change": 0.0,
        "wind_change": 0.0,
        
        # Time features
        "hour": int(now.hour),
        "day": int(now.day),
        "month": int(now.month),
    }
    
    return features


def classify_rainfall(rainfall_mm):
    """Classify rainfall into standard meteorological categories"""
    if rainfall_mm is None or np.isnan(rainfall_mm):
        return "Unknown"
    
    if rainfall_mm == 0:
        return "No Rain"
    elif rainfall_mm < LIGHT_RAIN_MM:
        return "Light"
    elif rainfall_mm < MODERATE_RAIN_MM:
        return "Moderate"
    else:
        return "Heavy"


def predict_rainfall(model, river, zone, weather_data, lat, lon):
    """Make rainfall prediction for a zone"""
    try:
        features = create_prediction_features(weather_data, lat, lon)
        feature_vector = pd.DataFrame([features])
        
        # Predict next-hour rainfall
        pred = model.predict(feature_vector)[0]
        predicted_rainfall = max(0.0, float(pred))
        
        classification = classify_rainfall(predicted_rainfall)
        alert = "⚠️ HIGH RAINFALL EXPECTED" if predicted_rainfall > ALERT_THRESHOLD_MM else ""
        
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "river": river,
            "zone": zone,
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "current_temp": weather_data.get("temperature"),
            "current_humidity": weather_data.get("humidity"),
            "current_rainfall": weather_data.get("rainfall"),
            "predicted_rainfall_next_hour": round(predicted_rainfall, 2),
            "rainfall_class": classification,
            "alert": alert,
        }
        
    except Exception as e:
        print(f"Error predicting for {zone}: {e}")
        return None


def collect_predictions(model):
    """Collect real-time predictions across all river zones"""
    predictions = []

    for river, zones in RIVER_ZONES.items():
        for zone_name, coords in zones.items():
            lat, lon = coords
            weather = get_weather(lat, lon)
            if weather:
                prediction = predict_rainfall(model, river, zone_name, weather, lat, lon)
                if prediction:
                    predictions.append(prediction)
                    alert_str = f" {prediction['alert']}" if prediction['alert'] else ""
                    print(
                        f"[PRED] {river:8} | {zone_name:20} | "
                        f"Predicted: {prediction['predicted_rainfall_next_hour']: 6.2f}mm "
                        f"({prediction['rainfall_class']:10}){alert_str}"
                    )
            time.sleep(FORECAST_RATE_LIMIT)

    return predictions


def save_predictions(predictions, output_file=PREDICTIONS_FILE, history_file=HISTORY_FILE):
    """Save predictions to CSV for dashboard & append to history"""
    df = pd.DataFrame(predictions)
    
    # Write latest snapshot
    df.to_csv(output_file, index=False)
    
    # Append to running historical log
    if Path(history_file).exists():
        df.to_csv(history_file, mode='a', header=False, index=False)
    else:
        df.to_csv(history_file, index=False)

    print(f"\n✓ Saved {len(predictions)} predictions to {output_file}")


def run_live_predictions(model_file=MODEL_FILE, output_file=PREDICTIONS_FILE):
    """Programmatic entrypoint for Streamlit or automation"""
    model_path = Path(model_file)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    predictions = collect_predictions(model)
    if predictions:
        save_predictions(predictions, output_file=output_file)
        return pd.DataFrame(predictions)
    return pd.DataFrame()


def main():
    """Main prediction loop"""
    if not Path(MODEL_FILE).exists():
        print(f"✗ Error: {MODEL_FILE} not found")
        print("  Train the model first with: python train_model.py")
        return

    print("\n" + "="*70)
    print("REAL-TIME RAINFALL PREDICTIONS (Open-Meteo + XGBoost)")
    print("="*70)
    print(f"Model: {MODEL_FILE}")
    print(f"Zones: {sum(len(z) for z in RIVER_ZONES.values())} zones across 3 river basins")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    try:
        model = joblib.load(MODEL_FILE)
        print("✓ Model loaded\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    predictions = collect_predictions(model)

    if predictions:
        save_predictions(predictions)
    else:
        print("\n✗ No predictions generated")


if __name__ == "__main__":
    main()

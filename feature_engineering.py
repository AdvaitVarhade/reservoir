import pandas as pd
import numpy as np

from config import (
    HISTORICAL_DATA,
    TRAINING_DATA,
    LIGHT_RAIN_MM,
    MODERATE_RAIN_MM,
)


def create_features(df):

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(
        by=["river", "zone", "timestamp"]
    )

    grouped = df.groupby(["river", "zone"])

    lag_columns = [
        "temperature",
        "humidity",
        "rainfall",
        "pressure",
        "wind_speed",
        "cloud_cover"
    ]

    # -----------------------------
    # Create lag features
    # -----------------------------
    for col in lag_columns:

        df[f"{col}_lag_1"] = grouped[col].shift(1)
        df[f"{col}_lag_2"] = grouped[col].shift(2)
        df[f"{col}_lag_3"] = grouped[col].shift(3)

    # -----------------------------
    # Rolling weather averages
    # -----------------------------
    df["rainfall_avg_3"] = grouped["rainfall"].transform(
        lambda x: x.rolling(3).mean()
    )

    df["humidity_avg_3"] = grouped["humidity"].transform(
        lambda x: x.rolling(3).mean()
    )

    df["temperature_avg_3"] = grouped["temperature"].transform(
        lambda x: x.rolling(3).mean()
    )

    # -----------------------------
    # Weather trend features
    # -----------------------------
    df["pressure_change"] = (
        df["pressure"]
        - df["pressure_lag_1"]
    )

    df["humidity_change"] = (
        df["humidity"]
        - df["humidity_lag_1"]
    )

    df["temperature_change"] = (
        df["temperature"]
        - df["temperature_lag_1"]
    )

    df["wind_change"] = (
        df["wind_speed"]
        - df["wind_speed_lag_1"]
    )

    # -----------------------------
    # Time Features
    # -----------------------------
    df["hour"]  = df["timestamp"].dt.hour
    df["day"]   = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month

    # -----------------------------
    # Target Variable
    # Predict rainfall in next hour
    # -----------------------------
    df["rainfall_next_hour"] = grouped["rainfall"].shift(-1)

    # -----------------------------
    # Rainfall Classification
    # Uses thresholds from config (LIGHT_RAIN_MM, MODERATE_RAIN_MM)
    # -----------------------------
    def classify_rainfall(rain):

        if pd.isna(rain):
            return np.nan

        if rain == 0:
            return 0
        elif rain < LIGHT_RAIN_MM:
            return 1
        elif rain < MODERATE_RAIN_MM:
            return 2
        else:
            return 3

    df["rainfall_class"] = df["rainfall_next_hour"].apply(classify_rainfall)

    # -----------------------------
    # Remove incomplete rows
    # -----------------------------
    df = df.dropna()

    return df


def main():

    print("Loading weather data...")

    df = pd.read_csv(HISTORICAL_DATA)

    print("Creating features...")

    final_df = create_features(df)

    final_df.to_csv(
        TRAINING_DATA,
        index=False
    )

    print("\nDone")
    print(f"Rows created: {len(final_df)}")
    print(f"Saved to: {TRAINING_DATA}")

    print("\nColumns:")
    print(final_df.columns.tolist())


if __name__ == "__main__":
    main()

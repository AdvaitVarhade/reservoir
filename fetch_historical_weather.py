"""
Fetch historical weather data from Open-Meteo Archive API
for all 14 river zones (hourly)

This generates ~500k+ rows for proper model training
"""

import requests
import pandas as pd
from datetime import datetime
import time
import sys

from config import (
    HISTORICAL_DATA,
    HISTORY_START_DATE,
    HISTORY_END_DATE,
    HISTORICAL_API_TIMEOUT,
    HISTORICAL_RATE_LIMIT,
)

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


def fetch_historical_weather(lat, lon, start_date=HISTORY_START_DATE, end_date=HISTORY_END_DATE):
    """
    Fetch hourly historical weather data from Open-Meteo Archive API
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"hourly=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,cloud_cover&"
        f"timezone=Asia/Kolkata"
    )

    try:
        print(f"  Fetching {lat}, {lon}...", end="", flush=True)
        response = requests.get(url, timeout=HISTORICAL_API_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        hourly_data = data.get("hourly", {})
        times = pd.to_datetime(hourly_data.get("time", []))
        
        records = []
        for i, timestamp in enumerate(times):
            records.append({
                "timestamp": timestamp,
                "temperature": hourly_data.get("temperature_2m", [None])[i],
                "humidity": hourly_data.get("relative_humidity_2m", [None])[i],
                "rainfall": hourly_data.get("precipitation", [None])[i],
                "pressure": hourly_data.get("surface_pressure", [None])[i],
                "wind_speed": hourly_data.get("wind_speed_10m", [None])[i],
                "cloud_cover": hourly_data.get("cloud_cover", [None])[i],
            })

        print(f" ✓ {len(records)} records")
        return records

    except requests.exceptions.RequestException as e:
        print(f" ✗ Error: {e}")
        return []


def build_historical_dataset():
    """
    Build complete historical weather dataset for all zones
    """
    all_records = []
    total_zones = sum(len(zones) for zones in RIVER_ZONES.values())
    current_zone = 0

    for river, zones in RIVER_ZONES.items():
        print(f"\n{river} ({len(zones)} zones):")

        for zone_name, (lat, lon) in zones.items():
            current_zone += 1
            print(f"  [{current_zone}/{total_zones}] {zone_name}", end="")

            records = fetch_historical_weather(lat, lon)

            for record in records:
                record["river"] = river
                record["zone"] = zone_name
                record["latitude"] = lat
                record["longitude"] = lon

            all_records.extend(records)

            # Rate limiting: Open-Meteo allows reasonable requests
            time.sleep(HISTORICAL_RATE_LIMIT)

    # Convert to DataFrame
    df = pd.DataFrame(all_records)

    # Ensure proper column order
    column_order = [
        "timestamp",
        "river",
        "zone",
        "latitude",
        "longitude",
        "temperature",
        "humidity",
        "rainfall",
        "pressure",
        "wind_speed",
        "cloud_cover",
    ]
    df = df[column_order]

    # Sort by river, zone, and timestamp
    df = df.sort_values(by=["river", "zone", "timestamp"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    print("=" * 70)
    print("FETCHING HISTORICAL WEATHER DATA")
    print("=" * 70)
    print(f"Target : {HISTORICAL_DATA}")
    print(f"Zones  : {sum(len(z) for z in RIVER_ZONES.values())} "
          f"(Krishna: 5, Narmada: 4, Kaveri: 5)")
    print(f"Period : {HISTORY_START_DATE} to {HISTORY_END_DATE} (hourly)")
    print("=" * 70)

    df = build_historical_dataset()

    print(f"\n{'='*70}")
    print(f"Dataset Summary:")
    print(f"  Total rows    : {len(df):,}")
    print(f"  Unique rivers : {df['river'].nunique()}")
    print(f"  Unique zones  : {df['zone'].nunique()}")
    print(f"  Date range    : {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Columns       : {', '.join(df.columns)}")
    print(f"{'='*70}\n")

    # Save to CSV
    df.to_csv(HISTORICAL_DATA, index=False)
    print(f"✓ Saved {HISTORICAL_DATA}")
    print(f"  Size: {len(df):,} rows x {len(df.columns)} columns")

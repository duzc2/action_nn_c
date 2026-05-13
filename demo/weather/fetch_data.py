#!/usr/bin/env python3
"""Fetch historical weather data from Open-Meteo API (free, no API key)."""

import csv
import json
import os
import urllib.request

CITIES = [
    ("beijing",   39.9042, 116.4074),
    ("shanghai",  31.2304, 121.4737),
    ("new_york",  40.7128, -74.0060),
]

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dataset")

DAILY_PARAMS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]


def interpolate_missing(values):
    """Linear interpolation for missing (None) values."""
    result = list(values)
    n = len(result)

    # Forward fill
    last_valid = None
    for i in range(n):
        if result[i] is not None:
            last_valid = result[i]
        elif last_valid is not None:
            result[i] = last_valid

    # Backward fill
    last_valid = None
    for i in range(n - 1, -1, -1):
        if result[i] is not None:
            last_valid = result[i]
        elif last_valid is not None:
            result[i] = last_valid

    return result


def fetch_city(name, lat, lon):
    """Fetch daily weather for one city and write CSV."""
    params = ",".join(DAILY_PARAMS)
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        f"&daily={params}&timezone=auto"
    )

    print(f"Fetching {name} ({lat}, {lon}) ...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    daily = data["daily"]
    dates = daily["time"]
    temp_max = interpolate_missing(daily["temperature_2m_max"])
    temp_min = interpolate_missing(daily["temperature_2m_min"])
    precip   = interpolate_missing(daily["precipitation_sum"])
    wind     = interpolate_missing(daily["wind_speed_10m_max"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, f"{name}.csv")
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "temp_max", "temp_min", "precipitation", "wind_speed"])
        for i in range(len(dates)):
            writer.writerow([
                dates[i],
                f"{temp_max[i]:.2f}" if temp_max[i] is not None else "",
                f"{temp_min[i]:.2f}" if temp_min[i] is not None else "",
                f"{precip[i]:.2f}" if precip[i] is not None else "",
                f"{wind[i]:.2f}" if wind[i] is not None else "",
            ])

    print(f"  -> {len(dates)} records saved to {outpath}")


def main():
    for name, lat, lon in CITIES:
        fetch_city(name, lat, lon)
    print("\nAll cities fetched successfully.")


if __name__ == "__main__":
    main()

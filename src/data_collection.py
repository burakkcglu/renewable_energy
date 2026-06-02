"""
data_collection.py
==================
Downloads weather data from NASA POWER API.

Run: python src/data_collection.py
"""

import os
import time
import requests
import pandas as pd
from tqdm import tqdm

import config as C


def load_provinces(filepath=C.PROVINCES_FILE):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} provinces.")

    required = {"province", "lat", "lon"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns. Expected {required}.")
    return df


def fetch_province_data(name, lat, lon, retries=3, delay=10):
    params = {
        "parameters": C.NASA_PARAMETERS,
        "community": C.NASA_COMMUNITY,
        "longitude": lon,
        "latitude": lat,
        "start": C.NASA_START_DATE,
        "end": C.NASA_END_DATE,
        "format": "JSON",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(C.NASA_BASE_URL, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = delay * (attempt + 2)
                time.sleep(wait)
            else:
                time.sleep(delay)
        except requests.exceptions.RequestException:
            time.sleep(delay)
    print(f"  Download failed: {name}")
    return None


def parse_api_response(data, name):
    try:
        p = data["properties"]["parameter"]
        solar, wind10, wind50, temp = (
            p["ALLSKY_SFC_SW_DWN"], p["WS10M"], p["WS50M"], p["T2M"]
        )
        records = []
        for d in solar.keys():
            def clean(v):
                return None if v == -999.0 else v
            records.append({
                "date": d,
                "solar_radiation": clean(solar[d]),
                "wind_speed": clean(wind10.get(d)),
                "wind_speed_50": clean(wind50.get(d)),
                "temperature": clean(temp.get(d)),
            })
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df["province"] = name
        return df
    except (KeyError, TypeError):
        return None


def fetch_all_provinces(provinces_df, output_dir=C.RAW_DIR):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(C.PROCESSED_DIR, exist_ok=True)
    all_frames, failed = [], []

    for _, row in tqdm(provinces_df.iterrows(), total=len(provinces_df), desc="Fetching data"):
        name, lat, lon = row["province"], row["lat"], row["lon"]
        outfile = os.path.join(output_dir, f"{name}.csv")
        if os.path.exists(outfile):
            all_frames.append(pd.read_csv(outfile, parse_dates=["date"]))
            continue
        raw = fetch_province_data(name, lat, lon)
        if raw is None:
            failed.append(name); continue
        df = parse_api_response(raw, name)
        if df is None:
            failed.append(name); continue
        df.to_csv(outfile, index=False)
        all_frames.append(df)
        time.sleep(2)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        path = os.path.join(C.PROCESSED_DIR, "all_provinces_daily.csv")
        combined.to_csv(path, index=False)
        print(f"\nSaved combined data to: {path}")
    if failed:
        print(f"\nErrors for: {failed}")
    return all_frames, failed


if __name__ == "__main__":
    fetch_all_provinces(load_provinces())
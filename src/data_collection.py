"""
data_collection.py
Fetches solar radiation, wind speed and temperature data
from NASA POWER API for all 81 Turkish provinces.

Usage:
    python src/data_collection.py
"""

import os
import time
import json
import requests
import pandas as pd
from tqdm import tqdm


# --- Config ---
BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = "ALLSKY_SFC_SW_DWN,WS10M,WS50M,T2M"
COMMUNITY = "RE"  # Renewable Energy community
START_DATE = "20040101"
END_DATE = "20241231"
FORMAT = "JSON"

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
PROVINCES_FILE = os.path.join(PROJECT_DIR, "data", "tr_provinces_coords.csv")


def load_provinces(filepath):
    """Load the 81 province coordinates from CSV and sanity-check naming."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} provinces from {filepath}")

    required_cols = {'province', 'lat', 'lon'}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Province CSV missing required columns. "
            f"Need: {required_cols}, found: {set(df.columns)}"
        )
    if len(df) != 81:
        print(f"  ⚠ Warning: Expected 81 provinces, found {len(df)}")
    if df['province'].duplicated().any():
        dupes = df[df['province'].duplicated()]['province'].tolist()
        print(f"  ⚠ Warning: Duplicate provinces: {dupes}")

    return df


def fetch_province_data(province_name, lat, lon, retries=3, delay=10):
    """
    Fetch daily solar, wind and temperature data for one province.
    
    Parameters
    ----------
    province_name : str
    lat : float
    lon : float
    retries : int - number of retry attempts on failure
    delay : int - seconds to wait between retries
    
    Returns
    -------
    dict or None
    """
    params = {
        "parameters": PARAMETERS,
        "community": COMMUNITY,
        "longitude": lon,
        "latitude": lat,
        "start": START_DATE,
        "end": END_DATE,
        "format": FORMAT
    }

    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=60)

            if resp.status_code == 200:
                data = resp.json()
                return data
            elif resp.status_code == 429:
                # Rate limited, wait longer
                wait = delay * (attempt + 2)
                print(f"  Rate limited for {province_name}, "
                      f"waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Error {resp.status_code} for {province_name}, "
                      f"attempt {attempt+1}/{retries}")
                time.sleep(delay)

        except requests.exceptions.RequestException as e:
            print(f"  Connection error for {province_name}: {e}")
            time.sleep(delay)

    print(f"  FAILED: Could not fetch data for {province_name}")
    return None


def parse_api_response(data, province_name):
    """
    Parse NASA POWER JSON response into a clean DataFrame.
    
    Returns DataFrame with columns:
        date, solar_radiation, wind_speed_10m, wind_speed_50m, temperature
    """
    try:
        params = data["properties"]["parameter"]

        solar = params["ALLSKY_SFC_SW_DWN"]  # kWh/m2/day
        wind = params["WS10M"]               # m/s (10 metre)
        wind_50 = params["WS50M"]            # m/s (50 metre)
        temp = params["T2M"]                 # Celsius

        # Build DataFrame from the parameter dicts
        records = []
        for date_str in solar.keys():
            s_val = solar[date_str]
            w_val = wind.get(date_str, None)
            w50_val = wind_50.get(date_str, None) # 50m hızını o gün için al
            t_val = temp.get(date_str, None)

            # NASA uses -999.0 for missing values
            if s_val == -999.0: s_val = None
            if w_val == -999.0: w_val = None
            if w50_val == -999.0: w50_val = None  # 50m için eksik veri kontrolü
            if t_val == -999.0: t_val = None

            records.append({
                "date": date_str,
                "solar_radiation": s_val,
                "wind_speed": w_val,         # 10m hızı
                "wind_speed_50": w50_val,    # 50m hızı eklendi!
                "temperature": t_val
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df["province"] = province_name

        return df

    except (KeyError, TypeError) as e:
        print(f"  Parse error for {province_name}: {e}")
        return None

def fetch_all_provinces(provinces_df, output_dir):
    """
    Fetch data for all 81 provinces and save individual CSVs.
    Also saves a combined CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_frames = []
    failed = []

    # Sanity: warn if province names look ASCII-only (could indicate
    # a non-canonical CSV that will break downstream merges)
    if 'province' in provinces_df.columns:
        sample = provinces_df['province'].astype(str).str.cat(sep=' ')
        if not any(c in sample for c in 'çğıöşüÇĞİÖŞÜ'):
            print("  ⚠ Warning: Province names appear ASCII-only. "
                  "Downstream merges expect Turkish characters (e.g. 'İstanbul').")

    for idx, row in tqdm(provinces_df.iterrows(),
                         total=len(provinces_df),
                         desc="Fetching provinces"):

        name = row["province"]
        lat = row["lat"]
        lon = row["lon"]

        # Check if already downloaded
        outfile = os.path.join(output_dir, f"{name}.csv")
        if os.path.exists(outfile):
            print(f"  {name} already exists, skipping...")
            df = pd.read_csv(outfile, parse_dates=["date"])
            all_frames.append(df)
            continue

        # Fetch from API
        raw_data = fetch_province_data(name, lat, lon)

        if raw_data is None:
            failed.append(name)
            continue

        # Parse response
        df = parse_api_response(raw_data, name)

        if df is None:
            failed.append(name)
            continue

        # Save individual province CSV
        df.to_csv(outfile, index=False)
        all_frames.append(df)

        # Be nice to the API - wait between requests
        time.sleep(2)

    # Combine all provinces into one big CSV
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = os.path.join(
            PROCESSED_DIR, "all_provinces_daily.csv"
        )
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined data saved: {combined_path}")
        print(f"Total rows: {len(combined):,}")
        print(f"Date range: {combined['date'].min()} to "
              f"{combined['date'].max()}")

    if failed:
        print(f"\nFailed provinces ({len(failed)}): {failed}")
    else:
        print("\nAll 81 provinces fetched successfully!")

    return all_frames, failed


def print_data_summary(data_dir):
    """Print a summary of downloaded data."""
    combined_path = os.path.join(PROCESSED_DIR, "all_provinces_daily.csv")

    if not os.path.exists(combined_path):
        print("No combined data found. Run fetch first.")
        return

    df = pd.read_csv(combined_path, parse_dates=["date"])

    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    print(f"Provinces: {df['province'].nunique()}")
    print(f"Date range: {df['date'].min().date()} to "
          f"{df['date'].max().date()}")
    print(f"Total rows: {len(df):,}")
    print(f"\nMissing values:")
    print(df[["solar_radiation", "wind_speed","wind_speed_50", "temperature"]]
          .isnull().sum())
    print(f"\nSolar radiation (kWh/m2/day):")
    print(df["solar_radiation"].describe().round(2))
    print(f"\nWind speed (m/s):")
    print(df["wind_speed"].describe().round(2))
    print(f"\nWind speed 50m (m/s):") # Eklendi
    print(df["wind_speed_50"].describe().round(2)) # Eklendi
    print(f"\nTemperature (C):")
    print(df["temperature"].describe().round(2))


if __name__ == "__main__":
    print("="*50)
    print("NASA POWER Data Collection")
    print("Turkey 81 Provinces - Solar & Wind")
    print("="*50)

    # Load province coordinates
    provinces = load_provinces(PROVINCES_FILE)

    # Fetch all data
    frames, failed = fetch_all_provinces(provinces, RAW_DIR)

    # Print summary
    print_data_summary(RAW_DIR)

"""
epias_demand.py
===============
Downloads monthly national electricity demand from EPIAS.
Outputs a normalized demand profile.
"""

import os
import time
import requests
import pandas as pd

import config as C

USERNAME = ""
PASSWORD = ""

CAS_URL = "https://giris.epias.com.tr/cas/v1/tickets"
CONSUMPTION_URL = ("https://seffaflik.epias.com.tr/electricity-service"
                   "/v1/consumption/data/consumption-quantity")

YEARS = [2021, 2022, 2023, 2024]


def get_tgt():
    r = requests.post(CAS_URL,
                      data={"username": USERNAME, "password": PASSWORD},
                      headers={"Content-Type": "application/x-www-form-urlencoded",
                               "Accept": "text/plain"})
    r.raise_for_status()
    return r.text.strip()


def fetch_month(tgt, year, month):
    period = f"{year}-{month:02d}-01T00:00:00+03:00"
    r = requests.post(CONSUMPTION_URL,
                      headers={"TGT": tgt, "Content-Type": "application/json",
                               "Accept": "application/json"},
                      json={"startDate": period, "endDate": period, "period": period})
    r.raise_for_status()
    items = r.json().get("items", [])
    
    total = 0.0
    for row in items:
        for k in ("serbestTuketiciTuketimMiktari",
                  "stHakkiBulunmayanTuketiciTuketimMiktari",
                  "stHakkiniKullanmayanTuketiciTuketimMiktari"):
            total += float(row.get(k) or 0)
    return total


def build_demand_profile():
    tgt = get_tgt()
    records = []
    for year in YEARS:
        for month in range(1, 13):
            try:
                mwh = fetch_month(tgt, year, month)
                records.append({"year": year, "month": month, "consumption_mwh": mwh})
                time.sleep(0.5)
            except Exception as e:
                print(f"  Request failed for {year}-{month:02d}")
                
    df = pd.DataFrame(records)

    # Filter out API anomalies (like extreme spikes in early 2021)
    med = df["consumption_mwh"].median()
    clean = df[df["consumption_mwh"] < 3 * med]
    
    monthly_avg = clean.groupby("month")["consumption_mwh"].mean()
    profile = (monthly_avg / monthly_avg.sum()).reset_index()
    profile.columns = ["month", "demand_share"]

    os.makedirs(C.PROCESSED_DIR, exist_ok=True)
    out = os.path.join(C.PROCESSED_DIR, "demand_profile.csv")
    profile.to_csv(out, index=False)
    print(f"Profile saved to {out}")
    return profile


if __name__ == "__main__":
    build_demand_profile()
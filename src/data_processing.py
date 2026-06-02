"""
data_processing.py
==================
Calculates capacity factors and builds the covariance matrix.
"""

import os
import numpy as np
import pandas as pd

import config as C


def solar_capacity_factor(ghi, temp,
                          performance_ratio=C.SOLAR_PERFORMANCE_RATIO,
                          temp_coeff=C.SOLAR_TEMP_COEFF,
                          t_ref=C.SOLAR_T_REF):
    # Calculate daily solar CF
    temp_correction = 1.0 + temp_coeff * (temp - t_ref)
    temp_correction = np.clip(temp_correction, 0.7, 1.05)
    cf = (ghi * performance_ratio) / 24.0 * temp_correction
    return np.clip(cf, 0.0, 1.0)


def wind_capacity_factor(ws_10m, ws_50m,
                         cut_in=C.WIND_CUT_IN, rated=C.WIND_RATED,
                         cut_out=C.WIND_CUT_OUT, hub_height=C.WIND_HUB_HEIGHT,
                         ref_height=C.WIND_REF_HEIGHT,
                         alpha_min=C.WIND_ALPHA_MIN, alpha_max=C.WIND_ALPHA_MAX):
    # Calculate daily wind CF using dynamic shear
    ws_10 = np.maximum(ws_10m, 0.1)
    ws_50 = np.maximum(ws_50m, 0.1)

    alpha = np.log(ws_50 / ws_10) / np.log(50.0 / 10.0)
    alpha = np.clip(alpha, alpha_min, alpha_max)

    v = ws_50 * ((hub_height / ref_height) ** alpha)

    # Cubic power curve
    cf = np.where(
        v < cut_in, 0.0,
        np.where(
            v < rated,
            (v ** 3 - cut_in ** 3) / (rated ** 3 - cut_in ** 3),
            np.where(v <= cut_out, 1.0, 0.0),
        ),
    )
    return np.clip(cf, 0.0, 1.0)


def compute_covariance_matrix(df, frequency="monthly"):
    # Generate covariance matrix for optimization
    provinces = sorted(df["province"].unique())
    frames = []
    for prov in provinces:
        pv = df[df["province"] == prov][["date", "solar_cf", "wind_cf"]].set_index("date")
        pv = pv.rename(columns={"solar_cf": f"{prov}_solar", "wind_cf": f"{prov}_wind"})
        frames.append(pv)

    wide = frames[0]
    for f in frames[1:]:
        wide = wide.join(f, how="outer")
    wide = wide.dropna()

    if frequency == "monthly":
        wide.index = pd.to_datetime(wide.index)
        wide_agg = wide.resample("ME").mean().dropna()
    else:
        wide_agg = wide

    asset_names = list(wide_agg.columns)
    mu_vector = wide_agg.mean().values
    cov_matrix = wide_agg.cov().values + np.eye(len(asset_names)) * 1e-10

    min_eig = np.linalg.eigvalsh(cov_matrix).min()
    return cov_matrix, mu_vector, asset_names, wide_agg


def process_all_data():
    input_path = os.path.join(C.PROCESSED_DIR, "all_provinces_daily.csv")
    if not os.path.exists(input_path):
        print("Data not found. Run data_collection.py first.")
        return None

    df = pd.read_csv(input_path, parse_dates=["date"])
    df = df.dropna(subset=["solar_radiation", "wind_speed", "wind_speed_50", "temperature"])

    df["solar_cf"] = solar_capacity_factor(df["solar_radiation"].values, df["temperature"].values)
    df["wind_cf"] = wind_capacity_factor(df["wind_speed"].values, df["wind_speed_50"].values)

    os.makedirs(C.PROCESSED_DIR, exist_ok=True)
    df.to_csv(os.path.join(C.PROCESSED_DIR, "daily_with_cf.csv"), index=False)

    summary = df.groupby("province").agg(
        solar_cf_mean=("solar_cf", "mean"), solar_cf_std=("solar_cf", "std"),
        wind_cf_mean=("wind_cf", "mean"), wind_cf_std=("wind_cf", "std"),
    ).round(4).reset_index()
    summary.to_csv(os.path.join(C.PROCESSED_DIR, "province_summary.csv"), index=False)

    cov, mu, assets, monthly_cf = compute_covariance_matrix(df, "monthly")
    pd.DataFrame(monthly_cf).to_csv(os.path.join(C.PROCESSED_DIR, "monthly_cf_matrix.csv"))
    pd.DataFrame(cov, index=assets, columns=assets).to_csv(
        os.path.join(C.PROCESSED_DIR, "covariance_matrix.csv"))
    pd.DataFrame({"asset": assets, "expected_cf": mu}).to_csv(
        os.path.join(C.PROCESSED_DIR, "mean_vector.csv"), index=False)

    _build_capacity_bounds(assets)
    print("Processing complete.")
    return df, summary, cov, mu, assets


def _build_capacity_bounds(asset_names):
    pf_path = os.path.join(C.PROCESSED_DIR, "province_features.csv")
    if not os.path.exists(pf_path):
        return
    pf = pd.read_csv(pf_path)
    bounds_by_class = {
        "high":   {"solar_mw": 500.0,  "wind_mw": 300.0},
        "medium": {"solar_mw": 1500.0, "wind_mw": 1000.0},
        "low":    {"solar_mw": 3000.0, "wind_mw": 2000.0},
    }
    rows = []
    for asset in asset_names:
        province, source = asset.rsplit("_", 1)
        row = pf[pf["province"] == province]
        cls = "medium" if len(row) == 0 else row.iloc[0].get("intensity_class", "medium")
        if cls not in bounds_by_class:
            cls = "medium"
        rows.append({"asset": asset, "province": province, "source": source,
                     "intensity_class": cls, "upper_mw": bounds_by_class[cls][f"{source}_mw"]})
    pd.DataFrame(rows).to_csv(os.path.join(C.PROCESSED_DIR, "capacity_bounds.csv"), index=False)


if __name__ == "__main__":
    process_all_data()
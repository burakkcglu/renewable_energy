"""
opt_common.py
=============
Shared loading/setup for all three optimizers so they see IDENTICAL data,
units, costs, bounds and constraints. This is what makes the three-method
comparison fair (the earlier versions re-derived these per file and drifted).

UNITS (hybrid choice = notebook's absolute system):
  mu_mwh[i] = mean_CF[i] * 8760   -> MWh/year produced per MW installed
  cost      = COST_*_PER_MW * MW  -> absolute USD
  x[i]      -> MW installed for asset i
So production reads in MWh/year (divide by 1e6 -> TWh) and cost in USD
(divide by 1e9 -> $B), with NO hidden scale factors.
"""

import os
import numpy as np
import pandas as pd

import config as C


def normalize_name(text):
    return text.translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosucgiosu")).lower().strip()

def load_monthly_profiles():
    """
    Returns (cf_by_month, demand_by_month) for Step-2 demand matching:
      cf_by_month   : (12, n_assets) mean CF for each calendar month
      demand_by_month: (12,) MWh/year-equivalent demand target per month
                       = annual D_target * demand_share[month]
    Falls back to None if the required files are missing (so the optimizers
    can still run in annual-only mode).
    """
    cf_path = os.path.join(C.PROCESSED_DIR, "monthly_cf_matrix.csv")
    dp_path = os.path.join(C.PROCESSED_DIR, "demand_profile.csv")
    if not (os.path.exists(cf_path) and os.path.exists(dp_path)):
        return None, None

    cf = pd.read_csv(cf_path, index_col=0)
    cf.index = pd.to_datetime(cf.index)
    # Takvim ayına göre ortalama CF -> (12, n_assets)
    cf_by_month = cf.groupby(cf.index.month).mean()
    cf_by_month = cf_by_month.reindex(range(1, 13)).values  # (12, n)

    dp = pd.read_csv(dp_path).sort_values("month")
    shares = dp["demand_share"].values                       # (12,)
    D_annual = C.annual_demand_target_mwh()
    demand_by_month = D_annual * shares                      # (12,) MWh

    return cf_by_month, demand_by_month

def load_problem():
    """
    Returns a dict with everything the optimizers need:
      cov (162x162), mu_mwh (162,), cost_vec (162,), asset_names,
      lower/upper bounds, regional_indices, D_target (MWh/yr), var_scale.
    """
    cov = pd.read_csv(os.path.join(C.PROCESSED_DIR, "covariance_matrix.csv"),
                      index_col=0).values.astype(float)
    mu_df = pd.read_csv(os.path.join(C.PROCESSED_DIR, "mean_vector.csv"))

    if "asset" in mu_df.columns:
        asset_names = list(mu_df["asset"].values)
        mean_col = [c for c in mu_df.columns if c != "asset"][0]
        mu_cf = mu_df[mean_col].values.astype(float)
    else:
        asset_names = list(mu_df.iloc[:, 0].values)
        mu_cf = mu_df.iloc[:, 1].values.astype(float)

    # CF (dimensionless) -> MWh/year per MW installed
    mu_mwh = mu_cf * C.HOURS_PER_YEAR

    cost_vec = np.array([
        C.COST_SOLAR_PER_MW if "solar" in a else C.COST_WIND_PER_MW
        for a in asset_names
    ])

    # Bounds
    bounds_path = os.path.join(C.PROCESSED_DIR, "capacity_bounds.csv")
    upper = []
    if os.path.exists(bounds_path):
        bdf = pd.read_csv(bounds_path, index_col=0)
        for a in asset_names:
            try:
                upper.append(float(bdf.loc[a, "upper_mw"]))
            except KeyError:
                upper.append(500.0 if "solar" in a else 300.0)
    else:
        upper = [500.0 if "solar" in a else 300.0 for a in asset_names]
    upper = np.array(upper)
    lower = np.zeros(len(asset_names))

    # Regional indices from province_features
    regional_indices = _load_regions(asset_names)

    # Demand target in MWh/year
    D_target = C.annual_demand_target_mwh()

    # Variance scale: calibrate from Sigma so normalized risk ~ [0,1].
    # Upper-bound full-deployment variance is a safe, data-driven scale.
    var_scale = float(upper @ cov @ upper)
    if not np.isfinite(var_scale) or var_scale <= 0:
        var_scale = C.VAR_SCALE_FALLBACK
    # CO2 scale: avoided CO2 at full deployment, so the 3rd objective normalizes
    # to roughly the same [0,1] range as risk and cost.
    co2_scale = float(C.GRID_EMISSION_FACTOR * (mu_mwh @ upper))
    if not np.isfinite(co2_scale) or co2_scale <= 0:
        co2_scale = 1.0

    return {
        "cov": cov, "mu_mwh": mu_mwh, "cost_vec": cost_vec,
        "asset_names": asset_names, "lower": lower, "upper": upper,
        "regional_indices": regional_indices, "D_target": D_target,
        "var_scale": var_scale, "co2_scale": co2_scale, "n": len(asset_names),
    }


def _load_regions(asset_names):
    pf_path = os.path.join(C.PROCESSED_DIR, "province_features.csv")
    regional = {}
    pf = None
    if os.path.exists(pf_path):
        pf = pd.read_csv(pf_path).drop_duplicates(subset=["province"], keep="last")
        pf["norm_prov"] = pf["province"].apply(normalize_name)
        pf = pf.set_index("norm_prov")
    for i, asset in enumerate(asset_names):
        prov = normalize_name(asset.split("_")[0])
        region = "Unknown"
        if pf is not None:
            try:
                region = pf.loc[prov, "region"]
                if isinstance(region, pd.Series):
                    region = region.iloc[0]
            except KeyError:
                region = "Unknown"
        regional.setdefault(region, []).append(i)
    return regional


def production(x, mu_mwh):
    return float(mu_mwh @ x)

def cost(x, cost_vec):
    return float(cost_vec @ x)

def variance(x, cov):
    return float(x @ cov @ x)

def avoided_co2(x, mu_mwh):
    return float(C.GRID_EMISSION_FACTOR * (mu_mwh @ x))
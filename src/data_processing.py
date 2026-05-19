"""
data_processing.py
Converts raw NASA POWER data into capacity factors,
computes the covariance matrix for portfolio optimization.

Usage:
    python src/data_processing.py
"""

import os
import numpy as np
import pandas as pd


# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")


# --- Solar capacity factor ---

def solar_capacity_factor(ghi, temp, panel_efficiency=0.18,
                          temp_coeff=-0.004, t_ref=25.0):
    """
    Convert Global Horizontal Irradiance to solar capacity factor.
    
    Parameters
    ----------
    ghi : float or array
        Solar radiation in kWh/m2/day
    temp : float or array
        Ambient temperature in Celsius
    panel_efficiency : float
        Panel efficiency at standard test conditions (default 18%)
    temp_coeff : float
        Power temperature coefficient (%/C, typical -0.4%)
    t_ref : float
        Reference temperature for STC (25 C)
    
    Returns
    -------
    float or array
        Capacity factor between 0 and 1
    """
    # Solar capacity factor methodology:
    #
    # GHI (Global Horizontal Irradiance) is energy in kWh/m²/day.
    # Theoretical maximum is roughly 10 kWh/m²/day (very high-irradiance
    # locations achieve 7-8 in summer; we use 10 as a normalization ceiling).
    #
    # Note: We do NOT multiply by η here, because rated panel capacity (MW)
    # is already defined at η efficiency (datasheet at STC). Multiplying
    # would double-count the efficiency factor — this is the η-double-counting
    # bug that some implementations make.
    #
    # CF = (GHI / GHI_max) × temperature_correction
    # where temperature_correction handles the -0.4%/°C derating above 25°C.

    # Temperature correction
    temp_correction = 1 + temp_coeff * (temp - t_ref)
    temp_correction = np.clip(temp_correction, 0.5, 1.2)

    # Capacity factor: GHI normalized by peak possible
    # Peak daily irradiance roughly 7-8 kWh/m2/day for best locations
    # We normalize by theoretical max (about 10 kWh/m2/day)
    cf = (ghi / 10.0) * temp_correction

    # Clip to valid range
    cf = np.clip(cf, 0.0, 1.0)

    return cf


# --- Wind capacity factor ---

def wind_capacity_factor(wind_speed_10m, wind_speed_50m, cut_in=3.0, rated=12.0,
                         cut_out=25.0, hub_height=80.0):
    """
    Convert wind speed to capacity factor using a DYNAMIC wind shear exponent.

    Methodology:
    -----------
    Most renewable-energy papers use a fixed alpha = 1/7 ≈ 0.143 (open
    terrain rule). This is a coarse approximation that ignores actual
    atmospheric conditions: alpha varies from ~0.10 (smooth open sea)
    to ~0.40 (urban/forested terrain), and changes with stability,
    season, and time of day.

    We compute alpha PER DATAPOINT from NASA's two wind speeds:
        v(h) = v_ref · (h / h_ref)^alpha
    Taking the ratio at 50m vs 10m:
        v50 / v10 = (50/10)^alpha
        alpha = log(v50/v10) / log(50/10) = log(v50/v10) / log(5)

    This makes our wind CFs physically more faithful to each province's
    actual terrain and atmospheric profile.
    """
    # Sıfıra bölme hatasını engellemek için çok küçük rüzgarları 0.1 m/s yapıyoruz
    ws_10 = np.maximum(wind_speed_10m, 0.1)
    ws_50 = np.maximum(wind_speed_50m, 0.1)
    
    # Dinamik Wind Shear Exponent (Alpha) hesabı
    alpha = np.log(ws_50 / ws_10) / np.log(50.0 / 10.0)
    alpha = np.clip(alpha, 0.1, 0.4) 
    
    # Hızı 50 metreden türbin göbek yüksekliğine (80m) çıkart
    ws_hub = ws_50 * ((hub_height / 50.0) ** alpha)

    # Simplified power curve
    cf = np.where(
        ws_hub < cut_in, 0.0,
        np.where(
            ws_hub < rated,
            ((ws_hub - cut_in) / (rated - cut_in)) ** 3,
            np.where(ws_hub < cut_out, 1.0, 0.0)
        )
    )

    return np.clip(cf, 0.0, 1.0)


def process_all_data():
    """
    Main processing pipeline:
    1. Load combined raw data
    2. Calculate capacity factors
    3. Create monthly averages per province
    4. Compute covariance matrix
    5. Save everything
    """
    input_path = os.path.join(PROCESSED_DIR, "all_provinces_daily.csv")

    if not os.path.exists(input_path):
        print("Error: Run data_collection.py first!")
        return

    print("Loading data...")
    df = pd.read_csv(input_path, parse_dates=["date"])

    # Drop rows with missing values
    initial_len = len(df)
    df = df.dropna(subset=["solar_radiation", "wind_speed","wind_speed_50", "temperature"])
    print(f"Dropped {initial_len - len(df)} rows with missing values "
          f"({len(df)} remaining)")

    # --- Calculate capacity factors ---
    print("Calculating solar capacity factors...")
    df["solar_cf"] = solar_capacity_factor(
        df["solar_radiation"].values,
        df["temperature"].values
    )

    print("Calculating wind capacity factors...")
    df["wind_cf"] = wind_capacity_factor(
        df["wind_speed"].values,
        df["wind_speed_50"].values
    )

    # --- Monthly averages per province ---
    print("Computing monthly averages...")
    df["year_month"] = df["date"].dt.to_period("M")

    monthly = df.groupby(["province", "year_month"]).agg({
        "solar_cf": "mean",
        "wind_cf": "mean",
        "solar_radiation": "mean",
        "wind_speed": "mean",
        "wind_speed_50": "mean",
        "temperature": "mean"
    }).reset_index()

    monthly["year_month"] = monthly["year_month"].astype(str)

    # Save daily with capacity factors
    daily_path = os.path.join(PROCESSED_DIR, "daily_with_cf.csv")
    df.to_csv(daily_path, index=False)
    print(f"Daily data saved: {daily_path}")

    # Save monthly averages
    monthly_path = os.path.join(PROCESSED_DIR, "monthly_averages.csv")
    monthly.to_csv(monthly_path, index=False)
    print(f"Monthly averages saved: {monthly_path}")

    # --- Province-level summary statistics ---
    print("Computing province summaries...")
    summary = df.groupby("province").agg({
        "solar_cf": ["mean", "std"],
        "wind_cf": ["mean", "std"],
        "solar_radiation": "mean",
        "wind_speed": "mean",
        "wind_speed_50": "mean",
        "temperature": "mean"
    }).round(4)

    summary.columns = [
        "solar_cf_mean", "solar_cf_std",
        "wind_cf_mean", "wind_cf_std",
        "avg_solar_radiation", "avg_wind_speed", "avg_wind_speed_50", "avg_temperature" 
    ]
    summary = summary.reset_index()

    summary_path = os.path.join(PROCESSED_DIR, "province_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Province summary saved: {summary_path}")

    # --- Covariance matrix (monthly aggregation for stability) ---
    print("Computing covariance matrix (162x162, monthly basis)...")
    cov_matrix, mu_vector, asset_names, monthly_cf_matrix = compute_covariance_matrix(
        df, frequency='monthly'
    )
    # Save monthly CF matrix for Monte Carlo robustness analysis
    monthly_cf_path = os.path.join(PROCESSED_DIR, "monthly_cf_matrix.csv")
    monthly_cf_matrix.to_csv(monthly_cf_path)
    print(f"Monthly CF matrix saved: {monthly_cf_matrix.shape} -> {monthly_cf_path}")

    # Daily CF matrix (for Monte Carlo with finer weather noise)
    daily_cf_path = os.path.join(PROCESSED_DIR, "daily_cf_matrix.csv")
    # Re-build daily wide format quickly (avoid recomputation)
    daily_wide_path = daily_cf_path  # placeholder
    # Sadece monthly tutuyoruz, daily zaten daily_with_cf.csv'de var (long format)

    # Save covariance matrix
    cov_df = pd.DataFrame(cov_matrix,
                          index=asset_names, columns=asset_names)
    cov_path = os.path.join(PROCESSED_DIR, "covariance_matrix.csv")
    cov_df.to_csv(cov_path)
    print(f"Covariance matrix saved: {cov_path}")

    # Save mean vector
    mu_df = pd.DataFrame({
        "asset": asset_names,
        "expected_cf": mu_vector
    })
    mu_path = os.path.join(PROCESSED_DIR, "mean_vector.csv")
    mu_df.to_csv(mu_path, index=False)
    print(f"Mean vector saved: {mu_path}")

    # Print summary
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Provinces: {df['province'].nunique()}")
    print(f"Assets (province-source pairs): {len(asset_names)}")
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    print(f"\nSolar CF across provinces:")
    print(f"  Best:  {summary['solar_cf_mean'].max():.3f} "
          f"({summary.loc[summary['solar_cf_mean'].idxmax(), 'province']})")
    print(f"  Worst: {summary['solar_cf_mean'].min():.3f} "
          f"({summary.loc[summary['solar_cf_mean'].idxmin(), 'province']})")
    print(f"\nWind CF across provinces:")
    print(f"  Best:  {summary['wind_cf_mean'].max():.3f} "
          f"({summary.loc[summary['wind_cf_mean'].idxmax(), 'province']})")
    print(f"  Worst: {summary['wind_cf_mean'].min():.3f} "
          f"({summary.loc[summary['wind_cf_mean'].idxmin(), 'province']})")

    # --- Density-based capacity upper bounds ---
    print("Computing density-based capacity upper bounds...")
    pf_path = os.path.join(PROCESSED_DIR, "province_features.csv")
    if not os.path.exists(pf_path):
        print(f"  ⚠ Warning: {pf_path} not found. Skipping bounds generation.")
    else:
        pf = pd.read_csv(pf_path)

        # Upper bound mapping: tighter for dense provinces (less open land)
        BOUNDS_BY_CLASS = {
            'high':   {'solar_mw': 500.0,  'wind_mw': 300.0},   # crowded metropolis
            'medium': {'solar_mw': 1500.0, 'wind_mw': 1000.0},  # mid-density province
            'low':    {'solar_mw': 3000.0, 'wind_mw': 2000.0},  # rural / spacious
        }

        bounds_rows = []
        for asset in asset_names:
            # asset is like 'Adana_solar' or 'Bursa_wind'
            province, source = asset.rsplit('_', 1)
            row = pf[pf['province'] == province]
            if len(row) == 0:
                # Fallback if province name mismatch
                print(f"  ⚠ No features for {province}, using medium class default")
                cls = 'medium'
            else:
                cls = row.iloc[0]['intensity_class']
            mw = BOUNDS_BY_CLASS[cls][f'{source}_mw']
            bounds_rows.append({
                'asset': asset,
                'province': province,
                'source': source,
                'intensity_class': cls,
                'upper_mw': mw,
            })

        bounds_df = pd.DataFrame(bounds_rows)
        bounds_path = os.path.join(PROCESSED_DIR, "capacity_bounds.csv")
        bounds_df.to_csv(bounds_path, index=False)
        print(f"Capacity bounds saved: {bounds_path}")
        print(f"  Class distribution:")
        print(bounds_df.groupby(['intensity_class', 'source'])['upper_mw']
              .first().to_string())
    return df, summary, cov_matrix, mu_vector, asset_names


def compute_covariance_matrix(df, frequency='monthly'):
    """
    Build the 162x162 covariance matrix from capacity factor time series.

    Parameters
    ----------
    df : DataFrame with columns date, province, solar_cf, wind_cf
    frequency : 'monthly' (default) or 'daily'
        Monthly aggregation gives a more stable covariance structure for
        portfolio optimization — daily values are dominated by short-term
        weather noise that doesn't matter for capacity planning.

    Returns
    -------
    cov_matrix : np.array (162, 162)
    mu_vector  : np.array (162,)
    asset_names : list of str (length 162)
    monthly_cf_matrix : DataFrame (T × 162) — kept for Monte Carlo simulation
    """
    provinces = sorted(df["province"].unique())

    # Build wide-format DataFrame: rows=dates, columns=(province, source) pairs
    pivot_frames = []
    for prov in provinces:
        prov_data = df[df["province"] == prov][["date", "solar_cf", "wind_cf"]]
        prov_data = prov_data.set_index("date")
        prov_data = prov_data.rename(columns={
            "solar_cf": f"{prov}_solar",
            "wind_cf": f"{prov}_wind"
        })
        pivot_frames.append(prov_data)

    wide = pivot_frames[0]
    for frame in pivot_frames[1:]:
        wide = wide.join(frame, how="outer")
    wide = wide.dropna()

    # Monthly aggregation: take mean CF for each (year, month, asset)
    if frequency == 'monthly':
        wide.index = pd.to_datetime(wide.index)
        wide_agg = wide.resample('ME').mean().dropna()
    else:
        wide_agg = wide

    asset_names = list(wide_agg.columns)
    mu_vector = wide_agg.mean().values
    cov_matrix = wide_agg.cov().values

    # Numerical stabilization: add tiny diagonal term to ensure PSD
    cov_matrix = cov_matrix + np.eye(len(asset_names)) * 1e-10

    # Verify positive semi-definite
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    min_eig = eigenvalues.min()
    if min_eig < -1e-8:
        print(f"  ⚠ Warning: covariance matrix has negative eigenvalue {min_eig:.2e}")
    else:
        print(f"  ✓ Covariance matrix is PSD (min eigenvalue: {min_eig:.2e})")

    return cov_matrix, mu_vector, asset_names, wide_agg




if __name__ == "__main__":
    process_all_data()

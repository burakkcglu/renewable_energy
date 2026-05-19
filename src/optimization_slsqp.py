import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

# Paths & Params
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

COST_SOLAR_MW = 0.8
COST_WIND_MW = 1.2
BUDGET = 50000.0  # Milyon USD
MIN_REGIONAL_MW = 500.0

# Ölçek Normalizasyonu
# Calibrated empirically from observed solution magnitudes:
#   - pure-cost solution: cost ≈ 17000, risk ≈ 1.2e7
#   - pure-risk solution: cost ≈ 50000, risk ≈ 4-6e6
# Scaling both objectives to roughly [0,1] makes λ interpretable.
VAR_SCALE = 1.2e7
COST_SCALE = BUDGET  # = 50000


def normalize_name(text):
    return text.translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosucgiosu")).lower().strip()


def load_optimization_data():
    cov_matrix = pd.read_csv(
        os.path.join(PROCESSED_DIR, "covariance_matrix.csv"), index_col=0
    ).values
    mu_df = pd.read_csv(os.path.join(PROCESSED_DIR, "mean_vector.csv"))

    if 'asset' in mu_df.columns:
        asset_names = mu_df['asset'].values
        mean_col = [c for c in mu_df.columns if c != 'asset'][0]
        mu_vector = mu_df[mean_col].values
    else:
        asset_names = mu_df.iloc[:, 0].values
        mu_vector = mu_df.iloc[:, 1].values

    return cov_matrix, mu_vector, list(asset_names)


def create_cost_vector(asset_names):
    return np.array(
        [COST_SOLAR_MW if 'solar' in asset else COST_WIND_MW for asset in asset_names]
    )


def objective_function(x, lambda_val, cov_matrix, cost_vector):
    risk = np.dot(x.T, np.dot(cov_matrix, x))
    cost = np.dot(cost_vector, x)

    normalized_risk = risk / VAR_SCALE
    normalized_cost = cost / COST_SCALE

    return lambda_val * normalized_risk + (1 - lambda_val) * normalized_cost


def run_slsqp():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cov_matrix, mu_vector, asset_names = load_optimization_data()
    n_vars = len(asset_names)

    # --- TÜİK Verisi ---
    df_features = pd.read_csv(os.path.join(PROCESSED_DIR, "province_features.csv"))
    df_features = df_features.drop_duplicates(subset=['province'], keep='last')
    df_features['norm_prov'] = df_features['province'].apply(normalize_name)
    df_features = df_features.set_index('norm_prov')

    total_demand_mwh = df_features['total_demand_mwh'].sum()
    TARGET_DEMAND = (total_demand_mwh / (365 * 24)) * 0.25

    cost_vector = create_cost_vector(asset_names)

    # --- Hazır Bounds Verisi ---
    bounds_df = pd.read_csv(
        os.path.join(PROCESSED_DIR, "capacity_bounds.csv"), index_col=0
    )
    bounds = []
    x0_max = []
    for asset in asset_names:
        try:
            upper_limit = bounds_df.loc[asset, 'upper_mw']
        except KeyError:
            upper_limit = 500.0 if 'solar' in asset else 300.0

        bounds.append((0, upper_limit))
        x0_max.append(upper_limit)

    bounds = tuple(bounds)
    x0_max = np.array(x0_max)

    # --- Regional Balance Constraint ---
    # Bölge bilgisini doğrudan province_features içinden çekiyoruz.
    regional_indices = {}
    for i, asset in enumerate(asset_names):
        prov_name = normalize_name(asset.split('_')[0])
        try:
            region = df_features.loc[prov_name, 'region']
            if isinstance(region, pd.Series):
                region = region.iloc[0]
        except KeyError:
            region = 'Unknown'

        if region not in regional_indices:
            regional_indices[region] = []
        regional_indices[region].append(i)

    # Kısıtlar
    cons = [
        {'type': 'ineq', 'fun': lambda x: np.dot(mu_vector, x) - TARGET_DEMAND},
        {'type': 'ineq', 'fun': lambda x: BUDGET - np.dot(cost_vector, x)},
    ]

    for region, indices in regional_indices.items():
        cons.append({
            'type': 'ineq',
            'fun': lambda x, idxs=indices: np.sum(x[idxs]) - MIN_REGIONAL_MW,
        })

    # --- Multi-Start Initial Conditions ---
    x0_zero = np.zeros(n_vars)
    results = []
    lambdas = np.linspace(0.01, 0.99, 20)

    print(f"Tracing Efficient Frontier with Dynamic Demand: {TARGET_DEMAND:.2f} MW...")

    np.random.seed(42)  # reproducibility
    for l in tqdm(lambdas, desc="Optimizing SLSQP"):
        candidates = []

        # Start 1: zeros (favors small-capacity solutions)
        res = minimize(
            objective_function, x0_zero, args=(l, cov_matrix, cost_vector),
            method='SLSQP', bounds=bounds, constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-7},
        )
        if res.success:
            candidates.append(res)

        # Start 2: max bounds (favors large-capacity solutions)
        res = minimize(
            objective_function, x0_max, args=(l, cov_matrix, cost_vector),
            method='SLSQP', bounds=bounds, constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-7},
        )
        if res.success:
            candidates.append(res)

        # Start 3: balanced — proportional to mu (favors productive assets)
        x0_mu = (BUDGET / np.dot(cost_vector, mu_vector / mu_vector.max())) * (
            mu_vector / mu_vector.max()
        )
        x0_mu = np.clip(x0_mu, 0, x0_max)
        res = minimize(
            objective_function, x0_mu, args=(l, cov_matrix, cost_vector),
            method='SLSQP', bounds=bounds, constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-7},
        )
        if res.success:
            candidates.append(res)

        # Start 4: random within bounds (escape pathological local minima)
        x0_rand = np.random.uniform(0, x0_max * 0.5)
        res = minimize(
            objective_function, x0_rand, args=(l, cov_matrix, cost_vector),
            method='SLSQP', bounds=bounds, constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-7},
        )
        if res.success:
            candidates.append(res)

        # Pick the best among successful candidates
        best_res = min(candidates, key=lambda r: r.fun) if candidates else None

        if best_res:
            p_risk = np.dot(best_res.x.T, np.dot(cov_matrix, best_res.x))
            p_cost = np.dot(cost_vector, best_res.x)
            p_prod = np.dot(mu_vector, best_res.x)
            row = [round(l, 3), p_cost, p_risk, p_prod] + list(best_res.x)
            results.append(row)
        else:
            print(f"Warning: No feasible solution found for lambda={l:.2f}")

    # ---------- Pareto Post-Processing ----------
    # The multi-start optimizer can land on different local optima for
    # different λ. The TRUE efficient frontier should be monotonic:
    # higher λ → lower risk, higher cost. We filter out dominated points
    # (any point that's both more expensive AND riskier than another).
    cols = ['lambda', 'total_cost', 'total_risk', 'expected_production'] + list(asset_names)
    df_raw = pd.DataFrame(results, columns=cols)

    # Sort by cost ascending
    df_sorted = df_raw.sort_values('total_cost').reset_index(drop=True)

    # Keep only points where risk strictly decreases as cost increases
    keep_idx = [0]  # always keep the cheapest point
    min_risk_so_far = df_sorted.loc[0, 'total_risk']
    for i in range(1, len(df_sorted)):
        if df_sorted.loc[i, 'total_risk'] < min_risk_so_far:
            keep_idx.append(i)
            min_risk_so_far = df_sorted.loc[i, 'total_risk']

    df_out = df_sorted.loc[keep_idx].reset_index(drop=True)
    n_dropped = len(df_raw) - len(df_out)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} dominated points "
              f"(kept {len(df_out)} on Pareto frontier).")

    out_path = os.path.join(RESULTS_DIR, "efficient_frontier_slsqp.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nSLSQP Complete! Pareto frontier has {len(df_out)} points. "
          f"Saved to {out_path}")


if __name__ == "__main__":
    run_slsqp()
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

# Paths & Params
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results") # RAW_DIR'i kaldırdım, gerek kalmadı

COST_SOLAR_MW = 0.8
COST_WIND_MW = 1.2
BUDGET = 50000.0  # Milyon USD
MIN_REGIONAL_MW = 500.0

# Ölçek Normalizasyonu (DÜZELTİLDİ)
VAR_SCALE = 1e7     # Eskiden 1e-3 idi. Şimdi 10 milyona bölerek [0,1] aralığına çekiyoruz.
COST_SCALE = BUDGET

def normalize_name(text):
    return text.translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosucgiosu")).lower().strip()

def load_optimization_data():
    cov_matrix = pd.read_csv(os.path.join(PROCESSED_DIR, "covariance_matrix.csv"), index_col=0).values
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
    return np.array([COST_SOLAR_MW if 'solar' in asset else COST_WIND_MW for asset in asset_names])

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
    df_features['norm_prov'] = df_features['province'].apply(normalize_name)
    df_features = df_features.set_index('norm_prov')
    
    total_demand_mwh = df_features['total_demand_mwh'].sum()
    TARGET_DEMAND = (total_demand_mwh / (365 * 24)) * 0.25 
    
    cost_vector = create_cost_vector(asset_names)
    
    # --- Hazır Bounds Verisi ---
    bounds_df = pd.read_csv(os.path.join(PROCESSED_DIR, "capacity_bounds.csv"), index_col=0)
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
    
    # --- Regional Balance Constraint (HARİKA DÜZELTME) ---
    # Harici dosyaya gerek kalmadı, df_features içinde 'region' zaten var!
    regional_indices = {}
    for i, asset in enumerate(asset_names):
        prov_name = normalize_name(asset.split('_')[0])
        try:
            # Bölge bilgisini doğrudan mevcut features DataFrame'inden çekiyoruz
            region = df_features.loc[prov_name, 'region']
            if isinstance(region, pd.Series): # Bazen aynı isimli mükerrer satır varsa ilkini al
                region = region.iloc[0]
        except KeyError:
            region = 'Unknown'
            
        if region not in regional_indices:
            regional_indices[region] = []
        regional_indices[region].append(i)

    # Kısıtlar
    cons = [
        {'type': 'ineq', 'fun': lambda x: np.dot(mu_vector, x) - TARGET_DEMAND},
        {'type': 'ineq', 'fun': lambda x: BUDGET - np.dot(cost_vector, x)}
    ]
    
    for region, indices in regional_indices.items():
        cons.append({
            'type': 'ineq',
            'fun': lambda x, idxs=indices: np.sum(x[idxs]) - MIN_REGIONAL_MW
        })
    
    # --- Warm-Start Chaining ---
    x0_zero = np.zeros(n_vars)
    results = []
    lambdas = np.linspace(0.01, 0.99, 20)
    
    print(f"Tracing Efficient Frontier with Dynamic Demand: {TARGET_DEMAND:.2f} MW...")
    
    for l in tqdm(lambdas, desc="Optimizing SLSQP"):
        res_zero = minimize(objective_function, x0_zero, args=(l, cov_matrix, cost_vector), 
                            method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000, 'ftol': 1e-6})
                            
        res_max = minimize(objective_function, x0_max, args=(l, cov_matrix, cost_vector), 
                           method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000, 'ftol': 1e-6})
        
        if res_zero.success and res_max.success:
            best_res = res_zero if res_zero.fun < res_max.fun else res_max
        elif res_zero.success:
            best_res = res_zero
        elif res_max.success:
            best_res = res_max
        else:
            best_res = None
            
        if best_res:
            p_risk = np.dot(best_res.x.T, np.dot(cov_matrix, best_res.x))
            p_cost = np.dot(cost_vector, best_res.x)
            p_prod = np.dot(mu_vector, best_res.x)
            row = [round(l, 3), p_cost, p_risk, p_prod] + list(best_res.x)
            results.append(row)
        else:
            print(f"Warning: No feasible solution found for lambda={l:.2f}")
            
    cols = ['lambda', 'total_cost', 'total_risk', 'expected_production'] + asset_names
    df_out = pd.DataFrame(results, columns=cols)
    out_path = os.path.join(RESULTS_DIR, "efficient_frontier_slsqp.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nSLSQP Complete! Found {len(df_out)} points. Saved to {out_path}")

if __name__ == "__main__":
    run_slsqp()
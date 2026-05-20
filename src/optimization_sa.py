import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths & Params
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

COST_SOLAR_MW = 0.8
COST_WIND_MW = 1.2
BUDGET = 80000.0  # Milyon USD. Calibrated from IRENA 2035 Turkey roadmap
                  # estimates (60-80B USD for 53GW solar + 30GW wind).
MIN_REGIONAL_MW = 500.0

# Scale normalization (same as SLSQP and NSGA-II)
VAR_SCALE = 8e6    # Re-calibrated for post-CF-fix: risk now ranges 3.9M-8M
                   # (was 4M-12M with inflated CF). Scaling to max risk
                   # keeps normalized_risk ∈ [0,1] like normalized_cost.
COST_SCALE = BUDGET

def normalize_name(text):
    return text.translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosucgiosu")).lower().strip()

def evaluate_fitness(x, cov_matrix, mu_vector, cost_vector, target_demand, regional_indices, lmbda=0.5):
    """
    Objective function with penalty method.
    lmbda=0.5 gives a balanced portfolio between risk and cost.
    """
    risk = np.dot(x.T, np.dot(cov_matrix, x))
    cost = np.dot(cost_vector, x)
    prod = np.dot(mu_vector, x)
    
    norm_risk = risk / VAR_SCALE
    norm_cost = cost / COST_SCALE
    
    base_obj = lmbda * norm_risk + (1 - lmbda) * norm_cost
    
    # --- Penalty calibration (between 100 and 500) ---
    penalty = 0.0
    
    # 1. Demand violation (very important, max penalty 500)
    if prod < target_demand:
        penalty += 500.0 * ((target_demand - prod) / target_demand)
        
    # 2. Budget violation (very important, max penalty 500)
    if cost > BUDGET:
        penalty += 500.0 * ((cost - BUDGET) / BUDGET)
        
    # 3. Regional balance violation (important, max penalty 100)
    for region, idxs in regional_indices.items():
        reg_prod = np.sum(x[idxs])
        if reg_prod < MIN_REGIONAL_MW:
            penalty += 100.0 * ((MIN_REGIONAL_MW - reg_prod) / MIN_REGIONAL_MW)
            
    return base_obj + penalty, cost, risk, prod

def run_simulated_annealing():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load data
    cov_matrix = pd.read_csv(os.path.join(PROCESSED_DIR, "covariance_matrix.csv"), index_col=0).values
    mu_df = pd.read_csv(os.path.join(PROCESSED_DIR, "mean_vector.csv"))
    
    if 'asset' in mu_df.columns:
        asset_names = mu_df['asset'].values
        mean_col = [c for c in mu_df.columns if c != 'asset'][0]
        mu_vector = mu_df[mean_col].values
    else:
        asset_names = mu_df.iloc[:, 0].values
        mu_vector = mu_df.iloc[:, 1].values
        
    cost_vector = np.array([COST_SOLAR_MW if 'solar' in a else COST_WIND_MW for a in asset_names])
    n_vars = len(asset_names)
    
    # 2. Demand and bounds
    df_features = pd.read_csv(os.path.join(PROCESSED_DIR, "province_features.csv"))
    df_features['norm_prov'] = df_features['province'].apply(normalize_name)
    df_features = df_features.set_index('norm_prov')
    
    # 2035 National Energy Plan update
    # Instead of the old TUIK demand, we use the 2035 projection of 510.5 TWh.
    total_demand_mwh = 510.5 * 1e6  # TWh to MWh
    TARGET_DEMAND = (total_demand_mwh / (365 * 24)) * 0.25
    
    bounds_df = pd.read_csv(os.path.join(PROCESSED_DIR, "capacity_bounds.csv"), index_col=0)
    x_max = np.array([bounds_df.loc[a, 'upper_mw'] if a in bounds_df.index else 500.0 for a in asset_names])
    
    # 3. Regional indices
    regional_indices = {}
    for i, asset in enumerate(asset_names):
        prov_name = normalize_name(asset.split('_')[0])
        try:
            region = df_features.loc[prov_name, 'region']
            if isinstance(region, pd.Series): region = region.iloc[0]
        except KeyError:
            region = 'Unknown'
        if region not in regional_indices:
            regional_indices[region] = []
        regional_indices[region].append(i)

    # 4. Simulated Annealing parameters
    T = 5000.0
    T_min = 1e-5
    alpha = 0.9995
    step_size = 50.0  # neighborhood search step in MW
    
    max_iter = int(np.log(T_min / T) / np.log(alpha))
    print(f"Simulated Annealing Starting... (About {max_iter} iterations)")
    
    # Starting point (random valid solution)
    x_current = np.random.uniform(0, x_max / 10) 
    best_obj, best_cost, best_risk, best_prod = evaluate_fitness(x_current, cov_matrix, mu_vector, cost_vector, TARGET_DEMAND, regional_indices)
    best_x = np.copy(x_current)
    
    current_obj = best_obj

    # 5. Cooling loop
    for _ in tqdm(range(max_iter), desc="Cooling Process"):
        # Create a neighbor solution (random perturbation + bounds clip)
        perturbation = np.random.normal(0, step_size, n_vars)
        x_new = np.clip(x_current + perturbation, 0, x_max)
        
        new_obj, new_cost, new_risk, new_prod = evaluate_fitness(x_new, cov_matrix, mu_vector, cost_vector, TARGET_DEMAND, regional_indices)
        
        # Acceptance criterion (Metropolis-Hastings)
        delta_e = new_obj - current_obj
        
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            x_current = np.copy(x_new)
            current_obj = new_obj
            
            # Update best solution
            if current_obj < best_obj:
                best_obj = current_obj
                best_x = np.copy(x_current)
                best_cost, best_risk, best_prod = new_cost, new_risk, new_prod
                
        # Cool down
        T *= alpha

    # 6. Post-Processing: if production is below target, scale up proportionally
    if best_prod < TARGET_DEMAND:
        print("\nWarning: Target production not reached, applying post-processing scale-up...")
        # Try scale-up: clip to bounds, recalculate, repeat (max 10 tries)
        for attempt in range(10):
            if best_prod >= TARGET_DEMAND:
                break
            scale_factor = TARGET_DEMAND / max(best_prod, 1e-6)
            best_x = np.clip(best_x * scale_factor, 0, x_max)
            best_obj, best_cost, best_risk, best_prod = evaluate_fitness(
                best_x, cov_matrix, mu_vector, cost_vector,
                TARGET_DEMAND, regional_indices
            )
        if best_prod < TARGET_DEMAND:
            print(f"  [WARNING] Post-processing not enough: prod={best_prod:.2f}, target={TARGET_DEMAND:.2f}")
            print(f"     The model cannot reach target production under capacity limits. "
                f"Consider relaxing upper bounds.")

    # 7. Save results
    results = [[best_cost, best_risk, best_prod] + list(best_x)]
    cols = ['total_cost', 'total_risk', 'expected_production'] + list(asset_names)
    df_out = pd.DataFrame(results, columns=cols)
    
    out_path = os.path.join(RESULTS_DIR, "best_solution_sa.csv")
    df_out.to_csv(out_path, index=False)
    
    print(f"\n[OK] Simulated Annealing Done!")
    print(f"Final Cost: {best_cost:.2f} M$ | Production: {best_prod:.2f} MW")
    print(f"Output saved: {out_path}")

if __name__ == "__main__":
    run_simulated_annealing()
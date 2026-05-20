import os
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.repair import Repair
# Paths & Params
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

COST_SOLAR_MW = 0.8
COST_WIND_MW = 1.2
BUDGET = 80000.0  # Milyon USD. Calibrated from IRENA 2035 Turkey roadmap
                  # estimates (60-80B USD for 53GW solar + 30GW wind)
MIN_REGIONAL_MW = 500.0

# --- Scale normalization (same as SLSQP) ---
VAR_SCALE = 8e6    # Re-calibrated for post-CF-fix: risk now ranges 3.9M-8M
                   # (was 4M-12M with inflated CF). Scaling to max risk
                   # keeps normalized_risk ∈ [0,1] like normalized_cost.
COST_SCALE = BUDGET # Cost scale

def normalize_name(text):
    return text.translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosucgiosu")).lower().strip()


class MinimumThresholdRepair(Repair):
    """
    Repair operator for the NSGA-II population.
    If a province has investment above 0 but below a threshold (e.g. 50 MW),
    set it to 0 to avoid small unprofitable allocations.
    """
    def __init__(self, threshold=50.0):
        super().__init__()
        self.threshold = threshold

    def _do(self, problem, X, **kwargs):
        # Find values between 0 and threshold using logical operators
        # and set them to 0.0 (repair the gene)
        mask = np.logical_and(X > 0, X < self.threshold)
        X[mask] = 0.0
        return X

# --- VECTORIZED PROBLEM CLASS ---
class RenewablePortfolioProblem(Problem):
    def __init__(self, cov_matrix, mu_vector, cost_vector, bounds, target_demand, budget, regional_indices):
        n_vars = len(mu_vector)
        n_obj = 2  # Objective 1: Risk, Objective 2: Cost
        n_constr = 2 + len(regional_indices) # Demand + Budget + 7 Regions
        
        xl = np.array([b[0] for b in bounds])
        xu = np.array([b[1] for b in bounds])
        
        super().__init__(n_var=n_vars, n_obj=n_obj, n_ieq_constr=n_constr, xl=xl, xu=xu)
        
        self.cov_matrix = cov_matrix
        self.mu_vector = mu_vector
        self.cost_vector = cost_vector
        self.target_demand = target_demand
        self.budget = budget
        self.regional_indices = regional_indices

    def _evaluate(self, X, out, *args, **kwargs):
        # 1. OBJECTIVES (F)
        # Vectorized risk calculation (all individuals at once using np.einsum)
        risk = np.einsum('ij,jk,ik->i', X, self.cov_matrix, X)
        cost = X @ self.cost_vector
        
        # Normalization
        f1_risk = risk / VAR_SCALE
        f2_cost = cost / COST_SCALE
        
        out["F"] = np.column_stack([f1_risk, f2_cost])
        
        # 2. CONSTRAINTS (G) -> In pymoo, constraints must be G <= 0
        G = []
        
        # Demand constraint: target_demand <= mu * X  =>  target_demand - mu * X <= 0
        g_demand = self.target_demand - (X @ self.mu_vector)
        G.append(g_demand)
        
        # Budget constraint: cost <= budget => cost - budget <= 0
        g_budget = cost - self.budget
        G.append(g_budget)
        
        # Regional balance: min_regional <= sum(region_x) => min_regional - sum(region_x) <= 0
        for region, idxs in self.regional_indices.items():
            g_reg = MIN_REGIONAL_MW - np.sum(X[:, idxs], axis=1)
            G.append(g_reg)
            
        out["G"] = np.column_stack(G)

def run_nsga2():
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
    
    # 2. Demand and bounds
    df_features = pd.read_csv(os.path.join(PROCESSED_DIR, "province_features.csv"))
    df_features = df_features.drop_duplicates(subset=['province'], keep='last')
    df_features['norm_prov'] = df_features['province'].apply(normalize_name)
    df_features = df_features.set_index('norm_prov')
    
    total_demand_mwh = 510.5 * 1e6  # TWh to MWh
    TARGET_DEMAND = (total_demand_mwh / (365 * 24)) * 0.25
    bounds_df = pd.read_csv(os.path.join(PROCESSED_DIR, "capacity_bounds.csv"), index_col=0)
    bounds = []
    for asset in asset_names:
        try:
            bounds.append((0, bounds_df.loc[asset, 'upper_mw']))
        except KeyError:
            bounds.append((0, 500.0))
    
    # 3. Regional indices (from features)
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

    # 4. Optimization (Problem + Algorithm setup)
    problem = RenewablePortfolioProblem(
        cov_matrix=cov_matrix, 
        mu_vector=mu_vector, 
        cost_vector=cost_vector, 
        bounds=bounds, 
        target_demand=TARGET_DEMAND, 
        budget=BUDGET, 
        regional_indices=regional_indices
    )
    
    # Consistent with proposal: Pop=200
    algorithm = NSGA2(
        pop_size=200,                
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        repair=MinimumThresholdRepair(threshold=50.0),
        eliminate_duplicates=True
    )
    
    print(f"NSGA-II Optimization Starting... (Population: 200, Generations: 500)")
    # Consistent with proposal: Gen=500
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 500),              
        seed=42,
        save_history=False,
        verbose=True                 
    )
    
    # 5. Extract the Pareto Front (feasibility filter)
    if res.X is not None:
        results = []
        # res.X contains only feasible individuals (those that satisfy constraints)
        for i, x in enumerate(res.X):
            actual_risk = np.dot(x.T, np.dot(cov_matrix, x))
            actual_cost = np.dot(cost_vector, x)
            actual_prod = np.dot(mu_vector, x)
            
            row = [actual_cost, actual_risk, actual_prod] + list(x)
            results.append(row)
            
        cols = ['total_cost', 'total_risk', 'expected_production'] + list(asset_names)
        df_out = pd.DataFrame(results, columns=cols)
        
        # Sort by risk so the Pareto line looks right on the graph
        df_out = df_out.sort_values(by='total_risk').reset_index(drop=True)
        
        out_path = os.path.join(RESULTS_DIR, "pareto_front_nsga2.csv")
        df_out.to_csv(out_path, index=False)
        print(f"\n[OK] NSGA-II Done! Found {len(df_out)} feasible solutions on the Pareto front.")
        print(f"Output saved: {out_path}")
    else:
        print("\n[WARNING] NSGA-II could not find any feasible solution.")

if __name__ == "__main__":
    run_nsga2()
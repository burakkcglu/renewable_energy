import os
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

# Paths & Params
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

COST_SOLAR_MW = 0.8
COST_WIND_MW = 1.2
BUDGET = 50000.0  # Milyon USD
MIN_REGIONAL_MW = 500.0

# --- SLSQP'de başarıyı getiren Ölçek Normalizasyonu ---
VAR_SCALE = 1e7     # Risk ölçeği
COST_SCALE = BUDGET # Maliyet ölçeği

def normalize_name(text):
    return text.translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosucgiosu")).lower().strip()

# --- VEKTÖRİZE EDİLMİŞ PROBLEM SINIFI (Senin Mimari Önerin) ---
class RenewablePortfolioProblem(Problem):
    def __init__(self, cov_matrix, mu_vector, cost_vector, bounds, target_demand, budget, regional_indices):
        n_vars = len(mu_vector)
        n_obj = 2  # Hedef 1: Risk (Var), Hedef 2: Cost
        n_constr = 2 + len(regional_indices) # Talep + Bütçe + 7 Bölge
        
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
        # 1. HEDEFLER (F)
        # Vektörize Risk Hesabı: (np.einsum ile popülasyondaki tüm bireyler tek seferde hesaplanır)
        risk = np.einsum('ij,jk,ik->i', X, self.cov_matrix, X)
        cost = X @ self.cost_vector
        
        # Normalizasyon
        f1_risk = risk / VAR_SCALE
        f2_cost = cost / COST_SCALE
        
        out["F"] = np.column_stack([f1_risk, f2_cost])
        
        # 2. KISITLAR (G) -> Pymoo'da kısıtlar G <= 0 şeklinde olmalıdır
        G = []
        
        # Talep Kısıtı: target_demand <= mu * X  =>  target_demand - mu * X <= 0
        g_demand = self.target_demand - (X @ self.mu_vector)
        G.append(g_demand)
        
        # Bütçe Kısıtı: cost <= budget => cost - budget <= 0
        g_budget = cost - self.budget
        G.append(g_budget)
        
        # Bölgesel Denge Kısıtı: min_regional <= sum(region_x) => min_regional - sum(region_x) <= 0
        for region, idxs in self.regional_indices.items():
            g_reg = MIN_REGIONAL_MW - np.sum(X[:, idxs], axis=1)
            G.append(g_reg)
            
        out["G"] = np.column_stack(G)

def run_nsga2():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Veri Yükleme
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
    
    # 2. Talep ve Sınırlar
    df_features = pd.read_csv(os.path.join(PROCESSED_DIR, "province_features.csv"))
    df_features['norm_prov'] = df_features['province'].apply(normalize_name)
    df_features = df_features.set_index('norm_prov')
    
    total_demand_mwh = df_features['total_demand_mwh'].sum()
    TARGET_DEMAND = (total_demand_mwh / (365 * 24)) * 0.25 
    
    bounds_df = pd.read_csv(os.path.join(PROCESSED_DIR, "capacity_bounds.csv"), index_col=0)
    bounds = []
    for asset in asset_names:
        try:
            bounds.append((0, bounds_df.loc[asset, 'upper_mw']))
        except KeyError:
            bounds.append((0, 500.0))
    
    # 3. Bölgesel İndeksler (Features üzerinden)
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

    # 4. Optimizasyon (Problem + Algoritma Tanımı)
    problem = RenewablePortfolioProblem(
        cov_matrix=cov_matrix, 
        mu_vector=mu_vector, 
        cost_vector=cost_vector, 
        bounds=bounds, 
        target_demand=TARGET_DEMAND, 
        budget=BUDGET, 
        regional_indices=regional_indices
    )
    
    # Proposal ile tutarlı: Pop=200
    algorithm = NSGA2(
        pop_size=200,                
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    print(f"NSGA-II Optimizasyonu Başlıyor... (Popülasyon: 200, Jenerasyon: 500)")
    # Proposal ile tutarlı: Gen=500
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 500),              
        seed=42,
        save_history=False,
        verbose=True                 
    )
    
    # 5. Pareto Cephesinin Çıkarılması (Feasibility Filter)
    if res.X is not None:
        results = []
        # res.X sadece geçerli (kısıtları sağlayan) bireyleri içerir
        for i, x in enumerate(res.X):
            actual_risk = np.dot(x.T, np.dot(cov_matrix, x))
            actual_cost = np.dot(cost_vector, x)
            actual_prod = np.dot(mu_vector, x)
            
            row = [actual_cost, actual_risk, actual_prod] + list(x)
            results.append(row)
            
        cols = ['total_cost', 'total_risk', 'expected_production'] + list(asset_names)
        df_out = pd.DataFrame(results, columns=cols)
        
        # Risk bazında sıralıyoruz ki grafikte Pareto çizgisi düzgün çizilebilsin
        df_out = df_out.sort_values(by='total_risk').reset_index(drop=True)
        
        out_path = os.path.join(RESULTS_DIR, "pareto_front_nsga2.csv")
        df_out.to_csv(out_path, index=False)
        print(f"\n✓ NSGA-II Tamamlandı! Pareto cephesinde {len(df_out)} geçerli çözüm bulundu.")
        print(f"Çıktı kaydedildi: {out_path}")
    else:
        print("\nKritik Uyarı: NSGA-II geçerli (feasible) bir çözüm bulamadı.")

if __name__ == "__main__":
    run_nsga2()
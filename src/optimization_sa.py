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
BUDGET = 50000.0  # Milyon USD
MIN_REGIONAL_MW = 500.0

# Ölçek Normalizasyonu (SLSQP ve NSGA-II ile aynı)
VAR_SCALE = 1e7
COST_SCALE = BUDGET

def normalize_name(text):
    return text.translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosucgiosu")).lower().strip()

def evaluate_fitness(x, cov_matrix, mu_vector, cost_vector, target_demand, regional_indices, lmbda=0.5):
    """
    Hedef Fonksiyonu + Penaltı Yöntemi
    lmbda=0.5 ile dengeli (balanced) bir portföy arıyoruz.
    """
    risk = np.dot(x.T, np.dot(cov_matrix, x))
    cost = np.dot(cost_vector, x)
    prod = np.dot(mu_vector, x)
    
    norm_risk = risk / VAR_SCALE
    norm_cost = cost / COST_SCALE
    
    base_obj = lmbda * norm_risk + (1 - lmbda) * norm_cost
    
    # --- Penaltı Kalibrasyonu (100 - 500 aralığında) ---
    penalty = 0.0
    
    # 1. Talep İhlali (Çok kritik -> Max ceza 500)
    if prod < target_demand:
        penalty += 500.0 * ((target_demand - prod) / target_demand)
        
    # 2. Bütçe İhlali (Çok kritik -> Max ceza 500)
    if cost > BUDGET:
        penalty += 500.0 * ((cost - BUDGET) / BUDGET)
        
    # 3. Bölgesel Denge İhlali (Kritik -> Max ceza 100)
    for region, idxs in regional_indices.items():
        reg_prod = np.sum(x[idxs])
        if reg_prod < MIN_REGIONAL_MW:
            penalty += 100.0 * ((MIN_REGIONAL_MW - reg_prod) / MIN_REGIONAL_MW)
            
    return base_obj + penalty, cost, risk, prod

def run_simulated_annealing():
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
    n_vars = len(asset_names)
    
    # 2. Talep ve Sınırlar
    df_features = pd.read_csv(os.path.join(PROCESSED_DIR, "province_features.csv"))
    df_features['norm_prov'] = df_features['province'].apply(normalize_name)
    df_features = df_features.set_index('norm_prov')
    
    TARGET_DEMAND = (df_features['total_demand_mwh'].sum() / (365 * 24)) * 0.25 
    
    bounds_df = pd.read_csv(os.path.join(PROCESSED_DIR, "capacity_bounds.csv"), index_col=0)
    x_max = np.array([bounds_df.loc[a, 'upper_mw'] if a in bounds_df.index else 500.0 for a in asset_names])
    
    # 3. Bölgesel İndeksler
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

    # 4. Simulated Annealing Parametreleri
    T = 5000.0
    T_min = 1e-5
    alpha = 0.9995
    step_size = 50.0  # MW cinsinden komşuluk arama adımı
    
    max_iter = int(np.log(T_min / T) / np.log(alpha))
    print(f"Simulated Annealing Başlıyor... (Yaklaşık {max_iter} İterasyon)")
    
    # Başlangıç noktası (Rastgele geçerli bir nokta üretimi)
    x_current = np.random.uniform(0, x_max / 10) 
    best_obj, best_cost, best_risk, best_prod = evaluate_fitness(x_current, cov_matrix, mu_vector, cost_vector, TARGET_DEMAND, regional_indices)
    best_x = np.copy(x_current)
    
    current_obj = best_obj

    # 5. Soğutma Döngüsü (Cooling Schedule)
    for _ in tqdm(range(max_iter), desc="Cooling Process"):
        # Komşu çözüm üret (Rastgele perturbation + Bounds kısıtı)
        perturbation = np.random.normal(0, step_size, n_vars)
        x_new = np.clip(x_current + perturbation, 0, x_max)
        
        new_obj, new_cost, new_risk, new_prod = evaluate_fitness(x_new, cov_matrix, mu_vector, cost_vector, TARGET_DEMAND, regional_indices)
        
        # Kabul kriteri (Metropolis-Hastings)
        delta_e = new_obj - current_obj
        
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            x_current = np.copy(x_new)
            current_obj = new_obj
            
            # En iyi çözümü güncelle
            if current_obj < best_obj:
                best_obj = current_obj
                best_x = np.copy(x_current)
                best_cost, best_risk, best_prod = new_cost, new_risk, new_prod
                
        # Soğuma
        T *= alpha

    # 6. Post-Processing: Eğer üretim hedefin altında kaldıysa orantılı olarak scale-up yap
    if best_prod < TARGET_DEMAND:
        print("\nUyarı: Hedef üretim sağlanamadı, post-processing (scale-up) uygulanıyor...")
        scale_factor = TARGET_DEMAND / best_prod
        best_x = np.clip(best_x * scale_factor, 0, x_max)
        # Yeniden hesapla
        best_obj, best_cost, best_risk, best_prod = evaluate_fitness(best_x, cov_matrix, mu_vector, cost_vector, TARGET_DEMAND, regional_indices)

    # 7. Sonuçları Kaydet
    results = [[best_cost, best_risk, best_prod] + list(best_x)]
    cols = ['total_cost', 'total_risk', 'expected_production'] + list(asset_names)
    df_out = pd.DataFrame(results, columns=cols)
    
    out_path = os.path.join(RESULTS_DIR, "best_solution_sa.csv")
    df_out.to_csv(out_path, index=False)
    
    print(f"\n✓ Simulated Annealing Tamamlandı!")
    print(f"Nihai Maliyet: {best_cost:.2f} M$ | Üretim: {best_prod:.2f} MW")
    print(f"Çıktı kaydedildi: {out_path}")

if __name__ == "__main__":
    run_simulated_annealing()
import os
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

def run_robustness():
    print("Robustness analizi başlatılıyor...")
    
    # 1. Verileri Oku
    monthly_cf = pd.read_csv(os.path.join(PROCESSED_DIR, "monthly_cf_matrix.csv"), index_col=0).values
    best_sa = pd.read_csv(os.path.join(RESULTS_DIR, "best_solution_sa.csv"))
    
    # SA çözümünden kapasiteleri al
    asset_cols = [c for c in best_sa.columns if 'solar' in c or 'wind' in c]
    x_opt = best_sa[asset_cols].values[0]
    
    # 2. Monte Carlo (1000 senaryo)
    n_scenarios = 1000
    shortfalls = []
    
    for _ in range(n_scenarios):
        # Yıllık bazda rastgele gürültü ekle (hava durumu değişkenliği)
        noise = np.random.normal(1.0, 0.15, monthly_cf.shape) 
        scenario_cf = np.clip(monthly_cf * noise, 0, 1)
        
        # Üretim simülasyonu
        prod = (scenario_cf @ x_opt).sum() / 252 # Aylık ortalama üretim
        shortfalls.append(max(0, 15000 - prod)) # 15000 MW hedef baz alındı
        
    # 3. Drought Year (En kötü 12 aylık performans)
    # 252 ay arasından en kötü performansı seç
    monthly_prods = monthly_cf @ x_opt
    drought_prod = np.min([np.sum(monthly_prods[i:i+12]) for i in range(240)]) / 12
    
    # 4. Fossil Fuel Shock (+%50 maliyet senaryosu)
    # (Basit LCOE karşılaştırması)
    base_cost = best_sa['total_cost'].values[0]
    shock_cost = base_cost * 1.5 # Şok etkisi
    
    # Raporu kaydet
    report = {
        'scenario': ['Monte Carlo Avg Shortfall', 'Drought Year Avg Prod', 'Fuel Shock Cost'],
        'value': [np.mean(shortfalls), drought_prod, shock_cost]
    }
    pd.DataFrame(report).to_csv(os.path.join(RESULTS_DIR, "robustness_report.csv"), index=False)
    
    print("✓ Robustness analizi tamamlandı. 'robustness_report.csv' kaydedildi.")

if __name__ == "__main__":
    run_robustness()
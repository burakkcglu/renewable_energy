"""
main.py
End-to-End Execution Pipeline for Renewable Energy Portfolio Optimization.
Includes Data Collection, Processing, and 3 Optimization Methods.
"""

import os
import time

# Scriptleri içeri aktar
from src.data_collection import load_provinces, fetch_all_provinces, print_data_summary
from src.data_processing import process_all_data
from src.optimization_slsqp import run_slsqp
from src.optimization_nsga2 import run_nsga2
from src.optimization_sa import run_simulated_annealing
from src.tuik_eurostat_prep import prepare_tuik_data

# Dosya Yolları
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROVINCES_FILE = os.path.join(PROJECT_DIR, "data", "tr_provinces_coords.csv")
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")

def run_pipeline():
    print("="*60)
    print("🚀 TURKEY RENEWABLE ENERGY PORTFOLIO PIPELINE INITIATED")
    print("="*60)
    start_time = time.time()

    # ---------------------------------------------------------
    # ADIM 0: TUIK Sosyo-Ekonomik Veri Hazırlığı (YENİ!)
    # ---------------------------------------------------------
    print("\n[STEP 0/5] Cleaning TUIK Demography & Demand Data...")
    try:
        prepare_tuik_data(reference_year=2024)
    except Exception as e:
        print(f"HATA: TUIK verileri işlenirken sorun çıktı: {e}")
        return
    print("="*60)
    print("🚀 TURKEY RENEWABLE ENERGY PORTFOLIO PIPELINE INITIATED")
    print("="*60)
    start_time = time.time()

    # ---------------------------------------------------------
    # ADIM 1: Veri Toplama (Data Collection)
    # ---------------------------------------------------------
    print("\n[STEP 1/5] Starting Data Collection (NASA POWER API)...")
    if not os.path.exists(PROVINCES_FILE):
        print(f"HATA: Koordinat dosyası bulunamadı! Lütfen {PROVINCES_FILE} dosyasını oluşturun.")
        return

    provinces = load_provinces(PROVINCES_FILE)
    fetch_all_provinces(provinces, RAW_DIR)
    print_data_summary(RAW_DIR)

    # ---------------------------------------------------------
    # ADIM 2: Veri Ön İşleme (Data Processing)
    # ---------------------------------------------------------
    print("\n[STEP 2/5] Starting Data Processing & Covariance Matrix Generation...")
    try:
        process_all_data()
    except Exception as e:
        print(f"HATA: Veri işleme sırasında bir sorun oluştu: {e}")
        return

    # ---------------------------------------------------------
    # ADIM 3: Optimizasyon Yöntemi 1 (SLSQP)
    # ---------------------------------------------------------
    print("\n[STEP 3/5] Running Method 1: Markowitz Mean-Variance (SLSQP)...")
    try:
        run_slsqp()
    except Exception as e:
        print(f"HATA: SLSQP sırasında bir sorun oluştu: {e}")

    # ---------------------------------------------------------
    # ADIM 4: Optimizasyon Yöntemi 2 (NSGA-II)
    # ---------------------------------------------------------
    print("\n[STEP 4/5] Running Method 2: Multi-Objective Genetic Algorithm (NSGA-II)...")
    try:
        run_nsga2()
    except Exception as e:
        print(f"HATA: NSGA-II sırasında bir sorun oluştu: {e}")

    # ---------------------------------------------------------
    # ADIM 5: Optimizasyon Yöntemi 3 (Simulated Annealing)
    # ---------------------------------------------------------
    print("\n[STEP 5/5] Running Method 3: Simulated Annealing...")
    try:
        run_simulated_annealing()
    except Exception as e:
        print(f"HATA: Simulated Annealing sırasında bir sorun oluştu: {e}")

    # Bitiş
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"✅ FULL PIPELINE SUCCESSFULLY COMPLETED IN {elapsed_time:.2f} SECONDS!")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
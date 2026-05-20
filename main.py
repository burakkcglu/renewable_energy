"""
main.py
========
Runs the full pipeline for Turkey renewable energy portfolio optimization.

Steps in the pipeline:
  1. Prepare TUIK + Eurostat data
  2. Collect NASA POWER data (skipped if already saved)
  3. Calculate capacity factors, covariance matrix, and bounds
  4. Markowitz QP optimization (SLSQP)
  5. Multi-objective genetic algorithm (NSGA-II)
  6. Simulated annealing
  7. Robustness checks (Monte Carlo, drought, fossil avoided cost)
  8. Strategic analysis and jury defense statements

Usage:
    python main.py                # run full pipeline (skips finished steps)
    python main.py --force-nasa   # download NASA data again
"""
 
import os
import sys
import time
import argparse
 
# Add project folder to path so we can import from src/
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
 
from src.tuik_eurostat_prep import prepare_tuik_data
from src.data_collection import load_provinces, fetch_all_provinces, print_data_summary
from src.data_processing import process_all_data
from src.optimization_slsqp import run_slsqp
from src.optimization_nsga2 import run_nsga2
from src.optimization_sa import run_simulated_annealing
from src.robustness import run_robustness
from src.analysis import run_advanced_analysis
 
 
# File and folder paths
PROVINCES_FILE = os.path.join(PROJECT_DIR, "data", "tr_provinces_coords.csv")
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
 
NASA_CACHE = os.path.join(PROCESSED_DIR, "all_provinces_daily.csv")
CF_CACHE = os.path.join(PROCESSED_DIR, "covariance_matrix.csv")
 
 
def step(num, total, name):
    """Print a nice header for each step."""
    print("\n" + "=" * 70)
    print(f"[STEP {num}/{total}] {name}")
    print("=" * 70)
 
 
def run_pipeline(force_nasa=False, force_processing=False):
    print("=" * 70)
    print("TURKEY RENEWABLE ENERGY PORTFOLIO PIPELINE")
    print("=" * 70)
    start_time = time.time()
 
    # ---------------------------------------------------------
    # STEP 1: Prepare TUIK + Eurostat data
    # ---------------------------------------------------------
    step(1, 8, "TUIK + Eurostat data preparation")
    try:
        prepare_tuik_data(reference_year=2024)
    except Exception as e:
        print(f"[ERROR] STEP 1 failed: {e}")
        return
 
    # ---------------------------------------------------------
    # STEP 2: Collect NASA POWER data (skip if already saved)
    # ---------------------------------------------------------
    step(2, 8, "NASA POWER data collection")
    if os.path.exists(NASA_CACHE) and not force_nasa:
        print(f"  [OK] {os.path.basename(NASA_CACHE)} already exists. Skipping.")
        print(f"    Use --force-nasa to re-download.")
    else:
        if not os.path.exists(PROVINCES_FILE):
            print(f"[ERROR] Coordinate file missing: {PROVINCES_FILE}")
            return
        try:
            provinces = load_provinces(PROVINCES_FILE)
            fetch_all_provinces(provinces, RAW_DIR)
            print_data_summary(RAW_DIR)
        except Exception as e:
            print(f"[ERROR] STEP 2 failed: {e}")
            return
 
    # ---------------------------------------------------------
    # STEP 3: Calculate capacity factors, covariance, and bounds
    # ---------------------------------------------------------
    step(3, 8, "Capacity factors, covariance matrix, bounds")
    if os.path.exists(CF_CACHE) and not force_processing:
        print(f"  [OK] {os.path.basename(CF_CACHE)} already exists. Skipping.")
        print(f"    Delete it to force recomputation.")
    else:
        try:
            process_all_data()
        except Exception as e:
            print(f"[ERROR] STEP 3 failed: {e}")
            return
 
    # ---------------------------------------------------------
    # STEP 4: Run SLSQP (Markowitz QP) optimization
    # ---------------------------------------------------------
    step(4, 8, "Markowitz Mean-Variance Optimization (SLSQP)")
    try:
        run_slsqp()
    except Exception as e:
        print(f"[ERROR] STEP 4 failed: {e}")
        # Keep going — try the other optimizers anyway
 
    # ---------------------------------------------------------
    # STEP 5: Run NSGA-II genetic algorithm
    # ---------------------------------------------------------
    step(5, 8, "Multi-Objective Genetic Algorithm (NSGA-II)")
    try:
        run_nsga2()
    except Exception as e:
        print(f"[ERROR] STEP 5 failed: {e}")
 
    # ---------------------------------------------------------
    # STEP 6: Run simulated annealing
    # ---------------------------------------------------------
    step(6, 8, "Simulated Annealing (from scratch)")
    try:
        run_simulated_annealing()
    except Exception as e:
        print(f"[ERROR] STEP 6 failed: {e}")
 
    # ---------------------------------------------------------
    # STEP 7: Run robustness checks
    # ---------------------------------------------------------
    step(7, 8, "Robustness (Monte Carlo + drought + avoided cost)")
    try:
        run_robustness()
    except Exception as e:
        print(f"[ERROR] STEP 7 failed: {e}")
 
    # ---------------------------------------------------------
    # STEP 8: Strategic analysis and jury defense
    # ---------------------------------------------------------
    step(8, 8, "Strategic analysis & jury defense statements")
    try:
        run_advanced_analysis()
    except Exception as e:
        print(f"[ERROR] STEP 8 failed: {e}")
 
    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[OK] FULL PIPELINE COMPLETED IN {elapsed:.1f} SECONDS "
          f"({elapsed/60:.1f} minutes)")
    print("=" * 70)
    print(f"\n   Outputs:")
    print(f"     - data/processed/    -- processed datasets")
    print(f"     - results/           -- optimization frontiers + reports")
    print()
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turkey RE Portfolio Pipeline")
    parser.add_argument(
        '--force-nasa', action='store_true',
        help='Re-download NASA data even if cache exists (slow: ~3 hours)'
    )
    parser.add_argument(
        '--force-processing', action='store_true',
        help='Recompute capacity factors and covariance even if cached'
    )
    args = parser.parse_args()
 
    run_pipeline(
        force_nasa=args.force_nasa,
        force_processing=args.force_processing,
    )
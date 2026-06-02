"""
main.py
=======
Main script to run the entire project pipeline.

Run:
    python src/main.py
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C


def step(n, total, name):
    print(f"\n[STEP {n}/{total}] {name}")
    print("-" * 40)


def run_pipeline(force_nasa=False, force_processing=False):
    t0 = time.time()
    nasa_cache = os.path.join(C.PROCESSED_DIR, "all_provinces_daily.csv")
    cov_cache = os.path.join(C.PROCESSED_DIR, "covariance_matrix.csv")

    step(1, 8, "Prepare data")
    try:
        from tuik_eurostat_prep import prepare_tuik_data
        prepare_tuik_data(reference_year=2024)
    except Exception as e:
        print(f"Error in step 1: {e}")

    step(2, 8, "Download weather data")
    if os.path.exists(nasa_cache) and not force_nasa:
        print("  Data cached. Use --force-nasa to download again.")
    else:
        from data_collection import load_provinces, fetch_all_provinces
        fetch_all_provinces(load_provinces())

    step(3, 8, "Calculate matrices")
    if os.path.exists(cov_cache) and not force_processing:
        print("  Data cached. Use --force-processing to calculate again.")
    else:
        from data_processing import process_all_data
        process_all_data()

    step(4, 8, "Run SLSQP Optimizer")
    try:
        from optimization_slsqp import run_slsqp
        run_slsqp()
    except Exception as e:
        print(f"Error in step 4: {e}")

    step(5, 8, "Run NSGA-II Optimizer")
    try:
        from optimization_nsga2 import run_nsga2
        run_nsga2()
    except Exception as e:
        print(f"Error in step 5: {e}")

    step(6, 8, "Run Simulated Annealing")
    try:
        from optimization_sa import run_simulated_annealing
        run_simulated_annealing()
    except Exception as e:
        print(f"Error in step 6: {e}")

    step(7, 8, "Check Robustness")
    try:
        from robustness import run_robustness
        run_robustness()
    except Exception as e:
        print(f"Error in step 7: {e}")

    step(8, 8, "Generate Analysis and Figures")
    try:
        from analysis import run_advanced_analysis
        run_advanced_analysis()
    except Exception as e:
        print(f"Error in analysis step: {e}")
    try:
        from make_figures import run_figures
        run_figures()
    except Exception as e:
        print(f"Error generating figures: {e}")

    print(f"\nPipeline finished in {(time.time()-t0)/60:.1f} minutes.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-nasa", action="store_true")
    ap.add_argument("--force-processing", action="store_true")
    a = ap.parse_args()
    run_pipeline(a.force_nasa, a.force_processing)
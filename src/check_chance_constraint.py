"""
check_chance_constraint.py
==========================
Validates the chance constraint for the generated portfolios.
Runs a Monte Carlo simulation and prints the annual production distribution.

Run: python src/check_chance_constraint.py
"""

import os
import numpy as np
import pandas as pd

import config as C
import opt_common as OC
from robustness import load_monthly_cf, load_solution

monthly_cf, asset_names, _ = load_monthly_cf()
D_target = C.annual_demand_target_mwh()
D_min = D_target * C.CHANCE_D_MIN_FRACTION

print(f"Target demand = {D_target/1e6:.2f} TWh/yr")
print(f"Min allowed demand = {D_min/1e6:.2f} TWh/yr")
print(f"Max failure prob = {C.CHANCE_EPSILON:.0%}")
print("-" * 50)

files = {
    "SLSQP": "efficient_frontier_slsqp.csv",
    "NSGA2": "pareto_front_nsga2.csv",
    "SA":    "best_solution_sa.csv",
}

rng_seed = C.SEED
for method, fname in files.items():
    x = load_solution(fname, asset_names)
    if x is None:
        continue

    # Re-run bootstrap from robustness check
    rng = np.random.default_rng(rng_seed)
    T = monthly_cf.shape[0]
    annual = []
    for _ in range(C.MC_N_SCENARIOS):
        start = rng.integers(0, T - 12)
        cf_year = monthly_cf[start:start + 12]
        annual.append(float((cf_year @ x * C.HOURS_PER_MONTH).sum()))
    annual = np.array(annual) / 1e6  # TWh

    below = (annual < D_min / 1e6)
    print(f"\n{method}:")
    print(f"  Failed scenarios : {below.sum()} / {len(annual)}")
    print(f"  Fail probability : {below.mean()*100:.2f}% -> "
          f"{'PASS' if below.mean() <= C.CHANCE_EPSILON else 'FAIL'}")
    print(f"  Annual production (TWh):")
    print(f"    min  = {annual.min():.2f}")
    print(f"    P1   = {np.percentile(annual,1):.2f}")
    print(f"    P5   = {np.percentile(annual,5):.2f}")
    print(f"    mean = {annual.mean():.2f}")
    print(f"    max  = {annual.max():.2f}")
    margin = (np.percentile(annual,5) - D_min/1e6)
    print(f"  P5 margin vs min: {margin:+.2f} TWh "
          f"({'safe' if margin > 5 else 'tight'})")
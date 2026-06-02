"""
diag_monthly.py
===============
Checks monthly production vs demand using the latest SLSQP results.
Helps identify bottleneck months.
"""

import os
import numpy as np
import pandas as pd

import config as C
import opt_common as OC

# Load latest result
df = pd.read_csv(os.path.join(C.RESULTS_DIR, "efficient_frontier_slsqp.csv"))
P = OC.load_problem()
assets = P["asset_names"]
x = df.iloc[len(df) // 2][assets].values.astype(float)

cf_by_month, demand_by_month = OC.load_monthly_profiles()
hours_per_month = C.HOURS_PER_YEAR / 12.0

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

print(f"{'Month':<6}{'Prod(TWh)':>13}{'Demand(TWh)':>13}{'Target×0.9':>12}"
      f"{'Coverage%':>12}{'Status':>9}")
print("-" * 65)
for m in range(12):
    prod_mwh = (cf_by_month[m] @ x) * hours_per_month
    dem_mwh = demand_by_month[m]
    target_90 = dem_mwh * C.MONTHLY_COVERAGE
    coverage = prod_mwh / dem_mwh * 100
    binding = "<- TIGHT" if prod_mwh <= target_90 * 1.01 else ""
    print(f"{months[m]:<6}{prod_mwh/1e6:>13.1f}{dem_mwh/1e6:>13.1f}"
          f"{target_90/1e6:>12.1f}{coverage:>11.1f}%{binding:>9}")

print("-" * 65)
total_prod = sum((cf_by_month[m] @ x) * hours_per_month for m in range(12))
total_dem = demand_by_month.sum()
print(f"{'YEAR':<6}{total_prod/1e6:>13.1f}{total_dem/1e6:>13.1f}"
      f"{'':>12}{total_prod/total_dem*100:>11.1f}%")
print(f"\nSolar GW: {x[np.array(['solar' in a for a in assets])].sum()/1000:.1f}  "
      f"Wind GW: {x[np.array(['wind' in a for a in assets])].sum()/1000:.1f}")
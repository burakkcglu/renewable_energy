"""
optimization_slsqp.py
=====================
Markowitz mean-variance via SLSQP, sweeping lambda to trace the efficient
frontier.

HYBRID: independent multi-start per lambda (notebook's insight — chaining warm
starts collapses the frontier to one local optimum) PLUS the src Pareto-dominance
post-filter (keeps the frontier monotonic). Gradient supplied analytically.
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

import config as C
import opt_common as OC


def run_slsqp():
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    P = OC.load_problem()
    cf_by_month, demand_by_month = OC.load_monthly_profiles()
    use_monthly = cf_by_month is not None
    if use_monthly:
        print(f"  Monthly demand matching ON: {len(demand_by_month)} months")
    else:
        print("  Monthly profiles not found — annual-only mode.")
    cov, mu, cvec = P["cov"], P["mu_mwh"], P["cost_vec"]
    upper, D_target = P["upper"], P["D_target"]
    var_scale = P["var_scale"]
    n = P["n"]

    def obj(x, lam):
        return lam * (x @ cov @ x) / var_scale + (1 - lam) * (cvec @ x) / C.COST_SCALE

    def grad(x, lam):
        return lam * (2 * cov @ x) / var_scale + (1 - lam) * cvec / C.COST_SCALE

    bounds = tuple((0.0, u) for u in upper)
    cons = [
        {"type": "ineq", "fun": lambda x: mu @ x - D_target, "jac": lambda x: mu},
        {"type": "ineq", "fun": lambda x: C.BUDGET_USD - cvec @ x, "jac": lambda x: -cvec},
    ]
    for region, idxs in P["regional_indices"].items():
        cons.append({"type": "ineq",
                     "fun": lambda x, i=idxs: np.sum(x[i]) - C.MIN_REGIONAL_MW})
    # Step 2: monthly demand matching.
    # For each calendar month m: (CF_m · x) * hours_in_month >= demand_m
    # CF_m·x is average MW that month; multiply by month hours to get MWh.
    if use_monthly:
        hours_per_month = C.HOURS_PER_YEAR / 12.0
        for m in range(12):
            cf_m = cf_by_month[m]
            d_m = demand_by_month[m] * C.MONTHLY_COVERAGE   # %90 kapsama
            cons.append({
                "type": "ineq",
                "fun": lambda x, c=cf_m, d=d_m: (c @ x) * hours_per_month - d,
                "jac": lambda x, c=cf_m: c * hours_per_month,
            })

    # Two diverse, INDEPENDENT starting points (not chained across lambda)
    half = upper / 2.0
    x0_solar = np.where(np.arange(n) < n // 2, half, half * 0.1)
    x0_wind = np.where(np.arange(n) < n // 2, half * 0.1, half)

    maxit = 1000
    lambdas = np.linspace(0.01, 0.99, 20)
    results = []
    print(f"SLSQP frontier: {len(lambdas)} lambdas x 2 independent starts | "
          f"D_target={D_target/1e6:.1f} TWh/yr")

    for lam in tqdm(lambdas, desc="SLSQP"):
        best = None
        for x0 in (x0_solar, x0_wind):
            res = minimize(obj, x0, args=(lam,), jac=grad, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"maxiter": maxit, "ftol": 1e-9})
            if (res.success or res.status in (1, 8)):
                if best is None or res.fun < best.fun:
                    best = res
        if best is not None:
            x = best.x
            results.append([round(lam, 3), OC.cost(x, cvec), OC.variance(x, cov),
                            OC.production(x, mu)] + list(x))

    cols = ["lambda", "total_cost", "total_risk", "expected_production"] + P["asset_names"]
    df = pd.DataFrame(results, columns=cols).sort_values("total_cost").reset_index(drop=True)

    # Pareto dominance filter (src): keep points where risk strictly decreases as cost rises
    keep = [0]
    min_risk = df.loc[0, "total_risk"]
    for i in range(1, len(df)):
        if df.loc[i, "total_risk"] < min_risk:
            keep.append(i); min_risk = df.loc[i, "total_risk"]
    df_out = df.loc[keep].reset_index(drop=True)

    path = os.path.join(C.RESULTS_DIR, "efficient_frontier_slsqp.csv")
    df_out.to_csv(path, index=False)
    print(f"SLSQP done: {len(df_out)} Pareto points (dropped {len(df)-len(df_out)}). "
          f"Saved {path}")
    return df_out


if __name__ == "__main__":
    run_slsqp()

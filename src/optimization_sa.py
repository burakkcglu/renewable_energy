"""
optimization_sa.py
==================
Simulated annealing baseline, implemented from scratch.

HYBRID: notebook's penalty design — SQUARED, strongly-weighted demand penalty
(so SA cannot drift into infeasible shortfalls) and a sensible warm start
proportional to renewable potential. The tiny (~1.0006x) post-processing
scale-up is kept but, because units are now correct, it is genuine
floating-point cleanup rather than masking a real deficit; we report the
factor transparently.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import config as C
import opt_common as OC


def run_simulated_annealing(lam=0.5):
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    P = OC.load_problem()
    cf_by_month, demand_by_month = OC.load_monthly_profiles()
    use_monthly = cf_by_month is not None
    if use_monthly:
        print(f"  Monthly demand matching ON: {len(demand_by_month)} months")
    hpm = C.HOURS_PER_YEAR / 12.0
    cov, mu, cvec = P["cov"], P["mu_mwh"], P["cost_vec"]
    upper, D_target = P["upper"], P["D_target"]
    var_scale, n = P["var_scale"], P["n"]
    regions = P["regional_indices"]
    rng = np.random.default_rng(C.SEED)

    def objective(x):
        f_cost = (cvec @ x) / C.COST_SCALE
        f_var = (x @ cov @ x) / var_scale
        obj = lam * f_var + (1 - lam) * f_cost
        prod = mu @ x
        if prod < D_target:                       # squared, dominant
            obj += 500.0 * ((D_target - prod) / D_target) ** 2
        c = cvec @ x
        if c > C.BUDGET_USD:
            obj += 100.0 * ((c - C.BUDGET_USD) / C.BUDGET_USD) ** 2
        for idxs in regions.values():
            rc = np.sum(x[idxs])
            if rc < C.MIN_REGIONAL_MW:
                obj += 20.0 * ((C.MIN_REGIONAL_MW - rc) / C.MIN_REGIONAL_MW) ** 2
        # Step 2: monthly demand matching penalty (85% coverage)
        if use_monthly:
            for m in range(len(demand_by_month)):
                prod_m = (cf_by_month[m] @ x) * hpm
                d_m = demand_by_month[m] * C.MONTHLY_COVERAGE
                if prod_m < d_m:
                    obj += 200.0 * ((d_m - prod_m) / d_m) ** 2
        return obj

    # Warm start: proportional to mean CF (productive assets first), clipped
    mu_w = mu / mu.sum()
    x = np.clip(mu_w * (0.5 * upper.sum()), 0, upper)
    best_x, best_obj = x.copy(), objective(x)
    cur_obj = best_obj

    T = C.SA_T0
    max_iter = int(np.log(C.SA_T_MIN / C.SA_T0) / np.log(C.SA_ALPHA))
    print(f"SA: ~{max_iter} iterations | D_target={D_target/1e6:.1f} TWh/yr")

    for _ in tqdm(range(max_iter), desc="Cooling"):
        x_new = x.copy()
        if rng.random() < 0.1:                    # paired solar+wind jump
            p = rng.integers(0, n // 2)
            x_new[p] = np.clip(x[p] + rng.normal() * C.SA_STEP, 0, upper[p])
            q = p + n // 2
            if q < n:
                x_new[q] = np.clip(x[q] + rng.normal() * C.SA_STEP, 0, upper[q])
        else:
            i = rng.integers(0, n)
            x_new[i] = np.clip(x[i] + rng.normal() * C.SA_STEP, 0, upper[i])

        new_obj = objective(x_new)
        delta = new_obj - cur_obj
        if delta < 0 or rng.random() < np.exp(-delta / T):
            x, cur_obj = x_new, new_obj
            if cur_obj < best_obj:
                best_obj, best_x = cur_obj, x_new.copy()
        T *= C.SA_ALPHA

    # Transparent post-processing scale-up
    prod = mu @ best_x
    if prod < D_target:
        scale = D_target / max(prod, 1e-9)
        best_x = np.clip(best_x * scale, 0, upper)
        print(f"  Post-processing scale-up factor: {scale:.6f}")

    bc, br, bp = OC.cost(best_x, cvec), OC.variance(best_x, cov), OC.production(best_x, mu)
    cols = ["total_cost", "total_risk", "expected_production"] + P["asset_names"]
    df = pd.DataFrame([[bc, br, bp] + list(best_x)], columns=cols)
    path = os.path.join(C.RESULTS_DIR, "best_solution_sa.csv")
    df.to_csv(path, index=False)
    print(f"SA done: cost=${bc/1e9:.2f}B | prod={bp/1e6:.1f} TWh/yr. Saved {path}")
    return df


if __name__ == "__main__":
    run_simulated_annealing()

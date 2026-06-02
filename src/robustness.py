"""
robustness.py
=============
Post-optimization stress testing, consistent units, ONE budget.

HYBRID choices:
  * Monte Carlo: notebook's HISTORICAL BOOTSTRAP (draw 12 consecutive real
    months) instead of the src additive-Gaussian noise. Bootstrap preserves
    the real seasonal & spatial correlation structure of weather.
  * Chance constraint: proposal-faithful P(annual production < D_min) <= eps
    with D_min = 90% of D_target (src checked P(month < target) which conflates
    monthly variability with the annual reliability the proposal asked for).
  * Scenario comparison: naive portfolios matched to the OPTIMIZED portfolio's
    own cost (the src bug used a different BUDGET here, making the comparison
    unfair). Here everything uses config.BUDGET / the optimized cost.

Outputs: results/robustness_report.csv, results/scenario_comparison.csv
"""

import os
import numpy as np
import pandas as pd

import config as C
import opt_common as OC


def load_monthly_cf():
    df = pd.read_csv(os.path.join(C.PROCESSED_DIR, "monthly_cf_matrix.csv"), index_col=0)
    return df.values.astype(float), list(df.columns), df.index.tolist()


def load_solution(filename, asset_names):
    path = os.path.join(C.RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  [WARNING] {filename} missing, skipping")
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    if "lambda" in df.columns and len(df) > 1:
        pick = df.iloc[[len(df) // 2]]
    elif "total_risk" in df.columns and len(df) > 1:
        s = df.sort_values("total_risk").reset_index(drop=True)
        pick = s.iloc[[len(s) // 2]]
    else:
        pick = df.iloc[[0]]
    cols = [c for c in asset_names if c in pick.columns]
    if len(cols) != len(asset_names):
        print(f"  [WARNING] {filename}: {len(cols)}/{len(asset_names)} assets matched")
        return None
    return pick[cols].values[0]


def monte_carlo_bootstrap(x, monthly_cf, D_target, n_scenarios=C.MC_N_SCENARIOS):
    """Draw 12 consecutive real months; annualise with HOURS_PER_MONTH."""
    rng = np.random.default_rng(C.SEED)
    T = monthly_cf.shape[0]
    annual = []
    for _ in range(n_scenarios):
        start = rng.integers(0, T - 12)
        cf_year = monthly_cf[start:start + 12]          # (12, 162) real weather
        annual.append(float((cf_year @ x * C.HOURS_PER_MONTH).sum()))
    annual = np.array(annual)
    return {
        "mean_twh": annual.mean() / 1e6,
        "p5_twh": np.percentile(annual, 5) / 1e6,
        "p95_twh": np.percentile(annual, 95) / 1e6,
        "pct_below_target": float((annual < D_target).mean()),
        "_annual": annual,
    }


def chance_constraint(annual, D_target):
    D_min = D_target * C.CHANCE_D_MIN_FRACTION
    p = float((annual < D_min).mean())
    return {"cc_D_min_twh": D_min / 1e6, "cc_prob": p,
            "cc_epsilon": C.CHANCE_EPSILON, "cc_pass": bool(p <= C.CHANCE_EPSILON)}


def drought_year(x, monthly_cf, D_target):
    monthly_prod = monthly_cf @ x * C.HOURS_PER_MONTH
    worst = np.argsort(monthly_prod)[:12]
    drought_annual = float(monthly_prod[worst].sum())
    return {"drought_twh": drought_annual / 1e6,
            "drought_coverage_pct": drought_annual / D_target * 100}


def fossil_avoided_cost(mean_twh):
    mwh = mean_twh * 1e6
    base = mwh * C.FOSSIL_BASE_USD_PER_MWH
    shock = mwh * C.FOSSIL_SHOCK_USD_PER_MWH
    return {"avoided_base_busd": base / 1e9, "avoided_shock_busd": shock / 1e9,
            "shock_premium_busd": (shock - base) / 1e9}


def scenario_comparison(asset_names, monthly_cf, D_target, optimized_x):
    """Naive portfolios matched to the optimized portfolio's OWN cost (fair)."""
    cvec = np.array([C.COST_SOLAR_PER_MW if "solar" in a else C.COST_WIND_PER_MW
                     for a in asset_names])
    n = len(asset_names)
    budget = float(cvec @ optimized_x)              # match optimized cost, not a separate constant
    solar_mask = np.array(["solar" in a for a in asset_names])
    wind_mask = ~solar_mask

    def uniform(kind):
        x = np.zeros(n)
        if kind == "solar":
            x[solar_mask] = budget / (solar_mask.sum() * C.COST_SOLAR_PER_MW)
        elif kind == "wind":
            x[wind_mask] = budget / (wind_mask.sum() * C.COST_WIND_PER_MW)
        else:  # equal
            x[solar_mask] = (budget / 2) / (solar_mask.sum() * C.COST_SOLAR_PER_MW)
            x[wind_mask] = (budget / 2) / (wind_mask.sum() * C.COST_WIND_PER_MW)
        return x

    portfolios = {"solar_only": uniform("solar"), "wind_only": uniform("wind"),
                  "equal_split": uniform("equal"), "optimized": optimized_x}
    out = {}
    for name, x in portfolios.items():
        mp = monthly_cf @ x * C.HOURS_PER_MONTH
        annual = mp.sum() / 12 * 12  # = sum over the 12-month mean cycle proxy
        out[name] = {"mean_annual_twh": float(mp.mean() * 12 / 1e6),
                     "std_monthly_twh": float(mp.std() / 1e6),
                     "coverage_pct": float(mp.mean() * 12 / D_target * 100)}
    return out


def run_robustness():
    print("=" * 60)
    print("ROBUSTNESS ANALYSIS (hybrid)")
    print("=" * 60)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    monthly_cf, asset_names, _ = load_monthly_cf()
    D_target = C.annual_demand_target_mwh()
    print(f"Target: {D_target/1e6:.1f} TWh/yr | budget ${C.BUDGET_USD/1e9:.0f}B")

    solutions = {
        "SLSQP": load_solution("efficient_frontier_slsqp.csv", asset_names),
        "NSGA2": load_solution("pareto_front_nsga2.csv", asset_names),
        "SA": load_solution("best_solution_sa.csv", asset_names),
    }
    solutions = {k: v for k, v in solutions.items() if v is not None}
    if not solutions:
        print("[ERROR] No optimizer results found.")
        return

    report = []
    for method, x in solutions.items():
        mc = monte_carlo_bootstrap(x, monthly_cf, D_target)
        cc = chance_constraint(mc.pop("_annual"), D_target)
        dr = drought_year(x, monthly_cf, D_target)
        fc = fossil_avoided_cost(mc["mean_twh"])
        print(f"\n--- {method} ---")
        print(f"  MC mean {mc['mean_twh']:.1f} TWh  (P5 {mc['p5_twh']:.1f})  "
              f"target {D_target/1e6:.1f}")
        print(f"  Chance P(annual<D_min)={cc['cc_prob']*100:.1f}% "
              f"(<= {C.CHANCE_EPSILON*100:.0f}%): {'PASS' if cc['cc_pass'] else 'FAIL'}")
        print(f"  Drought coverage {dr['drought_coverage_pct']:.1f}%  | "
              f"avoided ${fc['avoided_base_busd']:.1f}B/yr (+50%: "
              f"${fc['avoided_shock_busd']:.1f}B)")
        report.append({"method": method, **mc, **cc, **dr, **fc})

    pd.DataFrame(report).to_csv(os.path.join(C.RESULTS_DIR, "robustness_report.csv"),
                                index=False)

    best = max(report, key=lambda r: r["mean_twh"])["method"]
    sc = scenario_comparison(asset_names, monthly_cf, D_target, solutions[best])
    print(f"\n--- Scenario comparison (optimized = {best}) ---")
    for name, s in sc.items():
        mark = "*" if name == "optimized" else " "
        print(f"  {mark} {name:12s}: mean {s['mean_annual_twh']:6.1f} TWh  "
              f"cov {s['coverage_pct']:5.1f}%  std {s['std_monthly_twh']:.2f}")
    pd.DataFrame([{"scenario": k, **v} for k, v in sc.items()]).to_csv(
        os.path.join(C.RESULTS_DIR, "scenario_comparison.csv"), index=False)
    print("\nSaved robustness_report.csv, scenario_comparison.csv")


if __name__ == "__main__":
    run_robustness()

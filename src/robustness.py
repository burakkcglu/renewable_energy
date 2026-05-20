"""
robustness.py
==============
Stress testing for the renewable energy portfolio after optimization:

  1. Monte Carlo simulation (1000 weather scenarios per method)
  2. Chance constraint check (Eq. 6: P(monthly < target) <= 5%)
  3. Drought year analysis (worst 12-month window)
  4. Fossil fuel avoided-cost (baseline + 50% price shock)
  5. Scenario comparison (Solar-only / Wind-only / Equal-split / Optimized)

Input files from data/processed/:
  - monthly_cf_matrix.csv      (252 x 162) monthly capacity factor data
  - province_features.csv      (target demand)

Input files from results/:
  - efficient_frontier_slsqp.csv
  - pareto_front_nsga2.csv
  - best_solution_sa.csv

Output files in results/:
  - robustness_report.csv      per-method stress test results
  - scenario_comparison.csv    naive vs optimized strategies
"""
 
import os
import numpy as np
import pandas as pd
 
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
 
# Same constants as the optimizers (so the comparison is fair)
BUDGET = 50000.0
COST_SOLAR_MW = 0.8
COST_WIND_MW = 1.2
 
# Fossil fuel cost for avoided-cost calculation
FOSSIL_LCOE = 0.075          # USD/kWh, typical natural gas cost
FOSSIL_LCOE_SHOCK = 0.1125   # +50% price shock scenario
 
# Chance constraint limit from the proposal
EPSILON = 0.05  # P(monthly production < target) must be <= 5%
 
 
def normalize_name(text):
    return text.translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosucgiosu")).lower().strip()
 
 
# =====================================================================
# DATA LOADING
# =====================================================================
 
def load_data():
    """Load monthly capacity factor matrix, asset names, and target demand."""
    monthly_path = os.path.join(PROCESSED_DIR, "monthly_cf_matrix.csv")
    monthly_df = pd.read_csv(monthly_path, index_col=0)
    monthly_cf = monthly_df.values                  # shape (T, 162)
    asset_names = list(monthly_df.columns)
 
    # Target demand (same formula as the optimizers)
    df_features = pd.read_csv(os.path.join(PROCESSED_DIR, "province_features.csv"))
    df_features = df_features.drop_duplicates(subset=['province'], keep='last')
    total_demand_mwh = 510.5 * 1e6  # TWh to MWh
    target_demand_mw = (total_demand_mwh / (365 * 24)) * 0.25  # 25% of national demand

 
    print(f"Monthly CF matrix: {monthly_cf.shape}")
    print(f"Target demand: {target_demand_mw:.2f} MW (avg, 25% of national)")
    return monthly_cf, asset_names, target_demand_mw
 
 
def load_solution(filename, asset_names):
    """
    Load a saved result and pick a representative portfolio.
    For files with many points, we take the middle-risk
    solution for stress testing.
    """
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  [WARNING] {filename} not found, skipping")
        return None
 
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
 
    # Pick a middle point for stress testing
    if 'lambda' in df.columns and len(df) > 1:
        # Frontier file, pick middle lambda
        df_pick = df.iloc[[len(df) // 2]]
    elif 'total_risk' in df.columns and len(df) > 1:
        # Pareto file, pick middle risk
        df_sorted = df.sort_values('total_risk').reset_index(drop=True)
        df_pick = df_sorted.iloc[[len(df_sorted) // 2]]
    else:
        # Single solution file (SA)
        df_pick = df.iloc[[0]]
 
    # Get asset columns in the same order as the monthly CF matrix
    available_cols = [c for c in asset_names if c in df_pick.columns]
    if len(available_cols) != len(asset_names):
        print(f"  [WARNING] {filename}: only {len(available_cols)}/{len(asset_names)} "
              f"asset columns matched (possible name mismatch)")
        return None
 
    x = df_pick[available_cols].values[0]
    return x
 
 
# =====================================================================
# 1. MONTE CARLO
# =====================================================================
 
def monte_carlo(x, monthly_cf, target_demand_mw, n_scenarios=1000,
                noise_std=0.15, seed=42):
    """
    Add random noise (15% std) to monthly capacity factors to
    simulate different weather years across 1000 scenarios.

    Returns:
      - production stats across scenarios
      - P(monthly production < target) for chance constraint check
    """
    rng = np.random.default_rng(seed)
    T = monthly_cf.shape[0]
    scenario_avg_prods = []
    scenario_shortfalls = []
    total_months = 0
    months_below_target = 0
 
    for _ in range(n_scenarios):
        noise = rng.normal(1.0, noise_std, monthly_cf.shape)
        scenario_cf = np.clip(monthly_cf * noise, 0.0, 1.0)
        monthly_prod = scenario_cf @ x                     # shape (T,)
        avg_prod = monthly_prod.mean()
 
        scenario_avg_prods.append(avg_prod)
        scenario_shortfalls.append(max(0.0, target_demand_mw - avg_prod))
        months_below_target += int(np.sum(monthly_prod < target_demand_mw))
        total_months += T
 
    return {
        'n_scenarios': n_scenarios,
        'mean_production_mw': float(np.mean(scenario_avg_prods)),
        'std_production_mw': float(np.std(scenario_avg_prods)),
        'p5_production_mw': float(np.percentile(scenario_avg_prods, 5)),
        'p95_production_mw': float(np.percentile(scenario_avg_prods, 95)),
        'mean_shortfall_mw': float(np.mean(scenario_shortfalls)),
        'pct_scenarios_with_shortfall':
            float(sum(1 for s in scenario_shortfalls if s > 0) / n_scenarios),
        'pct_months_below_target': float(months_below_target / total_months),
    }
 
 
# =====================================================================
# 2. CHANCE CONSTRAINT (Proposal Eq. 6)
# =====================================================================
 
def chance_constraint_check(mc_result, epsilon=EPSILON):
    """
    Check the proposal's chance constraint:
        P(monthly production < target) <= epsilon
    """
    p = mc_result['pct_months_below_target']
    return {
        'cc_epsilon_target': epsilon,
        'cc_observed_probability': p,
        'cc_satisfied': bool(p <= epsilon),
        'cc_margin': float(epsilon - p),  # positive means constraint is met
    }
 
 
# =====================================================================
# 3. DROUGHT YEAR
# =====================================================================
 
def drought_year_analysis(x, monthly_cf, target_demand_mw):
    """
    Find the worst 12-month window in a row, the 'drought year'
    where this portfolio produces the least.
    """
    monthly_prod = monthly_cf @ x
    T = len(monthly_prod)
    if T < 12:
        return None
 
    worst_avg = np.inf
    worst_start = 0
    for i in range(T - 11):
        window_avg = monthly_prod[i:i + 12].mean()
        if window_avg < worst_avg:
            worst_avg = window_avg
            worst_start = i
 
    coverage_pct = (worst_avg / target_demand_mw * 100) if target_demand_mw > 0 else 0
    return {
        'drought_worst12mo_avg_mw': float(worst_avg),
        'drought_worst12mo_coverage_pct': float(coverage_pct),
        'drought_window_start_month': int(worst_start),
    }
 
 
# =====================================================================
# 4. FOSSIL AVOIDED COST
# =====================================================================
 
def fossil_avoided_cost(mean_production_mw):
    """
    Calculate how much fossil fuel this portfolio replaces, and
    what that is worth in dollars.

    Baseline LCOE = $0.075/kWh (natural gas)
    Shock LCOE    = $0.1125/kWh (+50%, price spike)

    We measure the money the country does not spend on imported gas.
    """
    annual_mwh = mean_production_mw * 8760
    # LCOE is $/kWh, production is MWh, so multiply by 1000 to get $
    avoided_base = annual_mwh * FOSSIL_LCOE * 1000
    avoided_shock = annual_mwh * FOSSIL_LCOE_SHOCK * 1000
    return {
        'annual_production_twh': float(annual_mwh / 1e6),
        'avoided_cost_base_musd': float(avoided_base / 1e6),
        'avoided_cost_shock_musd': float(avoided_shock / 1e6),
        'shock_premium_musd': float((avoided_shock - avoided_base) / 1e6),
    }
 
 
# =====================================================================
# 5. SCENARIO COMPARISON
# =====================================================================
 
def scenario_comparison(asset_names, monthly_cf, target_demand_mw, optimized_x=None):
    """
    Compare four simple allocation strategies (same total budget):
      - Solar-only:  spread budget equally across all solar assets
      - Wind-only:   spread budget equally across all wind assets
      - Equal-split: 50% solar + 50% wind, equal within each
      - Optimized:   the actual optimized portfolio (if given)

    Reports mean/std monthly production and coverage % of target.
    """
    n_solar = sum(1 for a in asset_names if 'solar' in a)
    n_wind = sum(1 for a in asset_names if 'wind' in a)
 
    def build_uniform(only):
        x = np.zeros(len(asset_names))
        if only == 'solar':
            mw = BUDGET / (n_solar * COST_SOLAR_MW)
            for i, a in enumerate(asset_names):
                if 'solar' in a:
                    x[i] = mw
        elif only == 'wind':
            mw = BUDGET / (n_wind * COST_WIND_MW)
            for i, a in enumerate(asset_names):
                if 'wind' in a:
                    x[i] = mw
        elif only == 'equal':
            half = BUDGET / 2
            solar_mw = half / (n_solar * COST_SOLAR_MW)
            wind_mw = half / (n_wind * COST_WIND_MW)
            for i, a in enumerate(asset_names):
                x[i] = solar_mw if 'solar' in a else wind_mw
        return x
 
    scenarios = {
        'solar_only': build_uniform('solar'),
        'wind_only': build_uniform('wind'),
        'equal_split': build_uniform('equal'),
    }
    if optimized_x is not None:
        scenarios['optimized'] = optimized_x
 
    summary = {}
    for name, x in scenarios.items():
        monthly_prod = monthly_cf @ x
        summary[name] = {
            'mean_mw': float(monthly_prod.mean()),
            'std_mw': float(monthly_prod.std()),
            'coverage_pct': float(monthly_prod.mean() / target_demand_mw * 100),
            'pct_months_below_target': float(np.mean(monthly_prod < target_demand_mw)),
        }
    return summary
 
 
# =====================================================================
# MAIN ORCHESTRATOR
# =====================================================================
 
def run_robustness():
    print("=" * 70)
    print("ROBUSTNESS ANALYSIS -- Portfolio stress testing")
    print("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)
 
    monthly_cf, asset_names, target_mw = load_data()
 
    # Load each optimizer's solution
    solutions = {
        'SLSQP': load_solution("efficient_frontier_slsqp.csv", asset_names),
        'NSGA2': load_solution("pareto_front_nsga2.csv", asset_names),
        'SA':    load_solution("best_solution_sa.csv", asset_names),
    }
    solutions = {k: v for k, v in solutions.items() if v is not None}
    if not solutions:
        print("[ERROR] No optimization results found. Run optimizers first.")
        return
 
    print(f"\nFound solutions: {list(solutions.keys())}")
 
    # --- Run stress tests on each solution ---
    full_report = []
    for method, x in solutions.items():
        print(f"\n--- {method} ---")
        mc = monte_carlo(x, monthly_cf, target_mw)
        cc = chance_constraint_check(mc)
        dr = drought_year_analysis(x, monthly_cf, target_mw)
        fc = fossil_avoided_cost(mc['mean_production_mw'])
 
        print(f"  Monte Carlo (1000 scenarios):")
        print(f"    Mean production: {mc['mean_production_mw']:.0f} MW "
              f"(target: {target_mw:.0f})")
        print(f"    P5 / P95:        {mc['p5_production_mw']:.0f} / "
              f"{mc['p95_production_mw']:.0f} MW")
        print(f"    Scenarios with shortfall: "
              f"{mc['pct_scenarios_with_shortfall'] * 100:.1f}%")
        print(f"  Chance constraint P(month < target) = "
              f"{cc['cc_observed_probability'] * 100:.1f}% "
              f"(target <= {EPSILON * 100:.0f}%): "
              f"{'PASS' if cc['cc_satisfied'] else 'FAIL'}")
        if dr:
            print(f"  Drought year coverage: "
                  f"{dr['drought_worst12mo_coverage_pct']:.1f}% of target")
        print(f"  Avoided fossil cost (base):  "
              f"${fc['avoided_cost_base_musd']:.0f} M/year")
        print(f"  Avoided fossil cost (+50% shock): "
              f"${fc['avoided_cost_shock_musd']:.0f} M/year")
        print(f"  Shock premium: ${fc['shock_premium_musd']:.0f} M/year extra savings")
 
        row = {'method': method, **mc, **cc, **fc}
        if dr:
            row.update(dr)
        full_report.append(row)
 
    pd.DataFrame(full_report).to_csv(
        os.path.join(RESULTS_DIR, "robustness_report.csv"), index=False)
 
    # --- Scenario Comparison (uses the best-performing method's solution) ---
    print(f"\n--- Scenario Comparison (Naive vs Optimized) ---")
    # Pick the portfolio with the highest mean production for comparison
    best_method = max(full_report, key=lambda r: r['mean_production_mw'])['method']
    optimized_x = solutions[best_method]
    print(f"  Optimized = {best_method}")
 
    sc = scenario_comparison(asset_names, monthly_cf, target_mw,
                              optimized_x=optimized_x)
    for name, stats in sc.items():
        marker = "*" if name == 'optimized' else " "
        print(f"  {marker} {name:13s}: "
              f"mean={stats['mean_mw']:7.0f} MW  "
              f"std={stats['std_mw']:6.0f}  "
              f"coverage={stats['coverage_pct']:5.1f}%  "
              f"P(<target)={stats['pct_months_below_target'] * 100:5.1f}%")
 
    sc_rows = [{'scenario': k, **v} for k, v in sc.items()]
    pd.DataFrame(sc_rows).to_csv(
        os.path.join(RESULTS_DIR, "scenario_comparison.csv"), index=False)
 
    print(f"\nSaved to {RESULTS_DIR}/:")
    print(f"   - robustness_report.csv")
    print(f"   - scenario_comparison.csv")
    print("=" * 70)
 
 
if __name__ == "__main__":
    run_robustness()
"""
analysis.py
===========
Strategic analysis + Eurostat benchmarking + report-defense notes, wired to the
hybrid pipeline (config units, new robustness column names).

Reads from results/ and data/processed/. Prints a terminal report and saves a
plain-text summary to results/analysis_report.txt.
"""

import os
import numpy as np
import pandas as pd

import config as C
import opt_common as OC


def load_eurostat_benchmarks(reference_year=None):
    """
    Eurostat 'Renewable energy - overall' share (nrg_ind_ren): renewables as a
    % of GROSS FINAL energy consumption (heating + transport + electricity),
    NOT electricity-only. Fallback values below match this overall metric for
    2024 (EU ~25%, DE ~22%, ES ~25%, IT ~19%) — used only if the file is absent.
    """
    path = os.path.join(C.PROCESSED_DIR, "eurostat_renewable_share.csv")
    fallback = {"eu_avg": 25.2, "germany": 22.5, "spain": 25.4, "italy": 19.4,
                "year": 2024, "source": "hardcoded fallback (overall RES share)"}
    if not os.path.exists(path):
        return fallback
    df = pd.read_csv(path)

    if "Yil" in df.columns:
        years = pd.to_numeric(df["Yil"], errors="coerce").dropna()
        if reference_year is None and len(years) > 0:
            reference_year = int(years.max())
    if reference_year is None:
        return fallback
    dl = df[df["Yil"] == reference_year] if "Yil" in df.columns else df

    def share(name):
        row = dl[dl["Country"] == name]
        return float(row["RenewableShare_Pct"].iloc[0]) if len(row) else None

    return {
        "eu_avg": share("European Union - 27 countries (from 2020)") or fallback["eu_avg"],
        "germany": share("Germany") or fallback["germany"],
        "spain": share("Spain") or fallback["spain"],
        "italy": share("Italy") or fallback["italy"],
        "year": reference_year or fallback["year"],
        "source": "Eurostat (overall RES share)",
    }


def _rep_solution(df, assets):
    if df is None or len(df) == 0:
        return None
    if "lambda" in df.columns and len(df) > 1:
        pick = df.iloc[[len(df) // 2]]
    elif "total_risk" in df.columns and len(df) > 1:
        pick = df.sort_values("total_risk").reset_index(drop=True).iloc[[len(df) // 2]]
    else:
        pick = df.iloc[[0]]
    cols = [c for c in assets if c in pick.columns]
    return pick[cols].values[0] if len(cols) == len(assets) else None


def run_advanced_analysis():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 64)
    out("STRATEGIC ANALYSIS & REPORT NOTES (hybrid)")
    out("=" * 64)

    P = None
    try:
        P = OC.load_problem()
    except Exception as e:
        out(f"[WARNING] could not load problem data: {e}")

    files = {"SLSQP": "efficient_frontier_slsqp.csv",
             "NSGA-II": "pareto_front_nsga2.csv",
             "SA": "best_solution_sa.csv"}
    if P is not None:
        assets = P["asset_names"]
        out("\n1. PORTFOLIO SUMMARY (representative solution per method)")
        out(f"   Target {C.annual_demand_target_mwh()/1e6:.1f} TWh/yr | "
            f"budget ${C.BUDGET_USD/1e9:.0f}B")
        for name, fname in files.items():
            path = os.path.join(C.RESULTS_DIR, fname)
            if not os.path.exists(path):
                continue
            x = _rep_solution(pd.read_csv(path), assets)
            if x is None:
                continue
            solar = x[np.array(["solar" in a for a in assets])].sum() / 1000
            wind = x[np.array(["wind" in a for a in assets])].sum() / 1000
            out(f"   {name:8s}: solar {solar:6.1f} GW | wind {wind:6.1f} GW | "
                f"cost ${OC.cost(x, P['cost_vec'])/1e9:5.1f}B | "
                f"prod {OC.production(x, P['mu_mwh'])/1e6:6.1f} TWh | "
                f"solar share {solar/(solar+wind)*100:4.0f}%")

    rob_path = os.path.join(C.RESULTS_DIR, "robustness_report.csv")
    if os.path.exists(rob_path):
        rob = pd.read_csv(rob_path)
        out("\n2. ROBUSTNESS")
        for _, r in rob.iterrows():
            out(f"   {r['method']:8s}: MC mean {r['mean_twh']:6.1f} TWh "
                f"(P5 {r['p5_twh']:.1f}) | chance P(<D_min) {r['cc_prob']*100:4.1f}% "
                f"{'PASS' if r['cc_pass'] else 'FAIL'} | drought "
                f"{r['drought_coverage_pct']:4.1f}% | avoided "
                f"${r['avoided_base_busd']:.1f}B/yr (+50%: ${r['avoided_shock_busd']:.1f}B)")

    scen_path = os.path.join(C.RESULTS_DIR, "scenario_comparison.csv")
    if os.path.exists(scen_path):
        sc = pd.read_csv(scen_path)
        out("\n3. DIVERSIFICATION (naive vs optimized, same budget)")
        for _, r in sc.iterrows():
            out(f"   {r['scenario']:12s}: mean {r['mean_annual_twh']:6.1f} TWh | "
                f"coverage {r['coverage_pct']:5.1f}% | std {r['std_monthly_twh']:.2f}")

    bm = load_eurostat_benchmarks()
    out(f"\n4. EU BENCHMARK ({bm['year']}, {bm['source']})")
    out(f"   Overall RES share (% of gross final energy): "
        f"EU {bm['eu_avg']:.1f}% | Germany {bm['germany']:.1f}% | "
        f"Spain {bm['spain']:.1f}% | Italy {bm['italy']:.1f}%")
    out(f"   NOTE: this Eurostat figure is share of TOTAL final energy "
        f"(heat+transport+electricity). Our {C.RENEWABLE_SHARE*100:.0f}% target is "
        f"of national ELECTRICITY demand from NEW solar+wind only, so the two are "
        f"different bases and not directly comparable — use the EU numbers as "
        f"directional context for where Turkey sits, not a like-for-like target.")

    out("\n5. REPORT-DEFENSE NOTES (proposal -> what we actually did)")
    out("   * Solar CF: dropped the eta multiplier from the proposal formula. "
        "Rated MW already embeds eta at STC; including it double-counts and "
        "shrinks CF ~5.6x. Result: realistic solar CF.")
    out("   * Wind CF: replaced fixed 1/7 shear with a dynamic per-datapoint "
        "exponent from WS10M+WS50M, kept a physical cubic power curve.")
    out("   * Chance constraint (Eq.6): post-hoc verification "
        "(P(annual<D_min)<=5%) rather than an in-loop constraint.")
    out("   * NSGA-II now genuinely 3-objective (risk, cost, avoided CO2): a true "
        "Pareto surface SLSQP cannot trace, plus the 0-or->=50 MW repair rule.")
    out("   * Single config.py: one BUDGET everywhere (fixes the earlier "
        "80k-vs-50k inconsistency).")

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(C.RESULTS_DIR, "analysis_report.txt"), "w") as f:
        f.write("\n".join(lines))
    out("\nSaved results/analysis_report.txt")


if __name__ == "__main__":
    run_advanced_analysis()

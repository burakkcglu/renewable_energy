"""
make_figures.py
===============
Creates and saves project charts to the figures/ folder.

Run: python src/make_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import opt_common as OC

os.makedirs(C.FIGURES_DIR, exist_ok=True)

SOLAR_C = "#E8A33D"
WIND_C = "#4A7FB5"
OPT_C = "#3C9A5F"
RISK_C = "#C0504D"


def _save(fig, name):
    path = os.path.join(C.FIGURES_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_efficient_frontier():
    slsqp_p = os.path.join(C.RESULTS_DIR, "efficient_frontier_slsqp.csv")
    nsga_p = os.path.join(C.RESULTS_DIR, "pareto_front_nsga2.csv")
    if not os.path.exists(slsqp_p):
        return
    s = pd.read_csv(slsqp_p)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(s["total_risk"], s["total_cost"] / 1e9, "o--", color=SOLAR_C,
            lw=2, ms=7, label="SLSQP efficient frontier", zorder=3)
    if os.path.exists(nsga_p):
        n = pd.read_csv(nsga_p)
        co2_col = "avoided_co2_tonnes" if "avoided_co2_tonnes" in n.columns else "avoided_co2"
        sc = ax.scatter(n["total_risk"], n["total_cost"] / 1e9,
                        c=n[co2_col] / 1e6, cmap="viridis", s=30, alpha=0.75,
                        label="NSGA-II Pareto front", zorder=2)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Avoided CO₂ (Mt/yr)")
    ax.set_xlabel("Portfolio risk  (xᵀ Σ x)")
    ax.set_ylabel("Investment cost (Billion USD)")
    ax.set_title("Efficient Frontier — SLSQP vs NSGA-II")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "efficient_frontier.png")


def _load_rep_solution(fname, assets):
    path = os.path.join(C.RESULTS_DIR, fname)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    if "lambda" in df.columns and len(df) > 1:
        pick = df.iloc[[len(df) // 2]]
    elif "total_risk" in df.columns and len(df) > 1:
        pick = df.sort_values("total_risk").reset_index(drop=True).iloc[[len(df) // 2]]
    else:
        pick = df.iloc[[0]]
    cols = [c for c in assets if c in pick.columns]
    if len(cols) != len(assets):
        return None
    return pick[cols].values[0]


def fig_method_comparison():
    P = OC.load_problem()
    assets = P["asset_names"]
    methods = {
        "SLSQP": _load_rep_solution("efficient_frontier_slsqp.csv", assets),
        "NSGA-II": _load_rep_solution("pareto_front_nsga2.csv", assets),
        "SA": _load_rep_solution("best_solution_sa.csv", assets),
    }
    methods = {k: v for k, v in methods.items() if v is not None}
    if not methods:
        return
    names = list(methods.keys())
    costs = [OC.cost(x, P["cost_vec"]) / 1e9 for x in methods.values()]
    prods = [OC.production(x, P["mu_mwh"]) / 1e6 for x in methods.values()]
    risks = [OC.variance(x, P["cov"]) for x in methods.values()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (costs, "Investment Cost (B USD)", C.BUDGET_USD / 1e9, "Budget", WIND_C),
        (prods, "Expected Production (TWh/yr)", C.annual_demand_target_mwh() / 1e6,
         "Target", OPT_C),
        (risks, "Portfolio Risk (xᵀΣx)", None, None, RISK_C),
    ]
    for ax, (vals, title, line, lbl, color) in zip(axes, panels):
        bars = ax.bar(names, vals, color=color, alpha=0.85, edgecolor="black", lw=0.6)
        if line is not None:
            ax.axhline(line, color="red", ls="--", lw=1.5, label=f"{lbl} ({line:.0f})")
            ax.legend(fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.2f}" if v < 1000 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9)
    fig.suptitle("Method Comparison: SLSQP vs NSGA-II vs Simulated Annealing", fontsize=13)
    fig.tight_layout()
    _save(fig, "method_comparison.png")


def fig_scenario_comparison():
    path = os.path.join(C.RESULTS_DIR, "scenario_comparison.csv")
    if not os.path.exists(path):
        return
    sc = pd.read_csv(path)
    order = ["solar_only", "wind_only", "equal_split", "optimized"]
    sc = sc.set_index("scenario").reindex([o for o in order if o in sc["scenario"].values
                                           or o in sc.index]).dropna(how="all")
    sc = sc.reset_index()
    colors = [SOLAR_C, WIND_C, "#9B7FB5", OPT_C][:len(sc)]
    target = C.annual_demand_target_mwh() / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(sc["scenario"], sc["mean_annual_twh"], color=colors,
                alpha=0.85, edgecolor="black", lw=0.6)
    axes[0].axhline(target, color="red", ls="--", lw=1.5, label=f"Target ({target:.0f})")
    axes[0].set_title("Expected Annual Production (TWh)")
    axes[0].set_ylabel("TWh/yr")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(sc["scenario"], sc["std_monthly_twh"], color=colors,
                alpha=0.85, edgecolor="black", lw=0.6)
    axes[1].set_title("Monthly Production Volatility")
    axes[1].set_ylabel("Std dev (TWh)")
    axes[1].grid(axis="y", alpha=0.3)
    fig.suptitle("Portfolio Diversification", fontsize=13)
    fig.tight_layout()
    _save(fig, "scenario_comparison.png")


def fig_cf_by_province():
    path = os.path.join(C.PROCESSED_DIR, "province_summary.csv")
    if not os.path.exists(path):
        return
    s = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ss = s.sort_values("solar_cf_mean", ascending=False)
    axes[0].bar(range(len(ss)), ss["solar_cf_mean"] * 100, color=SOLAR_C, alpha=0.85)
    axes[0].set_title("Mean Solar Capacity Factor by Province (%)")
    axes[0].set_ylabel("CF (%)")
    axes[0].set_xticks([])
    axes[0].grid(axis="y", alpha=0.3)
    sw = s.sort_values("wind_cf_mean", ascending=False)
    axes[1].bar(range(len(sw)), sw["wind_cf_mean"] * 100, color=WIND_C, alpha=0.85)
    axes[1].set_title("Mean Wind Capacity Factor by Province (%)")
    axes[1].set_ylabel("CF (%)")
    axes[1].set_xticks([])
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "solar_wind_cf.png")


def run_figures():
    fig_efficient_frontier()
    fig_method_comparison()
    fig_scenario_comparison()
    fig_cf_by_province()


if __name__ == "__main__":
    run_figures()
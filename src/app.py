"""
app.py
======
Streamlit dashboard for the renewable energy portfolio.
Reads settings from config.py and data from the results/ folder.

Run:
    streamlit run src/app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import opt_common as OC

st.set_page_config(page_title="Turkey Renewable Portfolio", page_icon="⚡", layout="wide")

# Cache data loading
@st.cache_data
def load_results():
    out = {}
    for key, fname in [("slsqp", "efficient_frontier_slsqp.csv"),
                       ("nsga2", "pareto_front_nsga2.csv"),
                       ("sa", "best_solution_sa.csv"),
                       ("robust", "robustness_report.csv"),
                       ("scenario", "scenario_comparison.csv")]:
        path = os.path.join(C.RESULTS_DIR, fname)
        out[key] = pd.read_csv(path) if os.path.exists(path) else None
    return out

@st.cache_data
def load_summary():
    path = os.path.join(C.PROCESSED_DIR, "province_summary.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_data
def load_problem_cached():
    try:
        return OC.load_problem()
    except Exception:
        return None

R = load_results()
summary = load_summary()

st.title("⚡ Turkey Renewable Energy Portfolio")
st.caption(f"Budget ${C.BUDGET_USD/1e9:.0f}B | Target "
           f"{C.annual_demand_target_mwh()/1e6:.0f} TWh/yr "
           f"({C.RENEWABLE_SHARE:.0%} of {C.NATIONAL_DEMAND_TWH:.0f} TWh)")

if R["slsqp"] is None and R["nsga2"] is None and R["sa"] is None:
    st.warning("No results found. Please run main.py first.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Efficient Frontier", "Method Comparison", "Robustness", "Capacity Factors", "Live Optimizer"]
)

# Tab 1: Frontier
with tab1:
    fig = go.Figure()
    if R["slsqp"] is not None:
        fig.add_trace(go.Scatter(
            x=R["slsqp"]["total_risk"], y=R["slsqp"]["total_cost"] / 1e9,
            mode="lines+markers", name="SLSQP frontier",
            line=dict(color="#E8A33D", dash="dash")))
    if R["nsga2"] is not None:
        fig.add_trace(go.Scatter(
            x=R["nsga2"]["total_risk"], y=R["nsga2"]["total_cost"] / 1e9,
            mode="markers", name="NSGA-II Pareto",
            marker=dict(color="#4A7FB5", size=6, opacity=0.6)))
    fig.update_layout(xaxis_title="Portfolio risk (xᵀΣx)",
                      yaxis_title="Investment cost (B USD)",
                      title="Efficient Frontier", height=520)
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Method comparison
with tab2:
    P = load_problem_cached()
    if P is None:
        st.info("Processed data not available.")
    else:
        assets = P["asset_names"]

        def rep(df):
            if df is None or len(df) == 0:
                return None
            if "lambda" in df.columns and len(df) > 1:
                pick = df.iloc[[len(df) // 2]]
            elif "total_risk" in df.columns and len(df) > 1:
                pick = df.sort_values("total_risk").reset_index(drop=True).iloc[[len(df)//2]]
            else:
                pick = df.iloc[[0]]
            cols = [c for c in assets if c in pick.columns]
            return pick[cols].values[0] if len(cols) == len(assets) else None

        rows = []
        for name, df in [("SLSQP", R["slsqp"]), ("NSGA-II", R["nsga2"]), ("SA", R["sa"])]:
            x = rep(df)
            if x is None:
                continue
            rows.append({
                "Method": name,
                "Cost ($B)": round(OC.cost(x, P["cost_vec"]) / 1e9, 2),
                "Production (TWh)": round(OC.production(x, P["mu_mwh"]) / 1e6, 2),
                "Risk (xᵀΣx)": round(OC.variance(x, P["cov"]), 0),
                "Solar (GW)": round(x[["solar" in a for a in assets]].sum() / 1000, 2),
                "Wind (GW)": round(x[["wind" in a for a in assets]].sum() / 1000, 2),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# Tab 3: Robustness
with tab3:
    if R["robust"] is not None:
        st.subheader("Stress-test report")
        st.dataframe(R["robust"], use_container_width=True, hide_index=True)
    if R["scenario"] is not None:
        st.subheader("Naive vs optimized (same budget)")
        sc = R["scenario"]
        fig = px.bar(sc, x="scenario", y="mean_annual_twh",
                     color="scenario", title="Expected annual production (TWh)")
        fig.add_hline(y=C.annual_demand_target_mwh() / 1e6, line_dash="dash",
                      line_color="red", annotation_text="Target")
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Capacity factors
with tab4:
    if summary is not None:
        c1, c2 = st.columns(2)
        with c1:
            top_s = summary.nlargest(15, "solar_cf_mean")
            st.plotly_chart(px.bar(top_s, x="solar_cf_mean", y="province",
                                   orientation="h", title="Top 15 solar CF",
                                   color_discrete_sequence=["#E8A33D"]),
                            use_container_width=True)
        with c2:
            top_w = summary.nlargest(15, "wind_cf_mean")
            st.plotly_chart(px.bar(top_w, x="wind_cf_mean", y="province",
                                   orientation="h", title="Top 15 wind CF",
                                   color_discrete_sequence=["#4A7FB5"]),
                            use_container_width=True)
    else:
        st.info("province_summary.csv not found. Run data_processing first.")

# Tab 5: Live Optimizer
with tab5:
    st.subheader("Live SLSQP")
    st.caption("Change values to re-run the Markowitz QP model instantly.")

    from scipy.optimize import minimize as _minimize

    P = load_problem_cached()
    if P is None:
        st.info("Data not available. Run data_processing first.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            budget_b = st.slider("Budget (Billion USD)", 40, 150,
                                 int(C.BUDGET_USD / 1e9), step=5)
        with c2:
            share_pct = st.slider("Renewable share of national demand (%)",
                                  10, 50, int(C.RENEWABLE_SHARE * 100), step=5)
        with c3:
            lam = st.slider("Risk penalty (λ)", 0.0, 1.0, 0.5, step=0.05)

        budget_usd = budget_b * 1e9
        D_target = C.NATIONAL_DEMAND_TWH * 1e6 * (share_pct / 100.0)

        cov, mu, cvec = P["cov"], P["mu_mwh"], P["cost_vec"]
        upper, var_scale, n = P["upper"], P["var_scale"], P["n"]
        assets = P["asset_names"]

        def obj(x, lm):
            return lm * (x @ cov @ x) / var_scale + (1 - lm) * (cvec @ x) / budget_usd

        def grad(x, lm):
            return lm * (2 * cov @ x) / var_scale + (1 - lm) * cvec / budget_usd

        bounds = tuple((0.0, u) for u in upper)
        cons = [
            {"type": "ineq", "fun": lambda x: mu @ x - D_target, "jac": lambda x: mu},
            {"type": "ineq", "fun": lambda x: budget_usd - cvec @ x, "jac": lambda x: -cvec},
        ]
        for idxs in P["regional_indices"].values():
            cons.append({"type": "ineq",
                         "fun": lambda x, i=idxs: np.sum(x[i]) - C.MIN_REGIONAL_MW})

        half = upper / 2.0
        x0 = np.where(np.arange(n) < n // 2, half, half * 0.3)

        with st.spinner("Solving..."):
            res = _minimize(obj, x0, args=(lam,), jac=grad, method="SLSQP",
                            bounds=bounds, constraints=cons,
                            options={"maxiter": 600, "ftol": 1e-8})

        x = res.x
        prod = OC.production(x, mu)
        cost = OC.cost(x, cvec)
        risk = OC.variance(x, cov)
        feasible = (prod >= D_target - 1e-3 * D_target) and (cost <= budget_usd + 1)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Production", f"{prod/1e6:.1f} TWh",
                  f"target {D_target/1e6:.0f}")
        m2.metric("Cost", f"${cost/1e9:.1f}B", f"budget {budget_b}")
        m3.metric("Risk (xᵀΣx)", f"{risk:.2e}")
        solar_gw = x[np.array(['solar' in a for a in assets])].sum() / 1000
        wind_gw = x[np.array(['wind' in a for a in assets])].sum() / 1000
        m4.metric("Solar / Wind", f"{solar_gw:.0f} / {wind_gw:.0f} GW")

        if not feasible:
            st.error("No valid portfolio found. Try increasing the budget.")
        else:
            st.success(f"Success. Solar share: {solar_gw/(solar_gw+wind_gw)*100:.0f}%.")

        # Top-15 province allocation
        alloc = pd.DataFrame({
            "province": [a.rsplit('_', 1)[0] for a in assets],
            "source": [a.rsplit('_', 1)[1] for a in assets],
            "MW": x,
        })
        alloc = alloc[alloc["MW"] > 1]
        top = (alloc.groupby(["province", "source"])["MW"].sum()
               .reset_index().sort_values("MW", ascending=False).head(20))
        if len(top):
            fig = px.bar(top, x="MW", y="province", color="source", orientation="h",
                         color_discrete_map={"solar": "#E8A33D", "wind": "#4A7FB5"},
                         title="Top province allocations (MW)")
            fig.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
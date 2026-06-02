"""
optimization_nsga2.py
=====================
Multi-objective (risk vs cost) genetic algorithm via pymoo NSGA-II.

HYBRID: keeps the src MinimumThresholdRepair operator (0 < x < 50 MW -> 0),
which enforces the proposal's "install 0 or at least 50 MW" rule and gives the
non-convex justification for using NSGA-II at all. Objectives/constraints use
the shared absolute units from opt_common, so the Pareto front is directly
comparable to the SLSQP frontier.
"""

import os
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.repair import Repair
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator

import config as C
import opt_common as OC


class MinimumThresholdRepair(Repair):
    """0 < x < threshold MW -> 0 (avoid tiny unviable allocations)."""
    def __init__(self, threshold=C.NSGA_MIN_THRESHOLD_MW):
        super().__init__()
        self.threshold = threshold

    def _do(self, problem, X, **kwargs):
        X[np.logical_and(X > 0, X < self.threshold)] = 0.0
        return X


class RenewablePortfolioProblem(Problem):
    def __init__(self, P, cf_by_month=None, demand_by_month=None):
        self.P = P
        self.cf_by_month = cf_by_month
        self.demand_by_month = demand_by_month
        n_monthly = 0 if cf_by_month is None else len(demand_by_month)
        self.n_monthly = n_monthly
        super().__init__(
            n_var=P["n"], n_obj=3,
            n_ieq_constr=2 + len(P["regional_indices"]) + n_monthly,
            xl=P["lower"], xu=P["upper"],
        )

    def _evaluate(self, X, out, *args, **kwargs):
        P = self.P
        risk = np.einsum("ij,jk,ik->i", X, P["cov"], X)
        cost = X @ P["cost_vec"]
        prod = X @ P["mu_mwh"]
        avoided = C.GRID_EMISSION_FACTOR * prod          # tCO2/yr per individual
        f3_co2 = -avoided / P["co2_scale"]
        # Normalised objectives (same scales as SLSQP) so fronts are comparable
        out["F"] = np.column_stack([risk / P["var_scale"], cost / C.COST_SCALE, f3_co2])

        G = [P["D_target"] - (X @ P["mu_mwh"]), cost - C.BUDGET_USD]
        for idxs in P["regional_indices"].values():
            G.append(C.MIN_REGIONAL_MW - np.sum(X[:, idxs], axis=1))
        # Step 2: monthly demand matching (90% coverage). pymoo wants G <= 0.
        if self.n_monthly:
            hpm = C.HOURS_PER_YEAR / 12.0
            for m in range(self.n_monthly):
                cf_m = self.cf_by_month[m]                 # (n,)
                d_m = self.demand_by_month[m] * C.MONTHLY_COVERAGE
                monthly_prod = (X @ cf_m) * hpm            # (pop,)
                G.append(d_m - monthly_prod)
        out["G"] = np.column_stack(G)


def run_nsga2():
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    P = OC.load_problem()
    cf_by_month, demand_by_month = OC.load_monthly_profiles()
    if cf_by_month is not None:
        print(f"  Monthly demand matching ON: {len(demand_by_month)} months")
    problem = RenewablePortfolioProblem(P, cf_by_month, demand_by_month)
    # Seed the population with the SLSQP solution(s) so NSGA-II starts inside
    # the (narrow) feasible region created by the monthly constraints.
    import pandas as pd
    seed_path = os.path.join(C.RESULTS_DIR, "efficient_frontier_slsqp.csv")
    sampling = FloatRandomSampling()
    if os.path.exists(seed_path):
        sdf = pd.read_csv(seed_path)
        seed_X = sdf[P["asset_names"]].values
        # Build initial population: SLSQP seeds + random fill to pop_size
        n_rand = max(0, C.NSGA_POP_SIZE - len(seed_X))
        rand_X = FloatRandomSampling().do(problem, n_rand).get("X") if n_rand else None
        init_X = np.vstack([seed_X, rand_X]) if rand_X is not None else seed_X
        init_X = np.clip(init_X, P["lower"], P["upper"])
        sampling = Population.new("X", init_X)
        print(f"  Seeded NSGA-II with {len(seed_X)} SLSQP solution(s)")
    algorithm = NSGA2(
        pop_size=C.NSGA_POP_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        repair=MinimumThresholdRepair(),
        eliminate_duplicates=True,
    )
    print(f"NSGA-II: pop={C.NSGA_POP_SIZE}, gen={C.NSGA_GENERATIONS}")
    res = minimize(problem, algorithm, ("n_gen", C.NSGA_GENERATIONS),
                   seed=C.SEED, verbose=False)

    if res.X is None:
        print("[WARNING] NSGA-II found no feasible solution.")
        return None

    cov, mu, cvec = P["cov"], P["mu_mwh"], P["cost_vec"]
    X = np.atleast_2d(res.X)
    rows = []
    for x in X:
        rows.append([OC.cost(x, cvec), OC.variance(x, cov), OC.production(x, mu),
                     OC.avoided_co2(x, mu)] + list(x))
    cols = ["total_cost", "total_risk", "expected_production", "avoided_co2"] + P["asset_names"]
    df = pd.DataFrame(rows, columns=cols).sort_values("total_risk").reset_index(drop=True)

    path = os.path.join(C.RESULTS_DIR, "pareto_front_nsga2.csv")
    df.to_csv(path, index=False)
    print(f"NSGA-II done: {len(df)} feasible Pareto solutions. Saved {path}")
    return df


if __name__ == "__main__":
    run_nsga2()

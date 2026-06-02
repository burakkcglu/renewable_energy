# Renewable Energy Portfolio Optimization for Turkey's 81 Provinces

This project applies financial portfolio theory (the Markowitz mean-variance
model) to renewable energy planning. We treat each of Turkey's 81 provinces,
split into a **solar** and a **wind** asset (162 assets in total), as an
investment option. The goal is to find the capacity mix that meets a target
share of national electricity demand at the **lowest cost** and the **lowest
production risk**, while also maximizing avoided CO₂ emissions.

Course: **YZV202E – Optimization for Data Science**, Istanbul Technical
University.

---

## What the project does

- Builds a **162-asset model** from 21 years (2004–2024) of NASA POWER weather
  data and real monthly demand data from the EPİAŞ Transparency Platform.
- Solves the problem with **three optimization methods** and compares them:
  - **SLSQP** – quadratic programming for the convex mean-variance problem.
  - **NSGA-II** – a multi-objective genetic algorithm with **three objectives**:
    cost, risk, and avoided CO₂.
  - **Simulated Annealing** – a single-objective baseline built from scratch.
- Matches production to a **real monthly demand profile**, which reveals a
  seasonal mismatch (large summer surplus, December shortfall).
- Runs a **robustness analysis**: Monte Carlo weather sampling, a chance
  constraint, a drought-year test, and a fossil-fuel price-shock scenario.
- Provides an **interactive dashboard** with sliders for budget, renewable
  share, and risk aversion.

---

## Key results

- Realistic capacity factors: solar **12.8–17.4%**, wind **1–28%**.
- The optimized mix meets the yearly target (~127.6 TWh) within the **$80B**
  budget, with much lower monthly volatility than any single-source strategy.
- Avoids about **62 million tonnes of CO₂ per year** (~13% of Turkey's
  energy-sector emissions).
- **Seasonal mismatch (key finding):** the portfolio over-produces in summer
  (147–156% of monthly demand) but only reaches **88.5%** in December. This is a
  clear, data-based argument for seasonal storage.

---

## Project structure

```
hybrid/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py              # all constants in one place (budget, costs, targets)
│   ├── main.py                # runs the full pipeline (8 steps)
│   ├── data_collection.py     # downloads NASA POWER weather data
│   ├── data_processing.py     # computes capacity factors + covariance matrix
│   ├── tuik_eurostat_prep.py  # cleans TUIK + Eurostat data
│   ├── epias_demand.py        # builds monthly demand profile from EPİAŞ
│   ├── opt_common.py          # shared data loader and helper functions
│   ├── optimization_slsqp.py  # Method 1: quadratic programming
│   ├── optimization_nsga2.py  # Method 2: NSGA-II (3 objectives)
│   ├── optimization_sa.py     # Method 3: simulated annealing
│   ├── robustness.py          # Monte Carlo, chance constraint, drought, price shock
│   ├── analysis.py            # strategic summary + EU benchmark
│   ├── make_figures.py        # generates the result figures
│   ├── check_chance_constraint.py  # verifies the reliability constraint
│   └── app.py                 # interactive Streamlit dashboard
├── data/
│   ├── tr_provinces_coords.csv     # input: 81 province coordinates
│   ├── raw/                        # input: TUIK, Eurostat, NASA cache
│   └── processed/                  # generated: capacity factors, demand profile, etc.
├── results/                        # generated: optimization outputs (CSV)
└── figures/                        # generated: result figures (PNG)
```

---

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Main libraries: `numpy`, `pandas`, `scipy`, `pymoo`, `matplotlib`, `requests`,
`streamlit`, `plotly`.

### EPİAŞ access (for demand data)

`epias_demand.py` downloads real monthly consumption from the EPİAŞ Transparency
Platform. You need a free account at
[seffaflik.epias.com.tr](https://seffaflik.epias.com.tr/). Put your username and
password in `epias_demand.py` before running it.

---

## How to run

**1. Build the monthly demand profile** (once, needs EPİAŞ login):

```bash
python src/epias_demand.py
```

**2. Run the full pipeline** (data → capacity factors → 3 optimizers → robustness
→ figures):

```bash
python src/main.py
```

Useful flags: `--force-nasa` to re-download weather data, `--force-processing`
to recompute capacity factors.

**3. Open the interactive dashboard:**

```bash
streamlit run src/app.py
```

The dashboard has five tabs: Efficient Frontier, Method Comparison, Robustness,
Capacity Factors, and a **Live Optimizer** where you can move sliders for budget,
renewable share, and risk aversion to re-solve the model in real time.

---

## Methodology notes

- **Solar capacity factor:** we divide irradiance by the standard test condition
  (1 kW/m²) and do **not** multiply by a separate panel efficiency, because the
  rated MW already includes it. Multiplying again would shrink the capacity
  factor about 5.6× and is physically wrong.
- **Wind capacity factor:** we estimate a per-point shear exponent from the 10 m
  and 50 m wind speeds and apply a cubic power curve.
- **Single config:** the budget, costs, and targets live in `config.py` only, so
  all three methods use identical settings and the comparison is fair.
- **Monthly coverage** (`ρ = 0.85`): each month must cover at least 85% of its
  demand. This floor is set by December, the hardest month.

---

## Future work

- **Seasonal storage:** add a storage capacity variable with a month-to-month
  energy balance to move the summer surplus into winter and close the December gap.
- **Transmission network:** model provinces as nodes and lines as edges with
  capacity and loss.
- **Cross-border trade:** allow import during shortfalls and export of surplus.
- **Hourly resolution:** replace monthly averages with hourly profiles to capture
  day–night and intra-day variability.

---

## Authors

- Mehmet Burak Koçoğlu
- Defne Yıldırım
- Ömer Faruk Irgaz

Istanbul Technical University, Dept. of Artificial Intelligence and Data
Engineering.

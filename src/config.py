"""
config.py
=========
Global configuration and constants.
Centralized settings to keep all modules synced.
"""

import os

# Paths
PROJECT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(PROJECT_DIR, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR   = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR   = os.path.join(PROJECT_DIR, "figures")

PROVINCES_FILE = os.path.join(DATA_DIR, "tr_provinces_coords.csv")

# NASA POWER settings
NASA_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_PARAMETERS = "ALLSKY_SFC_SW_DWN,WS10M,WS50M,T2M"
NASA_COMMUNITY  = "RE"
NASA_START_DATE = "20040101"
NASA_END_DATE   = "20241231"

# Solar settings
SOLAR_PERFORMANCE_RATIO = 0.80
SOLAR_TEMP_COEFF        = -0.004
SOLAR_T_REF             = 25.0
SOLAR_T_CELL_DELTA      = 25.0

# Wind settings
WIND_CUT_IN    = 3.0    # m/s
WIND_RATED     = 12.0   # m/s
WIND_CUT_OUT   = 25.0   # m/s
WIND_HUB_HEIGHT = 80.0  # m
WIND_REF_HEIGHT = 50.0  # m
WIND_ALPHA_MIN = 0.10
WIND_ALPHA_MAX = 0.40

# Economic parameters
COST_SOLAR_PER_MW = 550_000.0   # USD/MW
COST_WIND_PER_MW  = 800_000.0   # USD/MW
BUDGET_USD = 80e9               # Total budget
MIN_REGIONAL_MW = 500.0         # Minimum capacity per region

# Emissions
GRID_EMISSION_FACTOR = 0.42     # tCO2 / MWh

# Demand target
NATIONAL_DEMAND_TWH = 510.5
RENEWABLE_SHARE     = 0.25
HOURS_PER_YEAR      = 8760.0
MONTHLY_COVERAGE    = 0.85

def annual_demand_target_mwh():
    """Target energy in MWh/year."""
    return NATIONAL_DEMAND_TWH * 1e6 * RENEWABLE_SHARE

# Optimization scales
COST_SCALE = BUDGET_USD
VAR_SCALE_FALLBACK = 5e6

# Chance constraint / Monte Carlo
MC_N_SCENARIOS = 1000
CHANCE_D_MIN_FRACTION = 0.90
CHANCE_EPSILON        = 0.05
HOURS_PER_MONTH       = 730.0

FOSSIL_BASE_USD_PER_MWH    = 60.0
FOSSIL_SHOCK_USD_PER_MWH   = 90.0

# NSGA-II parameters
NSGA_POP_SIZE    = 200
NSGA_GENERATIONS = 500
NSGA_MIN_THRESHOLD_MW = 50.0

# SA parameters
SA_T0        = 5000.0
SA_T_MIN     = 1e-5
SA_ALPHA     = 0.9995
SA_STEP      = 200.0

SEED = 42
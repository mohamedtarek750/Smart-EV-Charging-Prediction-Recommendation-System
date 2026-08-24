"""Central configuration: paths, simulation window and shared constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- simulation
SIM_START = "2025-01-01"
SIM_END = "2025-12-31"
RANDOM_SEED = 42

# Egypt's weekend is Friday + Saturday (Mon=0 ... Sun=6 in pandas).
WEEKEND_DAYS = (4, 5)

# Fleet assumptions used by the session simulator.
BATTERY_KWH_CHOICES = (40.0, 50.0, 60.0, 75.0, 82.0, 100.0)
BATTERY_KWH_WEIGHTS = (0.18, 0.22, 0.24, 0.18, 0.12, 0.06)
CHARGE_EFFICIENCY = 0.92          # grid energy -> battery energy
TAPER_SOC = 0.80                  # above this SoC the charger tapers to 45 %
TAPER_FACTOR = 0.45
MAX_SESSION_HOURS = 8.0
MIN_SESSION_MINUTES = 8.0

# ---------------------------------------------------------------- grid / ops
GRID_SAFE_LOAD_KW = 4200.0        # city-wide contracted ceiling for EV charging
STATION_ALERT_UTILISATION = 0.85  # utilisation above this => congestion warning
WAIT_GREEN_MIN = 5.0              # <= 5 min  -> green
WAIT_AMBER_MIN = 15.0             # <= 15 min -> amber, above -> red

# ---------------------------------------------------------------- modelling
TARGETS = ("arrivals", "avg_wait_min", "energy_kwh")
TEST_MONTHS = 2                   # last N months held out for evaluation

# ---------------------------------------------------------------- tariff (EGP)
# Time-of-use price bands used by the recommender and the smart-charging optimiser.
TARIFF_EGP_PER_KWH = {"offpeak": 3.80, "shoulder": 5.00, "peak": 6.60}
PEAK_HOURS = tuple(range(17, 23))          # 17:00 - 22:59
OFFPEAK_HOURS = tuple(range(0, 7))         # 00:00 - 06:59
DC_FAST_PREMIUM = 1.35                     # surcharge factor for >= 50 kW chargers

# ---------------------------------------------------------------- driver model
AVG_CITY_SPEED_KMH = 34.0
PEAK_TRAFFIC_FACTOR = 1.35                 # slower driving during rush hour
CONSUMPTION_KWH_PER_KM = 0.17              # to check a station is actually reachable

"""Feature engineering for the station x hour panel.

Everything here is written to be *leak-free for a 1-hour-ahead forecast made at the
end of hour t-1*:

  * lags and rolling windows are always shifted by at least one hour;
  * the (station, weekday, hour) profile is an **expanding** mean, so row t only
    ever sees rows strictly before t;
  * weather and scheduled-event flags are used un-lagged on purpose - an operator
    genuinely has tomorrow's forecast and the events calendar in hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

RAW_PANEL = config.RAW_DIR / "station_hourly.csv.gz"
FEATURE_FILE = config.PROCESSED_DIR / "features.parquet"
FEATURE_FILE_CSV = config.PROCESSED_DIR / "features.csv.gz"

LAGS = (1, 2, 3, 24, 48, 168)
ROLLS = (3, 24, 168)

# Columns that describe the *state* of a station, safe to lag.
STATE_COLS = ("arrivals", "energy_kwh", "avg_wait_min", "utilisation", "avg_session_min")

CATEGORICAL = ["profile", "district"]


# --------------------------------------------------------------------------- #
def load_panel(path=RAW_PANEL) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df.timestamp
    df["hour"] = ts.dt.hour
    df["month"] = ts.dt.month
    df["dayofyear"] = ts.dt.dayofyear
    df["weekofyear"] = ts.dt.isocalendar().week.astype(int)
    # Cyclical encodings so midnight sits next to 23:00 for the model.
    df["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.dayofweek / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df.dayofyear / 365.25)
    df["is_peak_hour"] = df.hour.between(17, 20).astype(int)
    df["is_night"] = ((df.hour >= 23) | (df.hour <= 5)).astype(int)
    df["has_event"] = (df.event_multiplier > 1.01).astype(int)
    # Days since the start of the series: a proxy for EV-adoption growth.
    df["t_index"] = (ts - ts.min()).dt.total_seconds() / 3600.0
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("station_id", sort=False)
    for col in STATE_COLS:
        for lag in LAGS:
            df[f"{col}_lag{lag}"] = g[col].shift(lag)
        shifted = g[col].shift(1)
        for w in ROLLS:
            df[f"{col}_roll{w}"] = (shifted.groupby(df.station_id, sort=False)
                                    .rolling(w, min_periods=max(2, w // 4))
                                    .mean().reset_index(level=0, drop=True))
    # Short-term momentum: is this station warming up or cooling down?
    df["arrivals_delta_1_24"] = df.arrivals_lag1 - df.arrivals_lag24
    df["arrivals_ratio_3_24"] = df.arrivals_roll3 / df.arrivals_roll24.replace(0, np.nan)
    df["soc_arrival_roll24"] = (df.groupby("station_id", sort=False).avg_soc_arrival.shift(1)
                                .groupby(df.station_id, sort=False)
                                .rolling(24, min_periods=6).mean().reset_index(level=0, drop=True))
    return df


def add_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding (leak-free) mean demand for each station / weekday / hour cell."""
    key = ["station_id", "dayofweek", "hour"]
    for col, out in (("arrivals", "arrivals_profile"), ("energy_kwh", "energy_profile"),
                     ("avg_wait_min", "wait_profile")):
        g = df.groupby(key, sort=False)[col]
        df[out] = g.transform(lambda s: s.shift(1).expanding(min_periods=2).mean())
    df["arrivals_vs_profile"] = df.arrivals_lag168 - df.arrivals_profile
    return df


def add_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """City-wide conditions - a network-level view each station can react to."""
    city = (df.groupby("timestamp")
              .agg(city_arrivals=("arrivals", "sum"), city_load_kw=("avg_load_kw", "sum"))
              .sort_index())
    city["city_arrivals_lag1"] = city.city_arrivals.shift(1)
    city["city_arrivals_lag24"] = city.city_arrivals.shift(24)
    city["city_load_lag1"] = city.city_load_kw.shift(1)
    city["city_load_lag24"] = city.city_load_kw.shift(24)
    city["city_load_roll24"] = city.city_load_kw.shift(1).rolling(24, min_periods=6).mean()
    cols = ["city_arrivals_lag1", "city_arrivals_lag24", "city_load_lag1",
            "city_load_lag24", "city_load_roll24"]
    return df.merge(city[cols], left_on="timestamp", right_index=True, how="left")


def add_capacity_features(df: pd.DataFrame) -> pd.DataFrame:
    df["chargers_per_arrival_lag1"] = df.n_chargers / df.arrivals_lag1.replace(0, np.nan)
    df["expected_pressure"] = df.arrivals_roll3 / df.n_chargers
    df["headroom_lag1"] = 1.0 - df.utilisation_lag1
    df["power_per_charger"] = df.charger_kw
    return df


# --------------------------------------------------------------------------- #
def build(panel: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
    df = panel if panel is not None else load_panel()
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_profile_features(df)
    df = add_network_features(df)
    df = add_capacity_features(df)

    # The first week of every station has no 168-hour history - drop it.
    before = len(df)
    df = df.dropna(subset=["arrivals_lag168", "arrivals_roll168"]).reset_index(drop=True)
    print(f"dropped {before - len(df):,} warm-up rows -> {len(df):,} usable rows")

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    if save:
        try:
            df.to_parquet(FEATURE_FILE, index=False)
            print(f"saved -> {FEATURE_FILE}")
        except Exception as exc:                      # pyarrow not installed
            print(f"parquet unavailable ({exc.__class__.__name__}), writing csv.gz instead")
            df.to_csv(FEATURE_FILE_CSV, index=False)
            print(f"saved -> {FEATURE_FILE_CSV}")
    return df


def load_features() -> pd.DataFrame:
    if FEATURE_FILE.exists():
        df = pd.read_parquet(FEATURE_FILE)
    elif FEATURE_FILE_CSV.exists():
        df = pd.read_csv(FEATURE_FILE_CSV, parse_dates=["timestamp"])
    else:
        return build()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


# --------------------------------------------------------------------------- #
# Feature sets per model.  ``arrivals`` appears in the wait / energy sets because
# those two models are the *second stage* of a cascade: at serving time they are
# fed the demand model's prediction, not the truth (see src/models/train.py).
# --------------------------------------------------------------------------- #
BASE_FEATURES = [
    "hour", "dayofweek", "month", "is_weekend", "is_holiday", "is_peak_hour", "is_night",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "t_index",
    "temp_c", "is_rain", "has_event", "event_multiplier",
    "n_chargers", "charger_kw", "capacity_kw", "profile", "district",
    "city_arrivals_lag1", "city_arrivals_lag24", "city_load_lag1", "city_load_lag24",
    "city_load_roll24",
]

DEMAND_FEATURES = BASE_FEATURES + [
    *[f"arrivals_lag{l}" for l in LAGS],
    *[f"arrivals_roll{w}" for w in ROLLS],
    "arrivals_delta_1_24", "arrivals_ratio_3_24", "arrivals_profile", "arrivals_vs_profile",
    "utilisation_lag1", "utilisation_lag24", "utilisation_roll24",
    "energy_kwh_lag1", "energy_kwh_lag24", "avg_wait_min_lag1", "avg_wait_min_lag24",
    "soc_arrival_roll24", "expected_pressure", "headroom_lag1",
]

WAIT_FEATURES = BASE_FEATURES + [
    "arrivals",                       # 1st-stage prediction at serving time
    "arrivals_lag1", "arrivals_lag24", "arrivals_roll3", "arrivals_roll24", "arrivals_profile",
    "avg_wait_min_lag1", "avg_wait_min_lag2", "avg_wait_min_lag24", "avg_wait_min_roll3",
    "avg_wait_min_roll24", "wait_profile",
    "utilisation_lag1", "utilisation_lag24", "utilisation_roll3", "utilisation_roll24",
    "avg_session_min_lag1", "avg_session_min_roll24", "soc_arrival_roll24",
    "expected_pressure", "headroom_lag1", "chargers_per_arrival_lag1",
]

ENERGY_FEATURES = BASE_FEATURES + [
    "arrivals",                       # 1st-stage prediction at serving time
    "arrivals_lag1", "arrivals_lag24", "arrivals_roll3", "arrivals_roll24", "arrivals_profile",
    *[f"energy_kwh_lag{l}" for l in LAGS],
    *[f"energy_kwh_roll{w}" for w in ROLLS],
    "energy_profile", "utilisation_lag1", "utilisation_roll24",
    "avg_session_min_lag1", "avg_session_min_roll24", "soc_arrival_roll24",
]

FEATURE_SETS = {
    "arrivals": DEMAND_FEATURES,
    "avg_wait_min": WAIT_FEATURES,
    "energy_kwh": ENERGY_FEATURES,
}


if __name__ == "__main__":
    out = build()
    print(f"\ncolumns: {out.shape[1]}   rows: {len(out):,}")
    for target, feats in FEATURE_SETS.items():
        missing = [f for f in feats if f not in out.columns]
        print(f"{target:14s} -> {len(feats)} features" + (f"  MISSING: {missing}" if missing else ""))

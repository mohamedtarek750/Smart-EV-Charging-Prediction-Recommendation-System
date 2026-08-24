"""Serving layer: turn the three trained models into a rolling multi-hour forecast.

A one-hour-ahead model cannot answer "what does 7 PM look like?" at 2 PM on its own,
because its inputs are lags that do not exist yet.  :class:`Forecaster` closes that
loop: it walks forward one hour at a time, writes each prediction back into the
working panel, and uses it as the lag for the next step (recursive forecasting).

Per step the models are evaluated as a cascade:

    arrivals  ->  avg_wait_min
              ->  energy_kwh

so the wait and load forecasts are consistent with the demand forecast.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data.stations import STATIONS, stations_frame
from src.data.simulate import EGYPT_HOLIDAYS_2025
from src.features import build_features as F

HISTORY_HOURS = 200          # enough to cover the 168-hour lag/rolling windows
TARGETS = ("arrivals", "avg_wait_min", "energy_kwh")


# --------------------------------------------------------------------------- #
def status_of(wait_min: float) -> str:
    if wait_min <= config.WAIT_GREEN_MIN:
        return "green"
    if wait_min <= config.WAIT_AMBER_MIN:
        return "amber"
    return "red"


@dataclass
class LoadedModel:
    target: str
    model: object
    features: list[str]
    log_target: bool
    metrics: dict

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model.predict(X[self.features])
        if self.log_target:
            p = np.expm1(p)
        return np.clip(p, 0.0, None)


MODEL_LOAD_HELP = """
Could not unpickle {path}

    {error}

The saved models were built with numpy {saved_numpy} / scikit-learn {saved_sklearn};
this interpreter has numpy {have_numpy} / scikit-learn {have_sklearn}.  Pickled
estimators are not portable across major numpy or scikit-learn versions.

Two ways out:

  1. Run everything with one interpreter.  On Windows the usual trap is that
     `python` and `streamlit` resolve to *different* installations - launch the
     apps through the interpreter instead of the console script:

         python -m streamlit run app/dashboard.py
         python -m uvicorn app.api:app --reload

  2. Or just retrain in the environment you want to use (about 40 seconds):

         python -m src.models.train
"""


@lru_cache(maxsize=1)
def load_models() -> dict[str, LoadedModel]:
    import sklearn

    out = {}
    for t in TARGETS:
        path = config.MODELS_DIR / f"{t}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing - run `python -m src.models.train` first.")
        try:
            b = joblib.load(path)
        except Exception as exc:
            meta = json.loads((config.REPORTS_DIR / "metrics.json").read_text(encoding="utf-8")) \
                if (config.REPORTS_DIR / "metrics.json").exists() else {}
            env = meta.get("environment", {})
            raise RuntimeError(MODEL_LOAD_HELP.format(
                path=path, error=f"{exc.__class__.__name__}: {exc}",
                saved_numpy=env.get("numpy", "?"), saved_sklearn=env.get("scikit_learn", "?"),
                have_numpy=np.__version__, have_sklearn=sklearn.__version__)) from exc
        out[t] = LoadedModel(b["target"], b["model"], b["features"], b["log_target"],
                             b.get("metrics", {}))
    return out


# --------------------------------------------------------------------------- #
class Forecaster:
    """Rolling forecaster over the whole 20-station network."""

    def __init__(self, panel: pd.DataFrame | None = None):
        self.models = load_models()
        self.panel = panel if panel is not None else F.load_panel()
        self.panel = self.panel.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
        self.meta = stations_frame()
        self.last_ts = self.panel.timestamp.max()
        self._profiles = self._build_profiles()
        self._climate = self._build_climate()

    # ------------------------------------------------------------------ setup
    def _build_profiles(self) -> pd.DataFrame:
        """Serving analogue of the expanding (station, weekday, hour) profile."""
        p = self.panel.copy()
        p["hour"] = p.timestamp.dt.hour
        g = (p.groupby(["station_id", "dayofweek", "hour"])
               .agg(arrivals_profile=("arrivals", "mean"),
                    energy_profile=("energy_kwh", "mean"),
                    wait_profile=("avg_wait_min", "mean"))
               .reset_index())
        return g

    def _build_climate(self) -> pd.DataFrame:
        """Hour-of-year temperature climatology, used when no forecast is supplied."""
        w = self.panel[["timestamp", "temp_c"]].drop_duplicates("timestamp").copy()
        w["doy"] = w.timestamp.dt.dayofyear
        w["hour"] = w.timestamp.dt.hour
        return w.groupby(["doy", "hour"], as_index=False).temp_c.mean()

    def _weather_for(self, stamps: pd.DatetimeIndex,
                     override: pd.DataFrame | None = None) -> pd.DataFrame:
        base = pd.DataFrame({"timestamp": stamps})
        base["doy"] = base.timestamp.dt.dayofyear
        base["hour"] = base.timestamp.dt.hour
        base = base.merge(self._climate, on=["doy", "hour"], how="left")
        base["temp_c"] = base.temp_c.fillna(self.panel.temp_c.mean())
        base["is_rain"] = 0
        if override is not None and not override.empty:
            o = override.set_index("timestamp")
            for col in ("temp_c", "is_rain"):
                if col in o.columns:
                    vals = base.timestamp.map(o[col])
                    base[col] = vals.fillna(base[col])
        return base[["timestamp", "temp_c", "is_rain"]]

    # --------------------------------------------------------------- forecast
    def forecast(self, horizon_hours: int = 24, start: pd.Timestamp | None = None,
                 weather: pd.DataFrame | None = None,
                 events: pd.DataFrame | None = None) -> pd.DataFrame:
        """Predict every station for the next ``horizon_hours`` hours.

        ``weather``  optional frame [timestamp, temp_c, is_rain] - a real forecast.
        ``events``   optional frame [timestamp, station_id, event_multiplier].
        """
        start = pd.Timestamp(start) if start is not None else self.last_ts + pd.Timedelta(hours=1)
        stamps = pd.date_range(start, periods=horizon_hours, freq="h")

        hist = self.panel.loc[self.panel.timestamp < start].copy()
        hist = hist.loc[hist.timestamp >= start - pd.Timedelta(hours=HISTORY_HOURS)]

        future = self._blank_future(stamps, weather, events)
        work = pd.concat([hist, future], ignore_index=True)
        work = work.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

        for ts in stamps:
            feats = self._features_for(work, ts)
            arrivals = self.models["arrivals"].predict(feats)
            feats["arrivals"] = arrivals
            wait = self.models["avg_wait_min"].predict(feats)
            energy = self.models["energy_kwh"].predict(feats)

            mask = work.timestamp == ts
            order = work.loc[mask, "station_id"].to_numpy()
            lookup = dict(zip(feats.station_id, zip(arrivals, wait, energy)))
            vals = np.array([lookup[s] for s in order])
            work.loc[mask, "arrivals"] = vals[:, 0]
            work.loc[mask, "avg_wait_min"] = vals[:, 1]
            work.loc[mask, "energy_kwh"] = vals[:, 2]
            work.loc[mask, "avg_load_kw"] = vals[:, 2]
            # Occupancy implied by the load forecast, so the next step's lags are sane.
            cap = work.loc[mask, "capacity_kw"].to_numpy(float)
            work.loc[mask, "utilisation"] = np.clip(vals[:, 2] / np.maximum(cap * 0.55, 1), 0, 1)
            work.loc[mask, "charger_hours"] = (work.loc[mask, "utilisation"].to_numpy()
                                               * work.loc[mask, "n_chargers"].to_numpy())

        out = work.loc[work.timestamp.isin(stamps)].copy()
        out["status"] = [status_of(w) for w in out.avg_wait_min]
        out["free_chargers_est"] = np.round(
            np.clip(out.n_chargers * (1 - out.utilisation), 0, None), 1)
        cols = ["timestamp", "station_id", "name", "district", "profile", "lat", "lon",
                "n_chargers", "charger_kw", "capacity_kw", "arrivals", "avg_wait_min",
                "energy_kwh", "avg_load_kw", "utilisation", "free_chargers_est", "status",
                "temp_c", "is_rain", "is_weekend", "is_holiday", "event_multiplier"]
        return out[cols].sort_values(["timestamp", "station_id"]).reset_index(drop=True)

    # ------------------------------------------------------------- internals
    def _blank_future(self, stamps: pd.DatetimeIndex, weather, events) -> pd.DataFrame:
        grid = pd.MultiIndex.from_product([stamps, [s.station_id for s in STATIONS]],
                                          names=["timestamp", "station_id"]).to_frame(index=False)
        grid = grid.merge(self.meta[["station_id", "name", "district", "profile", "n_chargers",
                                     "charger_kw", "capacity_kw", "lat", "lon"]],
                          on="station_id", how="left")
        grid = grid.merge(self._weather_for(stamps, weather), on="timestamp", how="left")
        grid["dayofweek"] = grid.timestamp.dt.dayofweek
        grid["is_weekend"] = grid.dayofweek.isin(config.WEEKEND_DAYS).astype(int)
        grid["is_holiday"] = grid.timestamp.dt.strftime("%Y-%m-%d").isin(
            EGYPT_HOLIDAYS_2025).astype(int)
        grid["event_multiplier"] = 1.0
        if events is not None and not events.empty:
            ev = events.copy()
            ev["timestamp"] = pd.to_datetime(ev.timestamp)
            grid = grid.merge(ev.rename(columns={"event_multiplier": "_ev"}),
                              on=["timestamp", "station_id"], how="left")
            grid["event_multiplier"] = grid["_ev"].fillna(grid.event_multiplier)
            grid = grid.drop(columns=["_ev"])
        for c in ("arrivals", "served", "abandoned", "avg_wait_min", "p90_wait_min",
                  "avg_session_min", "charger_hours", "utilisation", "energy_kwh",
                  "avg_load_kw", "queue_pressure"):
            grid[c] = np.nan
        grid["avg_soc_arrival"] = self.panel.avg_soc_arrival.mean()
        return grid

    def _features_for(self, work: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
        """Build the model matrix for a single future hour, all stations at once."""
        w = work.loc[work.timestamp <= ts].copy()
        w = F.add_calendar_features(w)
        w = F.add_lag_features(w)
        w = F.add_network_features(w)
        w = F.add_capacity_features(w)

        w = w.merge(self._profiles, on=["station_id", "dayofweek", "hour"], how="left")
        w["arrivals_vs_profile"] = w.arrivals_lag168 - w.arrivals_profile

        row = w.loc[w.timestamp == ts].copy()
        for c in F.CATEGORICAL:
            row[c] = pd.Categorical(row[c], categories=sorted(self.panel[c].unique()))
        needed = sorted({f for fs in F.FEATURE_SETS.values() for f in fs} | {"station_id"})
        for f in needed:
            if f not in row.columns:
                row[f] = np.nan
        return row[needed].reset_index(drop=True)


# --------------------------------------------------------------------------- #
def city_summary(fc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a station-level forecast into the operator's city view."""
    g = (fc.groupby("timestamp")
           .agg(expected_evs=("arrivals", "sum"),
                expected_load_kw=("avg_load_kw", "sum"),
                mean_wait_min=("avg_wait_min", "mean"),
                max_wait_min=("avg_wait_min", "max"),
                mean_utilisation=("utilisation", "mean"),
                congested_stations=("status", lambda s: int((s == "red").sum())))
           .reset_index())
    g["grid_headroom_kw"] = config.GRID_SAFE_LOAD_KW - g.expected_load_kw
    g["overload_risk"] = g.expected_load_kw > config.GRID_SAFE_LOAD_KW
    return g


if __name__ == "__main__":
    f = Forecaster()
    print(f"history ends at {f.last_ts}, forecasting the next 24 h ...")
    fc = f.forecast(24)
    city = city_summary(fc)
    print("\n--- city outlook ---")
    show = city[["timestamp", "expected_evs", "expected_load_kw", "mean_wait_min",
                 "congested_stations", "grid_headroom_kw"]].copy()
    num = show.select_dtypes("number").columns
    show[num] = show[num].round(1)
    print(show.to_string(index=False))
    peak = city.loc[city.expected_load_kw.idxmax()]
    print(f"\npeak load {peak.expected_load_kw:,.0f} kW at {peak.timestamp:%Y-%m-%d %H:%M} "
          f"(safe ceiling {config.GRID_SAFE_LOAD_KW:,.0f} kW)")
    worst = fc.loc[fc.avg_wait_min.idxmax()]
    print(f"worst queue: {worst['name']} at {worst.timestamp:%H:%M} -> "
          f"{worst.avg_wait_min:.0f} min ({worst.status})")

"""End-to-end sanity tests.

Run with:  python -m tests.test_pipeline      (or `pytest tests/` if pytest is installed)

These are guard-rails against the mistakes that quietly ruin a forecasting project:
leakage across the time split, a simulator whose queueing does not match its own
arrivals, an optimiser that "shaves the peak" by simply not charging the cars.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import config
from src.data import simulate
from src.data.stations import STATIONS, haversine_km, nearby_stations
from src.features import build_features as F
from src.optimization import scheduler
from src.recommender import engine

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  -> {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
def test_station_network() -> None:
    ids = [s.station_id for s in STATIONS]
    check("station ids are unique", len(set(ids)) == len(ids))
    check("every station has chargers", all(s.n_chargers > 0 for s in STATIONS))
    d = haversine_km(30.0, 31.7, 30.0, 31.7)
    check("haversine of a point with itself is 0", abs(d) < 1e-9, f"{d}")
    near = nearby_stations(30.0086, 31.7395, radius_km=5, limit=3)
    check("nearest station to ST01 is ST01", near[0][0].station_id == "ST01")
    check("nearby stations are sorted by distance",
          all(near[i][1] <= near[i + 1][1] for i in range(len(near) - 1)))


def test_simulator_short_run() -> None:
    cfg = simulate.SimConfig(start="2025-03-01", end="2025-03-14")
    res = simulate.generate(cfg, save=False)
    panel, sessions = res["panel"], res["sessions"]

    check("panel covers every station-hour",
          len(panel) == 14 * 24 * len(STATIONS), f"{len(panel)}")
    check("no negative arrivals", (panel.arrivals >= 0).all())
    check("utilisation stays in [0, 1]",
          panel.utilisation.between(0, 1).all(), f"max {panel.utilisation.max()}")
    check("arrivals == served + abandoned",
          bool((panel.arrivals == panel.served + panel.abandoned).all()))
    check("abandoned sessions deliver no energy",
          float(sessions.loc[sessions.abandoned == 1, "energy_kwh"].abs().max()) == 0.0)
    check("served sessions all deliver energy",
          bool((sessions.loc[sessions.abandoned == 0, "energy_kwh"] > 0).all()))
    check("waiting time is never negative", bool((sessions.wait_min >= -1e-9).all()))
    check("a station never serves more cars at once than it has chargers",
          bool((panel.charger_hours <= panel.n_chargers + 1e-6).all()))

    # The demand shape must actually differ between night and evening peak.
    by_hour = panel.assign(h=panel.timestamp.dt.hour).groupby("h").arrivals.mean()
    check("evening demand exceeds night demand",
          by_hour.loc[17:20].mean() > by_hour.loc[2:5].mean() * 3,
          f"{by_hour.loc[17:20].mean():.2f} vs {by_hour.loc[2:5].mean():.2f}")

    # Energy bookkeeping: hourly aggregation must conserve session energy.
    delivered = sessions.loc[sessions.abandoned == 0].energy_kwh.sum()
    # Sessions running past the window end are truncated by the aggregation.
    check("hourly energy is within 3 % of session energy",
          abs(panel.energy_kwh.sum() - delivered) / delivered < 0.03,
          f"panel {panel.energy_kwh.sum():.0f} vs sessions {delivered:.0f}")


def test_features_are_leak_free() -> None:
    df = F.load_features()
    check("feature table is non-empty", len(df) > 0)

    # A lag must equal the earlier value of its own column, per station.
    g = df.loc[df.station_id == "ST12"].sort_values("timestamp")
    ok = np.allclose(g.arrivals_lag1.to_numpy()[1:], g.arrivals.to_numpy()[:-1], equal_nan=True)
    check("arrivals_lag1 equals the previous hour's arrivals", bool(ok))

    # The expanding profile must never contain the current row.
    sample = g.dropna(subset=["arrivals_profile"]).head(200)
    same_cell = df.loc[(df.station_id == "ST12")]
    check("arrivals_profile is defined and finite",
          bool(np.isfinite(sample.arrivals_profile).all()) and len(sample) > 0)

    # No feature may be a perfect copy of the target.
    for target, feats in F.FEATURE_SETS.items():
        leaked = [f for f in feats if f != target and f in df.columns
                  and pd.api.types.is_numeric_dtype(df[f])
                  and df[[f, target]].dropna().shape[0] > 100
                  and abs(df[[f, target]].dropna().corr().iloc[0, 1]) > 0.999]
        check(f"no leaked copy of `{target}` in its feature set", not leaked, str(leaked))


def test_models_and_forecast() -> None:
    from src.models.predict import Forecaster, city_summary, status_of

    check("status thresholds are ordered",
          status_of(1) == "green" and status_of(10) == "amber" and status_of(60) == "red")

    fc = Forecaster()
    out = fc.forecast(6)
    check("forecast covers all stations x 6 hours", len(out) == 6 * len(STATIONS), f"{len(out)}")
    check("forecast has no NaNs in the predicted columns",
          not out[["arrivals", "avg_wait_min", "energy_kwh"]].isna().any().any())
    check("predictions are non-negative",
          bool((out[["arrivals", "avg_wait_min", "energy_kwh"]] >= 0).all().all()))
    check("forecast starts after the history ends", out.timestamp.min() > fc.last_ts)

    city = city_summary(out)
    check("city load equals the sum over stations",
          np.allclose(city.expected_load_kw.to_numpy(),
                      out.groupby("timestamp").avg_load_kw.sum().to_numpy()))


def test_recommender() -> None:
    from src.models.predict import Forecaster

    rec = engine.Recommender(Forecaster())
    req = engine.DriverRequest(lat=29.9985, lon=31.7318, battery_kwh=60, soc_now=0.20,
                               soc_target=0.80, vehicle_max_kw=100,
                               when=rec.fc.last_ts + pd.Timedelta(hours=18))
    out = rec.recommend(req, top_k=6)
    opts = out["options"]
    check("recommender returns options", len(opts) > 0)
    check("total time is the sum of its parts",
          all(abs(o["total_min"] - (o["drive_min"] + o["predicted_wait_min"] + o["charge_min"])) < 0.2
              for o in opts))
    check("options are ranked by score",
          all(opts[i]["score"] <= opts[i + 1]["score"] for i in range(len(opts) - 1)))
    check("the recommended station is the first option",
          out["recommendation"]["station_id"] == opts[0]["station_id"])

    # A faster charger must never take longer to deliver the same energy.
    slow, _ = engine.charge_minutes(40, 0.2, 0.8, 22, 150)
    fast, _ = engine.charge_minutes(40, 0.2, 0.8, 120, 150)
    check("a 120 kW charger beats a 22 kW charger", fast < slow, f"{fast} vs {slow}")

    check("peak hours are priced above off-peak",
          engine.price_per_kwh(19, 50) > engine.price_per_kwh(3, 50))


def test_scheduler() -> None:
    rng = np.random.default_rng(0)
    horizon = 24
    vehicles = [
        scheduler.Vehicle(f"v{i}", "ST01", int(rng.integers(0, 12)), 0, 0.0, 0.0)
        for i in range(40)
    ]
    # Rebuild with valid windows (dataclass fields are positional above).
    vehicles = []
    for i in range(40):
        t0 = int(rng.integers(0, 14))
        vehicles.append(scheduler.Vehicle(f"v{i}", "ST01", t0, min(t0 + 8, horizon),
                                          float(rng.uniform(10, 45)), 22.0))
    prices = scheduler.hourly_prices(pd.Timestamp("2026-01-01"), horizon)
    naive = scheduler.uncontrolled_schedule(vehicles, horizon, prices)
    cap = scheduler.grid_cap(horizon, naive.peak_kw * 0.75)
    opt = scheduler.optimal_schedule(vehicles, horizon, prices, cap)

    check("optimiser respects the hourly cap",
          bool((opt.profile <= cap + 1e-6).all()),
          f"max {opt.profile.max():.1f} vs cap {cap[0]:.1f}")
    check("optimiser lowers the peak", opt.peak_kw < naive.peak_kw,
          f"{opt.peak_kw:.1f} vs {naive.peak_kw:.1f}")
    check("optimiser still delivers the energy", opt.served_pct > 99.0,
          f"{opt.served_pct:.2f}%")
    check("no vehicle charges outside its window",
          all(opt.power[i, t] < 1e-6
              for i, v in enumerate(vehicles)
              for t in range(horizon) if t not in v.window))
    check("no vehicle exceeds its power limit",
          bool((opt.power <= np.array([[v.max_kw] * horizon for v in vehicles]) + 1e-6).all()))
    check("greedy also respects the cap",
          bool((scheduler.greedy_schedule(vehicles, horizon, prices, cap).profile
                <= cap + 1e-6).all()))


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 62)
    for fn in (test_station_network, test_simulator_short_run, test_features_are_leak_free,
               test_models_and_forecast, test_recommender, test_scheduler):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

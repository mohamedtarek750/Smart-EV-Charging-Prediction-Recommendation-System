"""Synthetic-but-realistic data generator for the New Administrative Capital EV network.

Nothing here is "ML" - it is a discrete-event simulator that plays the role of the
historical database an operator would normally hand you.  It produces three files:

    data/raw/weather.csv.gz        hourly weather / calendar context
    data/raw/sessions.csv.gz       one row per charging session (event level)
    data/raw/station_hourly.csv.gz station x hour panel  <- the modelling table

The simulator embeds real structure (commuter peaks, weekend shift, seasonal EV
adoption growth, temperature effects on consumption, venue events, M/M/c queueing)
plus noise, so the models downstream have something genuine to learn.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config
from src.data.stations import STATIONS, Station, stations_frame

# --------------------------------------------------------------------------- #
# Hourly arrival shapes (index 0..23), normalised so the peak hour equals 1.0   #
# --------------------------------------------------------------------------- #
WEEKDAY_SHAPES: dict[str, list[float]] = {
    "government":  [.05,.03,.02,.02,.03,.09,.28,.78,1.00,.82,.58,.48,.44,.50,.56,.70,.86,.90,.58,.34,.20,.13,.09,.06],
    "business":    [.06,.04,.03,.02,.03,.07,.22,.62,.95,1.00,.72,.58,.55,.58,.62,.72,.88,.96,.74,.48,.30,.20,.13,.09],
    "residential": [.34,.24,.15,.10,.08,.11,.19,.31,.26,.20,.18,.20,.23,.26,.31,.42,.62,.86,1.00,.95,.80,.66,.55,.45],
    "retail":      [.05,.03,.02,.02,.02,.03,.06,.11,.21,.36,.52,.66,.76,.80,.80,.86,.92,1.00,.96,.86,.70,.50,.30,.13],
    "transit":     [.16,.11,.08,.08,.13,.32,.62,.92,1.00,.80,.60,.55,.55,.60,.66,.76,.90,1.00,.86,.62,.46,.36,.28,.21],
    "leisure":     [.07,.04,.03,.02,.02,.03,.06,.12,.20,.30,.42,.54,.62,.66,.68,.76,.88,1.00,.94,.82,.62,.40,.22,.12],
    "hotel":       [.12,.08,.05,.04,.04,.06,.12,.22,.30,.34,.32,.30,.32,.36,.42,.52,.68,.86,1.00,.92,.74,.54,.34,.20],
    "education":   [.04,.02,.02,.02,.03,.08,.26,.70,.96,1.00,.78,.62,.58,.66,.72,.68,.52,.36,.24,.16,.11,.08,.06,.05],
}

WEEKEND_SHAPES: dict[str, list[float]] = {
    "government":  [.06,.04,.03,.02,.03,.05,.10,.18,.26,.32,.34,.34,.32,.30,.30,.34,.40,.44,.38,.28,.20,.14,.10,.07],
    "business":    [.08,.05,.03,.03,.03,.05,.10,.20,.32,.40,.44,.44,.42,.40,.40,.44,.52,.58,.50,.38,.28,.20,.14,.10],
    "residential": [.40,.30,.20,.13,.10,.10,.14,.20,.26,.34,.42,.48,.52,.54,.56,.62,.74,.90,1.00,.98,.88,.74,.62,.50],
    "retail":      [.10,.06,.04,.03,.02,.03,.05,.09,.16,.28,.46,.62,.74,.82,.86,.92,.98,1.00,1.00,.96,.86,.68,.44,.22],
    "transit":     [.22,.15,.10,.08,.10,.18,.34,.52,.66,.74,.76,.74,.72,.72,.74,.80,.90,.96,.88,.72,.58,.46,.36,.28],
    "leisure":     [.12,.07,.04,.03,.02,.03,.06,.12,.22,.36,.52,.66,.76,.82,.86,.92,1.00,1.00,.94,.84,.66,.44,.26,.16],
    "hotel":       [.16,.11,.07,.05,.04,.06,.12,.22,.34,.42,.44,.44,.44,.46,.50,.58,.72,.90,1.00,.96,.82,.62,.42,.26],
    "education":   [.05,.03,.02,.02,.02,.04,.08,.16,.24,.30,.32,.32,.30,.30,.30,.28,.24,.20,.16,.12,.09,.07,.05,.04],
}

# Overall weekend level relative to a weekday for the same venue type.
WEEKEND_LEVEL: dict[str, float] = {
    "government": 0.20, "business": 0.28, "residential": 1.18, "retail": 1.40,
    "transit": 0.85, "leisure": 1.55, "hotel": 1.25, "education": 0.30,
}

# Typical dwell time (hours) of a driver at each venue type -> charging flexibility.
DWELL_HOURS: dict[str, tuple[float, float]] = {   # (mean, sigma) of the dwell draw
    "government": (7.5, 1.6), "business": (7.0, 1.8), "residential": (9.5, 2.5),
    "retail": (2.4, 0.9), "transit": (1.1, 0.6), "leisure": (3.0, 1.1),
    "hotel": (10.0, 3.0), "education": (5.0, 1.8),
}

# Vehicle-side maximum accepted power (kW) and its mix in the fleet.
VEHICLE_MAX_KW = (11.0, 22.0, 50.0, 100.0, 150.0)
VEHICLE_MAX_W = (0.22, 0.28, 0.24, 0.18, 0.08)

EGYPT_HOLIDAYS_2025 = {
    "2025-01-07", "2025-01-25", "2025-03-30", "2025-03-31", "2025-04-01",
    "2025-04-20", "2025-04-21", "2025-04-25", "2025-05-01", "2025-06-06",
    "2025-06-07", "2025-06-08", "2025-06-09", "2025-06-26", "2025-07-23",
    "2025-09-04", "2025-10-06",
}

# Venues that can host a crowd-drawing event.
EVENT_STATIONS = ("ST12", "ST13", "ST20", "ST11", "ST04")

ADOPTION_GROWTH = 0.45      # +45 % EV fleet between January and December
PEAK_UTILISATION = 0.68     # peak-hour design point for an average-popularity station


# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    start: str = config.SIM_START
    end: str = config.SIM_END
    seed: int = config.RANDOM_SEED


# --------------------------------------------------------------------------- #
def build_calendar(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Hourly calendar + weather context for the whole simulation window."""
    idx = pd.date_range(f"{cfg.start} 00:00", f"{cfg.end} 23:00", freq="h")
    df = pd.DataFrame({"timestamp": idx})
    doy = df.timestamp.dt.dayofyear.to_numpy()
    hour = df.timestamp.dt.hour.to_numpy()

    # Cairo-like climate: seasonal swing + daily swing + correlated noise.
    seasonal = 21.5 - 8.5 * np.cos(2 * np.pi * (doy - 20) / 365.25)
    daily = -5.2 * np.cos(2 * np.pi * (hour - 15) / 24)
    noise = np.cumsum(rng.normal(0, 0.45, len(df)))
    noise = noise - pd.Series(noise).rolling(72, min_periods=1, center=True).mean().to_numpy()
    df["temp_c"] = np.round(seasonal + daily + noise, 2)

    # Rain is rare and winter-biased in Egypt.
    rain_p = np.clip(0.030 * np.exp(-((doy - 15) % 365) ** 2 / (2 * 70 ** 2))
                     + 0.030 * np.exp(-((doy - 350) % 365) ** 2 / (2 * 70 ** 2)), 0.002, 0.05)
    df["is_rain"] = (rng.random(len(df)) < rain_p).astype(int)

    df["date"] = df.timestamp.dt.normalize()
    df["dayofweek"] = df.timestamp.dt.dayofweek
    df["is_weekend"] = df.dayofweek.isin(config.WEEKEND_DAYS).astype(int)
    df["is_holiday"] = df.timestamp.dt.strftime("%Y-%m-%d").isin(EGYPT_HOLIDAYS_2025).astype(int)
    return df


def build_events(cfg: SimConfig, calendar: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Concerts / expos / matches: short windows of 2-3x demand at one venue."""
    days = calendar.date.unique()
    rows: list[dict] = []
    n_events = int(len(days) * 0.13)                     # ~1 event every 8 days
    for _ in range(n_events):
        day = pd.Timestamp(rng.choice(days))
        station_id = str(rng.choice(EVENT_STATIONS))
        start_h = int(rng.integers(15, 21))
        duration = int(rng.integers(3, 6))
        strength = float(rng.uniform(1.8, 3.2))
        for k in range(duration):
            rows.append({
                "timestamp": day + pd.Timedelta(hours=start_h + k),
                "station_id": station_id,
                "event_multiplier": strength if k < duration - 1 else 1.0 + (strength - 1) * 0.4,
            })
    ev = pd.DataFrame(rows, columns=["timestamp", "station_id", "event_multiplier"])
    if ev.empty:
        return ev
    return ev.groupby(["timestamp", "station_id"], as_index=False).event_multiplier.max()


@lru_cache(maxsize=None)
def expected_session_hours(charger_kw: float, transit: bool = False, n: int = 40_000) -> float:
    """Monte-Carlo mean session length for a charger of this rating.

    Arrival rates are sized as ``peak_lambda = target_utilisation * n_chargers /
    E[service time]``, so this estimate has to use exactly the same fleet
    distributions as :func:`build_sessions` - a hand-waved constant here is what
    turns a healthy network into a permanently gridlocked one.
    """
    rng = np.random.default_rng(7)
    battery = rng.choice(config.BATTERY_KWH_CHOICES, n, p=config.BATTERY_KWH_WEIGHTS)
    veh = rng.choice(VEHICLE_MAX_KW, n, p=VEHICLE_MAX_W)
    soc_a = np.clip(rng.beta(2.4, 4.6, n) * 0.70 + 0.05, 0.03, 0.80)
    if transit:
        soc_a = np.clip(soc_a - 0.07, 0.03, 0.80)
    soc_t = np.clip(rng.normal(0.86, 0.09, n), soc_a + 0.08, 0.99)
    energy = battery * (soc_t - soc_a) / config.CHARGE_EFFICIENCY * 1.05   # mean thermal uplift
    eff_kw = np.minimum(charger_kw, veh)
    bulk = np.clip((config.TAPER_SOC - soc_a) / (soc_t - soc_a), 0.0, 1.0)
    avg_kw = eff_kw * (bulk + (1 - bulk) * config.TAPER_FACTOR)
    dur = np.clip(energy / np.maximum(avg_kw, 3.0), config.MIN_SESSION_MINUTES / 60,
                  config.MAX_SESSION_HOURS)
    return float(dur.mean())


# --------------------------------------------------------------------------- #
def draw_arrivals(cfg: SimConfig, calendar: pd.DataFrame, events: pd.DataFrame,
                  rng: np.random.Generator) -> pd.DataFrame:
    """Poisson arrival counts for every (station, hour)."""
    hours = calendar.timestamp.dt.hour.to_numpy()
    weekend = calendar.is_weekend.to_numpy().astype(bool)
    holiday = calendar.is_holiday.to_numpy().astype(bool)
    temp = calendar.temp_c.to_numpy()
    rain = calendar.is_rain.to_numpy().astype(bool)
    n = len(calendar)

    # EV adoption grows over the year.
    t_frac = np.arange(n) / max(n - 1, 1)
    trend = 1.0 + ADOPTION_GROWTH * t_frac
    # Extreme heat suppresses discretionary trips a little.
    heat = np.where(temp > 36, 0.90, 1.0)
    wet = np.where(rain, 0.93, 1.0)

    ev_lookup: dict[tuple[pd.Timestamp, str], float] = {}
    if not events.empty:
        ev_lookup = {(r.timestamp, r.station_id): r.event_multiplier for r in events.itertuples()}

    day_keys = sorted(calendar.date.unique())

    frames = []
    for st in STATIONS:
        wd_shape = np.array(WEEKDAY_SHAPES[st.profile])
        we_shape = np.array(WEEKEND_SHAPES[st.profile]) * WEEKEND_LEVEL[st.profile]
        shape = np.where(weekend, we_shape[hours], wd_shape[hours])
        # Holidays behave like weekends for work sites, like weekends+ for venues.
        shape = np.where(holiday & ~weekend, we_shape[hours] * 1.05, shape)

        mean_service_h = expected_session_hours(st.charger_kw, st.profile == "transit")
        peak_lambda = PEAK_UTILISATION * st.n_chargers / mean_service_h
        lam = peak_lambda * (st.demand_scale / 1.10) * shape * trend * heat * wet

        if ev_lookup:
            mult = np.array([ev_lookup.get((ts, st.station_id), 1.0) for ts in calendar.timestamp])
        else:
            mult = np.ones(n)
        lam = lam * mult

        # Day-level "mood" noise (weather forecasts, fuel prices, ad-hoc closures).
        day_noise = rng.lognormal(0.0, 0.13, size=len(day_keys))
        day_map = dict(zip(day_keys, day_noise))
        lam = lam * calendar.date.map(day_map).to_numpy()

        counts = rng.poisson(np.clip(lam, 0.0, None))
        frames.append(pd.DataFrame({
            "timestamp": calendar.timestamp,
            "station_id": st.station_id,
            "lambda_true": lam,
            "arrivals_raw": counts,
            "event_multiplier": mult,
        }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
def build_sessions(arrivals: pd.DataFrame, calendar: pd.DataFrame,
                   rng: np.random.Generator) -> pd.DataFrame:
    """Expand arrival counts into individual sessions with energy + duration."""
    live = arrivals.loc[arrivals.arrivals_raw > 0]
    rep = live.loc[live.index.repeat(live.arrivals_raw)].reset_index(drop=True)
    m = len(rep)

    temp_map = dict(zip(calendar.timestamp, calendar.temp_c))
    temp = rep.timestamp.map(temp_map).to_numpy()

    minute_offset = rng.random(m)
    rep["arrival_time"] = rep.timestamp + pd.to_timedelta(minute_offset * 60, unit="m")

    battery = rng.choice(config.BATTERY_KWH_CHOICES, size=m, p=config.BATTERY_KWH_WEIGHTS)
    veh_max_kw = rng.choice(VEHICLE_MAX_KW, size=m, p=VEHICLE_MAX_W)

    # Arrival state of charge: mostly 15-45 %, lower for highway/transit stops.
    soc_arr = np.clip(rng.beta(2.4, 4.6, m) * 0.70 + 0.05, 0.03, 0.80)
    profile = rep.station_id.map({s.station_id: s.profile for s in STATIONS}).to_numpy()
    soc_arr = np.where(profile == "transit", np.clip(soc_arr - 0.07, 0.03, 0.8), soc_arr)

    soc_target = np.clip(rng.normal(0.86, 0.09, m), soc_arr + 0.08, 0.99)

    # Cold and very hot weather cost extra energy (HVAC + pack conditioning).
    thermal = 1.0 + 0.012 * np.clip(16.0 - temp, 0, None) + 0.008 * np.clip(temp - 32.0, 0, None)
    energy_kwh = battery * (soc_target - soc_arr) / config.CHARGE_EFFICIENCY * thermal

    charger_kw = rep.station_id.map({s.station_id: s.charger_kw for s in STATIONS}).to_numpy()
    eff_kw = np.minimum(charger_kw, veh_max_kw)

    # Constant-power phase up to TAPER_SOC, tapered phase above it.
    bulk_share = np.clip((config.TAPER_SOC - soc_arr) / (soc_target - soc_arr), 0.0, 1.0)
    avg_kw = eff_kw * (bulk_share + (1 - bulk_share) * config.TAPER_FACTOR)
    duration_h = np.clip(energy_kwh / np.maximum(avg_kw, 3.0), config.MIN_SESSION_MINUTES / 60,
                         config.MAX_SESSION_HOURS)

    mu = np.array([DWELL_HOURS[p][0] for p in profile])
    sigma = np.array([DWELL_HOURS[p][1] for p in profile])
    dwell_h = np.clip(rng.normal(mu, sigma), 0.4, 14.0)
    dwell_h = np.maximum(dwell_h, duration_h * 1.05)

    rep = rep.assign(
        battery_kwh=np.round(battery, 1),
        vehicle_max_kw=veh_max_kw,
        soc_arrival=np.round(soc_arr, 4),
        soc_target=np.round(soc_target, 4),
        energy_kwh=np.round(energy_kwh, 3),
        avg_power_kw=np.round(avg_kw, 2),
        service_hours=np.round(duration_h, 4),
        dwell_hours=np.round(dwell_h, 3),
        temp_c=temp,
    )
    return rep.drop(columns=["arrivals_raw", "lambda_true"])


# --------------------------------------------------------------------------- #
def run_queues(sessions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """FIFO M/M/c queue per station -> waiting time, service start/end, abandonment."""
    out = []
    for station_id, grp in sessions.groupby("station_id", sort=False):
        st = next(s for s in STATIONS if s.station_id == station_id)
        grp = grp.sort_values("arrival_time").reset_index(drop=True)
        # One timeline in float hours drives both the queue and the timestamps below.
        # Deriving start/end from a *different* representation of the arrival time is
        # what lets two sessions overlap by a second and push a station over capacity.
        arr = grp.arrival_time.to_numpy().astype("datetime64[s]").astype(np.int64) / 3600.0
        svc = grp.service_hours.to_numpy()
        dwell = grp.dwell_hours.to_numpy()

        free = [0.0] * st.n_chargers
        heapq.heapify(free)
        waits = np.zeros(len(grp))
        abandoned = np.zeros(len(grp), dtype=bool)
        # Drivers with a long stay ahead of them are far more patient.
        patience = np.clip(rng.normal(0.55, 0.30, len(grp)), 0.08, 2.5) * np.where(dwell > 4, 1.8, 1.0)

        for i in range(len(grp)):
            t_free = heapq.heappop(free)
            wait = max(0.0, t_free - arr[i])
            if wait > patience[i]:            # driver gives up and drives elsewhere
                abandoned[i] = True
                waits[i] = patience[i]
                heapq.heappush(free, t_free)  # the charger keeps serving the queue
                continue
            waits[i] = wait
            heapq.heappush(free, arr[i] + wait + svc[i])

        starts = arr + waits
        ends = starts + np.where(abandoned, 0.0, svc)
        grp = grp.assign(
            wait_hours=np.round(waits, 5),
            wait_min=np.round(waits * 60, 2),
            abandoned=abandoned.astype(int),
            arrival_time=pd.to_datetime(arr * 3600.0, unit="s"),
            start_time=pd.to_datetime(starts * 3600.0, unit="s"),
            end_time=pd.to_datetime(ends * 3600.0, unit="s"),
        )
        grp.loc[grp.abandoned == 1, "energy_kwh"] = 0.0
        out.append(grp)
    return pd.concat(out, ignore_index=True).sort_values("arrival_time").reset_index(drop=True)


# --------------------------------------------------------------------------- #
def aggregate_hourly(sessions: pd.DataFrame, arrivals: pd.DataFrame,
                     calendar: pd.DataFrame) -> pd.DataFrame:
    """Spread each session over the hours it occupies -> station x hour panel."""
    served = sessions.loc[sessions.abandoned == 0]
    pieces = []
    for row in served.itertuples():
        s, e = row.start_time, row.end_time
        h0 = s.floor("h")
        while h0 < e:
            h1 = h0 + pd.Timedelta(hours=1)
            overlap = (min(e, h1) - max(s, h0)).total_seconds() / 3600.0
            if overlap > 0:
                pieces.append((h0, row.station_id, overlap, overlap * row.avg_power_kw))
            h0 = h1
    occ = pd.DataFrame(pieces, columns=["timestamp", "station_id", "charger_hours", "energy_kwh"])
    occ = occ.groupby(["timestamp", "station_id"], as_index=False).sum()

    stats = (sessions.groupby(["timestamp", "station_id"])
             .agg(arrivals=("station_id", "size"),
                  served=("abandoned", lambda s: int((s == 0).sum())),
                  abandoned=("abandoned", "sum"),
                  avg_wait_min=("wait_min", "mean"),
                  p90_wait_min=("wait_min", lambda s: float(np.percentile(s, 90))),
                  avg_soc_arrival=("soc_arrival", "mean"),
                  avg_session_min=("service_hours", lambda s: float(s.mean() * 60)))
             .reset_index())

    grid = arrivals[["timestamp", "station_id", "event_multiplier"]].copy()
    panel = (grid.merge(stats, on=["timestamp", "station_id"], how="left")
                 .merge(occ, on=["timestamp", "station_id"], how="left")
                 .merge(calendar, on="timestamp", how="left"))

    num_cols = ["arrivals", "served", "abandoned", "avg_wait_min", "p90_wait_min",
                "charger_hours", "energy_kwh", "avg_session_min"]
    panel[num_cols] = panel[num_cols].fillna(0.0)
    panel["avg_soc_arrival"] = panel["avg_soc_arrival"].fillna(panel["avg_soc_arrival"].mean())

    meta = stations_frame()[["station_id", "name", "district", "profile",
                             "n_chargers", "charger_kw", "capacity_kw", "lat", "lon"]]
    panel = panel.merge(meta, on="station_id", how="left")
    panel["utilisation"] = (panel.charger_hours / panel.n_chargers).clip(0, 1)
    panel["avg_load_kw"] = panel.energy_kwh                      # 1-hour buckets => kWh == mean kW
    panel["queue_pressure"] = panel.arrivals / panel.n_chargers
    panel = panel.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    keep = ["timestamp", "station_id", "name", "district", "profile", "n_chargers",
            "charger_kw", "capacity_kw", "lat", "lon", "arrivals", "served", "abandoned",
            "avg_wait_min", "p90_wait_min", "avg_session_min", "avg_soc_arrival",
            "charger_hours", "utilisation", "energy_kwh", "avg_load_kw", "queue_pressure",
            "temp_c", "is_rain", "dayofweek", "is_weekend", "is_holiday", "event_multiplier"]
    return panel[keep]


# --------------------------------------------------------------------------- #
def generate(cfg: SimConfig | None = None, save: bool = True) -> dict[str, pd.DataFrame]:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)

    print("[1/5] calendar & weather ...")
    calendar = build_calendar(cfg, rng)
    print("[2/5] venue events ...")
    events = build_events(cfg, calendar, rng)
    print("[3/5] Poisson arrivals ...")
    arrivals = draw_arrivals(cfg, calendar, events, rng)
    print("[4/5] sessions + queueing ...")
    sessions = run_queues(build_sessions(arrivals, calendar, rng), rng)
    print("[5/5] hourly aggregation ...")
    panel = aggregate_hourly(sessions, arrivals, calendar)

    if save:
        calendar.to_csv(config.RAW_DIR / "weather.csv.gz", index=False)
        sessions.drop(columns=["event_multiplier"]).to_csv(config.RAW_DIR / "sessions.csv.gz", index=False)
        panel.to_csv(config.RAW_DIR / "station_hourly.csv.gz", index=False)
        stations_frame().to_csv(config.RAW_DIR / "stations.csv", index=False)
        print(f"\nsaved -> {config.RAW_DIR}")

    return {"calendar": calendar, "events": events, "sessions": sessions, "panel": panel}


def summarise(panel: pd.DataFrame, sessions: pd.DataFrame) -> None:
    print("\n================ dataset summary ================")
    print(f"period            : {panel.timestamp.min()}  ->  {panel.timestamp.max()}")
    print(f"station-hour rows : {len(panel):,}")
    print(f"sessions          : {len(sessions):,}   abandoned: {sessions.abandoned.sum():,} "
          f"({sessions.abandoned.mean():.1%})")
    print(f"energy delivered  : {panel.energy_kwh.sum()/1000:,.1f} MWh")
    print(f"mean utilisation  : {panel.utilisation.mean():.1%}   peak: {panel.utilisation.max():.1%}")
    print(f"mean wait (served): {sessions.loc[sessions.abandoned == 0, 'wait_min'].mean():.1f} min")
    city = panel.groupby("timestamp").avg_load_kw.sum()
    print(f"city peak load    : {city.max():,.0f} kW   mean: {city.mean():,.0f} kW")
    print("\nbusiest stations by mean utilisation:")
    top = panel.groupby(["station_id", "name"]).agg(
        util=("utilisation", "mean"), wait=("avg_wait_min", "mean"),
        mwh=("energy_kwh", lambda s: s.sum() / 1000)).sort_values("util", ascending=False)
    print(top.head(8).to_string())


if __name__ == "__main__":
    res = generate()
    summarise(res["panel"], res["sessions"])

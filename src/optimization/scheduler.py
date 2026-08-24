"""Smart charging scheduler - the part that is deliberately *not* machine learning.

The models say what demand and load are going to be.  Deciding how to spread that
load over the evening is a constrained optimisation problem, and treating it as one
is both more honest and more effective than bolting another regressor onto it.

Decision variables
    x[v, t]   power (kW) delivered to vehicle v during hour t

Objective
    minimise   sum_t price[t] * sum_v x[v,t]        energy bill (time-of-use tariff)
             + peak_weight * P_peak                 flatten the aggregate profile
             + unmet_penalty * sum_v u[v]           energy not delivered by the deadline

Constraints
    sum_t x[v,t]  + u[v] = E_v         every car gets the energy it asked for
    0 <= x[v,t] <= P_v                 charger + vehicle power limit, zero outside
                                       the [arrival, departure] window
    sum_v x[v,t] <= cap[t]             feeder / grid ceiling for that hour
    sum_v x[v,t] <= P_peak             links the peak-shaving term

Solved with HiGHS via ``scipy.optimize.linprog``.  A greedy earliest-deadline-first
scheduler is kept as a fallback and as a reference point in the comparison report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config

try:
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    HAVE_SCIPY = True
except ImportError:                                        # pragma: no cover
    HAVE_SCIPY = False

UNMET_PENALTY = 500.0        # EGP per kWh not delivered - dominates every other term
PEAK_WEIGHT = 12.0           # EGP per kW of aggregate peak


@dataclass
class Vehicle:
    vehicle_id: str
    station_id: str
    arrival_hour: int              # index into the scheduling horizon
    deadline_hour: int             # exclusive: must be charged by the start of this hour
    energy_kwh: float
    max_kw: float

    @property
    def window(self) -> range:
        return range(self.arrival_hour, min(self.deadline_hour, 10_000))

    @property
    def laxity(self) -> float:
        """Slack hours between the time needed and the time available."""
        available = max(self.deadline_hour - self.arrival_hour, 0)
        needed = self.energy_kwh / max(self.max_kw, 1e-6)
        return available - needed


@dataclass
class ScheduleResult:
    method: str
    power: np.ndarray                       # (n_vehicles, horizon) kW
    profile: np.ndarray                     # (horizon,) aggregate kW
    unmet_kwh: np.ndarray                   # (n_vehicles,)
    cost_egp: float
    peak_kw: float
    served_pct: float
    feasible: bool = True
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
def hourly_prices(start: pd.Timestamp, horizon: int) -> np.ndarray:
    from src.recommender.engine import tariff_band
    hours = [(start + pd.Timedelta(hours=h)).hour for h in range(horizon)]
    return np.array([config.TARIFF_EGP_PER_KWH[tariff_band(h)] for h in hours])


def uncontrolled_schedule(vehicles: list[Vehicle], horizon: int,
                          prices: np.ndarray) -> ScheduleResult:
    """Baseline: everyone charges at full power the moment they plug in."""
    x = np.zeros((len(vehicles), horizon))
    for i, v in enumerate(vehicles):
        remaining = v.energy_kwh
        for t in v.window:
            if remaining <= 0 or t >= horizon:
                break
            p = min(v.max_kw, remaining)
            x[i, t] = p
            remaining -= p
    unmet = np.array([v.energy_kwh for v in vehicles]) - x.sum(axis=1)
    return _wrap("uncontrolled", x, unmet, prices, vehicles)


def greedy_schedule(vehicles: list[Vehicle], horizon: int, prices: np.ndarray,
                    cap: np.ndarray) -> ScheduleResult:
    """Least-laxity-first valley filling under an hourly power cap."""
    x = np.zeros((len(vehicles), horizon))
    used = np.zeros(horizon)
    order = sorted(range(len(vehicles)), key=lambda i: vehicles[i].laxity)
    for i in order:
        v = vehicles[i]
        remaining = v.energy_kwh
        # Cheapest hours first, but never past the deadline.
        slots = sorted([t for t in v.window if t < horizon], key=lambda t: (prices[t], t))
        for t in slots:
            if remaining <= 1e-9:
                break
            room = max(cap[t] - used[t], 0.0)
            p = min(v.max_kw, remaining, room)
            if p <= 0:
                continue
            x[i, t] = p
            used[t] += p
            remaining -= p
    unmet = np.array([v.energy_kwh for v in vehicles]) - x.sum(axis=1)
    return _wrap("greedy", x, unmet, prices, vehicles)


def optimal_schedule(vehicles: list[Vehicle], horizon: int, prices: np.ndarray,
                     cap: np.ndarray, peak_weight: float = PEAK_WEIGHT) -> ScheduleResult:
    """Linear program: minimise energy cost + peak, subject to deadlines and caps."""
    if not HAVE_SCIPY:
        res = greedy_schedule(vehicles, horizon, prices, cap)
        res.meta["note"] = "scipy unavailable - greedy fallback used"
        return res

    n = len(vehicles)
    if n == 0:
        z = np.zeros((0, horizon))
        return ScheduleResult("lp", z, np.zeros(horizon), np.zeros(0), 0.0, 0.0, 100.0)

    # Variable layout: [x(n*horizon)] [u(n)] [P_peak(1)]
    nx = n * horizon
    n_var = nx + n + 1
    idx = lambda i, t: i * horizon + t                       # noqa: E731

    c = np.zeros(n_var)
    for i in range(n):
        for t in range(horizon):
            c[idx(i, t)] = prices[t]
    c[nx:nx + n] = UNMET_PENALTY
    c[-1] = peak_weight

    # Upper bounds: power only inside the vehicle's window.
    ub = np.zeros(n_var)
    for i, v in enumerate(vehicles):
        for t in v.window:
            if t < horizon:
                ub[idx(i, t)] = v.max_kw
    ub[nx:nx + n] = [v.energy_kwh for v in vehicles]
    ub[-1] = float(cap.max()) if len(cap) else 0.0
    bounds = list(zip(np.zeros(n_var), ub))

    # Equality: delivered + unmet = requested energy (1-hour buckets => kW == kWh).
    rows, cols, vals = [], [], []
    for i, v in enumerate(vehicles):
        for t in v.window:
            if t < horizon:
                rows.append(i); cols.append(idx(i, t)); vals.append(1.0)
        rows.append(i); cols.append(nx + i); vals.append(1.0)
    A_eq = coo_matrix((vals, (rows, cols)), shape=(n, n_var)).tocsr()
    b_eq = np.array([v.energy_kwh for v in vehicles])

    # Inequalities: hourly grid cap, and the peak-linking constraint.
    rows, cols, vals, b_ub = [], [], [], []
    for t in range(horizon):
        r = len(b_ub)
        for i in range(n):
            if ub[idx(i, t)] > 0:
                rows.append(r); cols.append(idx(i, t)); vals.append(1.0)
        b_ub.append(cap[t])
    for t in range(horizon):
        r = len(b_ub)
        for i in range(n):
            if ub[idx(i, t)] > 0:
                rows.append(r); cols.append(idx(i, t)); vals.append(1.0)
        rows.append(r); cols.append(n_var - 1); vals.append(-1.0)
        b_ub.append(0.0)
    A_ub = coo_matrix((vals, (rows, cols)), shape=(len(b_ub), n_var)).tocsr()

    sol = linprog(c, A_ub=A_ub, b_ub=np.array(b_ub), A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not sol.success:
        res = greedy_schedule(vehicles, horizon, prices, cap)
        res.meta["note"] = f"LP failed ({sol.message}) - greedy fallback used"
        res.feasible = False
        return res

    x = sol.x[:nx].reshape(n, horizon)
    unmet = sol.x[nx:nx + n]
    out = _wrap("lp", x, unmet, prices, vehicles)
    out.meta["lp_objective"] = float(sol.fun)
    return out


def _wrap(method: str, x: np.ndarray, unmet: np.ndarray, prices: np.ndarray,
          vehicles: list[Vehicle]) -> ScheduleResult:
    profile = x.sum(axis=0)
    requested = sum(v.energy_kwh for v in vehicles) or 1.0
    return ScheduleResult(
        method=method,
        power=x,
        profile=profile,
        unmet_kwh=np.maximum(unmet, 0.0),
        cost_egp=float((profile * prices).sum()),
        peak_kw=float(profile.max()) if len(profile) else 0.0,
        served_pct=float((1 - max(unmet.sum(), 0.0) / requested) * 100),
    )


# --------------------------------------------------------------------------- #
def vehicles_from_forecast(fc: pd.DataFrame, station_ids: list[str] | None = None,
                           rng: np.random.Generator | None = None) -> list[Vehicle]:
    """Turn a station forecast into the fleet of cars the scheduler has to place.

    The forecast gives *how many* cars arrive each hour and *how much* energy the
    station will need; the per-car split (pack size, deadline) is sampled from the
    same fleet distributions the simulator uses.
    """
    from src.data.simulate import DWELL_HOURS, VEHICLE_MAX_KW, VEHICLE_MAX_W
    from src.data.stations import STATION_BY_ID

    rng = rng or np.random.default_rng(config.RANDOM_SEED)
    stamps = sorted(fc.timestamp.unique())
    hour_of = {ts: i for i, ts in enumerate(stamps)}
    horizon = len(stamps)
    rows = fc if station_ids is None else fc.loc[fc.station_id.isin(station_ids)]

    vehicles: list[Vehicle] = []
    for r in rows.itertuples():
        n_cars = int(round(r.arrivals))
        if n_cars <= 0:
            continue
        st = STATION_BY_ID[r.station_id]
        per_car = float(r.energy_kwh) / n_cars if r.energy_kwh > 0 else 25.0
        t0 = hour_of[r.timestamp]
        mu, sigma = DWELL_HOURS[st.profile]
        for k in range(n_cars):
            energy = float(np.clip(rng.normal(per_car, per_car * 0.25), 4.0, 90.0))
            dwell = float(np.clip(rng.normal(mu, sigma), 0.6, 14.0))
            veh_kw = float(rng.choice(VEHICLE_MAX_KW, p=VEHICLE_MAX_W))
            max_kw = min(st.charger_kw, veh_kw)
            deadline = min(t0 + int(np.ceil(dwell)), horizon)
            if deadline <= t0:
                deadline = min(t0 + 1, horizon)
            vehicles.append(Vehicle(f"{r.station_id}-{t0:02d}-{k:02d}", r.station_id,
                                    t0, deadline, round(energy, 2), max_kw))
    return vehicles


def grid_cap(horizon: int, limit_kw: float = config.GRID_SAFE_LOAD_KW) -> np.ndarray:
    return np.full(horizon, float(limit_kw))


def compare(fc: pd.DataFrame, limit_kw: float | None = None,
            station_ids: list[str] | None = None,
            peak_weight: float = PEAK_WEIGHT) -> dict:
    """Uncontrolled vs greedy vs LP on the same fleet - the operator's business case."""
    stamps = sorted(fc.timestamp.unique())
    horizon = len(stamps)
    start = pd.Timestamp(stamps[0])
    prices = hourly_prices(start, horizon)
    vehicles = vehicles_from_forecast(fc, station_ids)

    naive = uncontrolled_schedule(vehicles, horizon, prices)
    # Default ceiling: 80 % of what an uncontrolled evening would draw.
    limit = float(limit_kw if limit_kw is not None else max(naive.peak_kw * 0.80, 1.0))
    cap = grid_cap(horizon, limit)

    results = {
        "uncontrolled": naive,
        "greedy": greedy_schedule(vehicles, horizon, prices, cap),
        "optimised": optimal_schedule(vehicles, horizon, prices, cap, peak_weight),
    }
    return {"timestamps": [pd.Timestamp(s) for s in stamps], "prices": prices,
            "cap_kw": limit, "n_vehicles": len(vehicles), "results": results,
            "total_energy_kwh": sum(v.energy_kwh for v in vehicles)}


def comparison_frame(cmp: dict) -> pd.DataFrame:
    rows = []
    base = cmp["results"]["uncontrolled"]
    for name, r in cmp["results"].items():
        rows.append({
            "strategy": name,
            "peak_kw": round(r.peak_kw, 1),
            "peak_reduction_pct": round((base.peak_kw - r.peak_kw) / base.peak_kw * 100, 1),
            "energy_cost_egp": round(r.cost_egp, 0),
            "cost_saving_pct": round((base.cost_egp - r.cost_egp) / base.cost_egp * 100, 1),
            "energy_served_pct": round(r.served_pct, 2),
            "over_cap_hours": int((r.profile > cmp["cap_kw"] + 1e-6).sum()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.models.predict import Forecaster

    print("forecasting the next 24 h ...")
    fc = Forecaster().forecast(24)
    cmp = compare(fc)
    print(f"\nfleet: {cmp['n_vehicles']} vehicles, {cmp['total_energy_kwh']:,.0f} kWh requested")
    print(f"grid ceiling applied: {cmp['cap_kw']:,.0f} kW\n")
    print(comparison_frame(cmp).to_string(index=False))

    prof = pd.DataFrame({"hour": [t.strftime("%H:%M") for t in cmp["timestamps"]],
                         "price": cmp["prices"]})
    for name, r in cmp["results"].items():
        prof[name] = np.round(r.profile, 0)
    print("\naggregate load profile (kW):")
    print(prof.to_string(index=False))

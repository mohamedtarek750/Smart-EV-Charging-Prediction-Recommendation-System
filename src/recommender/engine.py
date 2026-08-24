"""Driver-side recommendation engine.

Given "I'm at X, my battery is at 22 %, I want to leave around 7 PM", rank the
nearby stations by the thing a driver actually cares about - **total time from now
until the car is charged** - and not by distance alone.

    total time = drive there + predicted queue + charging session

The queue term is the ML prediction (:mod:`src.models.predict`); the charging term
is physics (pack size, state of charge, charger rating, taper); the drive term is a
simple travel-time model.  Cost is computed from the time-of-use tariff so the app
can also answer "which one is cheapest".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from src import config
from src.data.stations import STATION_BY_ID, Station, haversine_km, nearby_stations
from src.models.predict import Forecaster, status_of

# Weighting used by the default "balanced" ranking objective.
WEIGHTS = {"fastest": (1.0, 0.0), "cheapest": (0.15, 1.0), "balanced": (1.0, 0.25)}


@dataclass
class DriverRequest:
    lat: float
    lon: float
    battery_kwh: float = 60.0
    soc_now: float = 0.22             # 0..1
    soc_target: float = 0.85
    vehicle_max_kw: float = 100.0
    when: pd.Timestamp | None = None  # when the driver wants to start charging
    leave_by: pd.Timestamp | None = None
    radius_km: float = 10.0
    objective: str = "balanced"       # fastest | cheapest | balanced


@dataclass
class StationOption:
    station_id: str
    name: str
    district: str
    lat: float
    lon: float
    n_chargers: int
    charger_kw: float
    distance_km: float
    drive_min: float
    predicted_wait_min: float
    charge_min: float
    total_min: float
    energy_kwh: float
    effective_kw: float
    cost_egp: float
    price_per_kwh: float
    utilisation: float
    free_chargers_est: float
    status: str
    ready_at: str
    fits_deadline: bool
    score: float
    reason: str


# --------------------------------------------------------------------------- #
def tariff_band(hour: int) -> str:
    if hour in config.PEAK_HOURS:
        return "peak"
    if hour in config.OFFPEAK_HOURS:
        return "offpeak"
    return "shoulder"


def price_per_kwh(hour: int, charger_kw: float) -> float:
    p = config.TARIFF_EGP_PER_KWH[tariff_band(hour)]
    return round(p * (config.DC_FAST_PREMIUM if charger_kw >= 50 else 1.0), 3)


def drive_minutes(distance_km: float, hour: int) -> float:
    speed = config.AVG_CITY_SPEED_KMH / (config.PEAK_TRAFFIC_FACTOR if hour in config.PEAK_HOURS else 1.0)
    return round(distance_km / speed * 60 + 1.5, 1)      # +1.5 min to park and plug in


def charge_minutes(energy_kwh: float, soc_now: float, soc_target: float,
                   charger_kw: float, vehicle_max_kw: float) -> tuple[float, float]:
    """Session length in minutes and the average power actually achieved.

    Mirrors the charging physics used by the simulator: constant power up to
    ``TAPER_SOC``, then a tapered phase at ``TAPER_FACTOR`` of rated power.
    """
    eff_kw = min(charger_kw, vehicle_max_kw)
    span = max(soc_target - soc_now, 1e-6)
    bulk = float(np.clip((config.TAPER_SOC - soc_now) / span, 0.0, 1.0))
    avg_kw = eff_kw * (bulk + (1 - bulk) * config.TAPER_FACTOR)
    minutes = energy_kwh / max(avg_kw, 3.0) * 60
    return round(float(np.clip(minutes, 5.0, config.MAX_SESSION_HOURS * 60)), 1), round(avg_kw, 1)


def energy_required(battery_kwh: float, soc_now: float, soc_target: float) -> float:
    return round(max(battery_kwh * (soc_target - soc_now), 0.0) / config.CHARGE_EFFICIENCY, 2)


# --------------------------------------------------------------------------- #
class Recommender:
    def __init__(self, forecaster: Forecaster | None = None, horizon_hours: int = 24):
        self.fc = forecaster or Forecaster()
        self.horizon_hours = horizon_hours
        self._cache: dict[pd.Timestamp, pd.DataFrame] = {}

    def _forecast_for(self, when: pd.Timestamp) -> pd.DataFrame:
        """Station forecast for the hour containing ``when`` (cached per start hour)."""
        start = self.fc.last_ts + pd.Timedelta(hours=1)
        if when < start:
            when = start
        horizon = int((when - start).total_seconds() // 3600) + 1
        horizon = max(1, min(horizon, self.horizon_hours))
        key = start
        if key not in self._cache:
            self._cache[key] = self.fc.forecast(self.horizon_hours, start=start)
        fc = self._cache[key]
        target_hour = when.floor("h")
        slot = fc.loc[fc.timestamp == target_hour]
        if slot.empty:                                  # beyond horizon -> use the last hour
            slot = fc.loc[fc.timestamp == fc.timestamp.max()]
        return slot

    # ------------------------------------------------------------------ main
    def recommend(self, req: DriverRequest, top_k: int = 5) -> dict:
        when = pd.Timestamp(req.when) if req.when is not None else self.fc.last_ts + pd.Timedelta(hours=1)
        slot = self._forecast_for(when)
        hour = when.hour

        energy = energy_required(req.battery_kwh, req.soc_now, req.soc_target)
        range_km = req.battery_kwh * req.soc_now / config.CONSUMPTION_KWH_PER_KM

        options: list[StationOption] = []
        for st, dist in nearby_stations(req.lat, req.lon, req.radius_km, limit=12):
            row = slot.loc[slot.station_id == st.station_id]
            if row.empty:
                continue
            row = row.iloc[0]
            wait = float(row.avg_wait_min)
            drive = drive_minutes(dist, hour)
            c_min, eff_kw = charge_minutes(energy, req.soc_now, req.soc_target,
                                           st.charger_kw, req.vehicle_max_kw)
            total = round(drive + wait + c_min, 1)
            price = price_per_kwh(hour, st.charger_kw)
            cost = round(energy * price, 2)
            ready = when + pd.Timedelta(minutes=total)
            fits = True if req.leave_by is None else ready <= pd.Timestamp(req.leave_by)

            options.append(StationOption(
                station_id=st.station_id, name=st.name, district=st.district,
                lat=st.lat, lon=st.lon, n_chargers=st.n_chargers, charger_kw=st.charger_kw,
                distance_km=round(dist, 2), drive_min=drive,
                predicted_wait_min=round(wait, 1), charge_min=c_min, total_min=total,
                energy_kwh=energy, effective_kw=eff_kw, cost_egp=cost, price_per_kwh=price,
                utilisation=round(float(row.utilisation), 3),
                free_chargers_est=float(row.free_chargers_est),
                status=status_of(wait), ready_at=ready.strftime("%Y-%m-%d %H:%M"),
                fits_deadline=bool(fits), score=0.0, reason="",
            ))

        if not options:
            return {"error": "no station within range", "request": asdict(req)}

        options = self._score(options, req)
        best = options[0]

        return {
            "requested_at": str(when),
            "energy_needed_kwh": energy,
            "current_range_km": round(range_km, 1),
            "objective": req.objective,
            "recommendation": {
                "station_id": best.station_id,
                "name": best.name,
                "message": (f"Go to {best.name} - predicted wait {best.predicted_wait_min:.0f} min, "
                            f"ready by {best.ready_at[-5:]} for about {best.cost_egp:.0f} EGP."),
                "reason": best.reason,
            },
            "options": [asdict(o) for o in options[:top_k]],
            "unreachable": [o.station_id for o in options if o.distance_km > range_km],
        }

    # ----------------------------------------------------------------- scoring
    def _score(self, options: list[StationOption], req: DriverRequest) -> list[StationOption]:
        w_time, w_cost = WEIGHTS.get(req.objective, WEIGHTS["balanced"])
        times = np.array([o.total_min for o in options])
        costs = np.array([o.cost_egp for o in options])
        t_norm = (times - times.min()) / max(float(np.ptp(times)), 1e-6)
        c_norm = (costs - costs.min()) / max(float(np.ptp(costs)), 1e-6)

        range_km = req.battery_kwh * req.soc_now / config.CONSUMPTION_KWH_PER_KM
        for o, tn, cn in zip(options, t_norm, c_norm):
            penalty = 0.0
            if not o.fits_deadline:
                penalty += 1.5                       # will not be done before departure
            if o.distance_km > range_km * 0.85:
                penalty += 2.5                       # not safely reachable on this charge
            o.score = round(float(w_time * tn + w_cost * cn + penalty), 4)

        options.sort(key=lambda o: (o.score, o.total_min))
        fastest = min(options, key=lambda o: o.total_min)
        cheapest = min(options, key=lambda o: o.cost_egp)
        for o in options:
            bits = []
            if o is fastest:
                bits.append("fastest overall")
            if o is cheapest:
                bits.append("cheapest")
            if o.status == "green":
                bits.append("no queue expected")
            elif o.status == "red":
                bits.append(f"busy - {o.predicted_wait_min:.0f} min queue predicted")
            if not o.fits_deadline:
                bits.append("will not finish before you leave")
            if o.distance_km > range_km * 0.85:
                bits.append("outside safe range on the current charge")
            o.reason = "; ".join(bits) or f"{o.distance_km:.1f} km away, {o.total_min:.0f} min in total"
        return options


# --------------------------------------------------------------------------- #
def demo() -> None:
    rec = Recommender()
    mall = STATION_BY_ID["ST12"]
    req = DriverRequest(lat=mall.lat + 0.008, lon=mall.lon - 0.006, battery_kwh=60,
                        soc_now=0.22, soc_target=0.85, vehicle_max_kw=100,
                        when=rec.fc.last_ts + pd.Timedelta(hours=19), objective="balanced")
    out = rec.recommend(req, top_k=12)

    print(f"Driver near Downtown, battery 22 %, wants 85 % "
          f"({out['energy_needed_kwh']} kWh) at {out['requested_at']}\n")
    cols = ["name", "distance_km", "drive_min", "predicted_wait_min", "charge_min",
            "total_min", "cost_egp", "status"]
    print(pd.DataFrame(out["options"])[cols].to_string(index=False))
    print(f"\nAI recommendation: {out['recommendation']['message']}")
    print(f"why: {out['recommendation']['reason']}")


if __name__ == "__main__":
    demo()

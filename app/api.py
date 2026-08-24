"""FastAPI service exposing the whole system.

    GET  /health                 liveness + which models are loaded
    GET  /stations               the 20-station network
    GET  /forecast               station-level forecast for the next N hours
    GET  /city                   operator view: demand, load, grid headroom, alerts
    GET  /schedule               smart-charging plan vs uncontrolled charging
    GET  /metrics                held-out accuracy of the three models
    POST /recommend              driver view: ranked stations + one recommendation

Run with:  uvicorn app.api:app --reload
"""
from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config                                             # noqa: E402
from src.data.stations import stations_frame                       # noqa: E402
from src.models.predict import Forecaster, city_summary            # noqa: E402
from src.optimization import scheduler                             # noqa: E402
from src.recommender.engine import DriverRequest, Recommender      # noqa: E402

STATE: dict = {}
MAX_HORIZON = 72


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("warming up models ...")
    fc = Forecaster()
    STATE["forecaster"] = fc
    STATE["recommender"] = Recommender(fc)
    STATE["forecast_cache"] = {}
    print(f"ready - history ends {fc.last_ts}")
    yield
    STATE.clear()


app = FastAPI(
    title="Smart EV Charging Prediction & Recommendation System",
    description="Demand, waiting-time and grid-load forecasting for the "
                "New Administrative Capital charging network.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
def get_forecast(hours: int) -> pd.DataFrame:
    hours = max(1, min(hours, MAX_HORIZON))
    cache = STATE["forecast_cache"]
    if hours not in cache:
        cache[hours] = STATE["forecaster"].forecast(hours)
    return cache[hours]


def records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M")
    return json.loads(out.to_json(orient="records"))


# --------------------------------------------------------------------------- #
class RecommendRequest(BaseModel):
    lat: float = Field(..., description="driver latitude", examples=[30.0065])
    lon: float = Field(..., description="driver longitude", examples=[31.7258])
    battery_kwh: float = Field(60.0, gt=5, le=200)
    soc_now: float = Field(0.22, ge=0.0, le=1.0, description="current state of charge 0-1")
    soc_target: float = Field(0.85, ge=0.05, le=1.0)
    vehicle_max_kw: float = Field(100.0, gt=1, le=400)
    when: str | None = Field(None, description="when to start charging, ISO 8601")
    leave_by: str | None = Field(None, description="hard departure time, ISO 8601")
    radius_km: float = Field(10.0, gt=0, le=40)
    objective: str = Field("balanced", pattern="^(fastest|cheapest|balanced)$")
    top_k: int = Field(5, ge=1, le=20)


# --------------------------------------------------------------------------- #
@app.get("/health", tags=["meta"])
def health():
    fc: Forecaster = STATE["forecaster"]
    return {
        "status": "ok",
        "history_ends": str(fc.last_ts),
        "models": {t: {"features": len(m.features), "log_target": m.log_target}
                   for t, m in fc.models.items()},
        "stations": len(stations_frame()),
    }


@app.get("/stations", tags=["network"])
def stations():
    return records(stations_frame())


@app.get("/metrics", tags=["meta"])
def metrics():
    path = config.REPORTS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "metrics.json not found - run python -m src.models.train")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/forecast", tags=["forecast"])
def forecast(hours: int = Query(24, ge=1, le=MAX_HORIZON),
             station_id: str | None = None):
    fc = get_forecast(hours)
    if station_id:
        fc = fc.loc[fc.station_id == station_id.upper()]
        if fc.empty:
            raise HTTPException(404, f"unknown station {station_id}")
    return {"horizon_hours": hours, "rows": len(fc), "forecast": records(fc)}


@app.get("/city", tags=["forecast"])
def city(hours: int = Query(24, ge=1, le=MAX_HORIZON)):
    fc = get_forecast(hours)
    summary = city_summary(fc)
    peak = summary.loc[summary.expected_load_kw.idxmax()]
    hot = (fc.groupby(["station_id", "name"])
             .agg(peak_wait_min=("avg_wait_min", "max"),
                  peak_load_kw=("avg_load_kw", "max"),
                  red_hours=("status", lambda s: int((s == "red").sum())))
             .reset_index()
             .sort_values("peak_wait_min", ascending=False)
             .head(5))
    return {
        "horizon_hours": hours,
        "grid_safe_load_kw": config.GRID_SAFE_LOAD_KW,
        "peak": {"timestamp": peak.timestamp.strftime("%Y-%m-%d %H:%M"),
                 "expected_load_kw": round(float(peak.expected_load_kw), 1),
                 "expected_evs": round(float(peak.expected_evs), 1),
                 "grid_headroom_kw": round(float(peak.grid_headroom_kw), 1),
                 "overload_risk": bool(peak.overload_risk)},
        "hourly": records(summary),
        "stations_to_watch": records(hot),
    }


@app.get("/schedule", tags=["optimisation"])
def schedule(hours: int = Query(24, ge=4, le=MAX_HORIZON),
             limit_kw: float | None = Query(None, gt=0),
             peak_weight: float = Query(scheduler.PEAK_WEIGHT, ge=0)):
    fc = get_forecast(hours)
    cmp = scheduler.compare(fc, limit_kw=limit_kw, peak_weight=peak_weight)
    table = scheduler.comparison_frame(cmp)
    profiles = {name: [round(float(v), 1) for v in r.profile]
                for name, r in cmp["results"].items()}
    return {
        "horizon_hours": hours,
        "cap_kw": round(cmp["cap_kw"], 1),
        "n_vehicles": cmp["n_vehicles"],
        "total_energy_kwh": round(cmp["total_energy_kwh"], 1),
        "timestamps": [t.strftime("%Y-%m-%d %H:%M") for t in cmp["timestamps"]],
        "price_egp_per_kwh": [round(float(p), 2) for p in cmp["prices"]],
        "load_profiles_kw": profiles,
        "comparison": records(table),
    }


@app.post("/recommend", tags=["driver"])
def recommend(req: RecommendRequest):
    rec: Recommender = STATE["recommender"]
    dr = DriverRequest(
        lat=req.lat, lon=req.lon, battery_kwh=req.battery_kwh,
        soc_now=req.soc_now, soc_target=max(req.soc_target, req.soc_now + 0.02),
        vehicle_max_kw=req.vehicle_max_kw,
        when=pd.Timestamp(req.when) if req.when else None,
        leave_by=pd.Timestamp(req.leave_by) if req.leave_by else None,
        radius_km=req.radius_km, objective=req.objective,
    )
    out = rec.recommend(dr, top_k=req.top_k)
    if "error" in out:
        raise HTTPException(404, out["error"])
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)

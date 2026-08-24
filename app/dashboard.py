"""Streamlit front end - the two faces of the system in one app.

    Driver          where should I charge right now, and how long will it take?
    City operator   what is the network going to do over the next 24 hours?
    Smart charging  what does the optimiser save the grid?
    Model quality   how accurate is any of this, on data the models never saw?

Run with:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config                                              # noqa: E402
from src.data.stations import STATIONS, STATION_BY_ID, stations_frame  # noqa: E402
from src.models.predict import Forecaster, city_summary             # noqa: E402
from src.optimization import scheduler                              # noqa: E402
from src.recommender.engine import DriverRequest, Recommender       # noqa: E402

st.set_page_config(page_title="Smart EV Charging - New Administrative Capital",
                   page_icon="⚡", layout="wide")

STATUS_COLOUR = {"green": "#16a34a", "amber": "#d97706", "red": "#dc2626"}
PLOT_TEMPLATE = "plotly_white"


# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading models and history …")
def get_engines():
    fc = Forecaster()
    return fc, Recommender(fc)


@st.cache_data(show_spinner="Forecasting the network …")
def get_forecast(hours: int) -> pd.DataFrame:
    fc, _ = get_engines()
    return fc.forecast(hours)


@st.cache_data
def get_metrics() -> dict:
    path = config.REPORTS_DIR / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data
def get_test_predictions() -> pd.DataFrame:
    path = config.REPORTS_DIR / "test_predictions.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["timestamp"])


@st.cache_data(show_spinner="Solving the charging schedule …")
def get_schedule(hours: int, limit_kw: float | None, peak_weight: float) -> dict:
    fc = get_forecast(hours)
    cmp = scheduler.compare(fc, limit_kw=limit_kw, peak_weight=peak_weight)
    return {
        "cap_kw": cmp["cap_kw"],
        "n_vehicles": cmp["n_vehicles"],
        "total_energy_kwh": cmp["total_energy_kwh"],
        "timestamps": cmp["timestamps"],
        "prices": cmp["prices"].tolist(),
        "profiles": {k: v.profile.tolist() for k, v in cmp["results"].items()},
        "table": scheduler.comparison_frame(cmp),
    }


def badge(status: str) -> str:
    dot = {"green": "🟢", "amber": "🟡", "red": "🔴"}[status]
    return f"{dot} {status}"


# --------------------------------------------------------------------------- #
forecaster, recommender = get_engines()
NOW = forecaster.last_ts + pd.Timedelta(hours=1)

st.title("⚡ Smart EV Charging — New Administrative Capital")
st.caption(
    f"20 stations · {sum(s.n_chargers for s in STATIONS)} chargers · "
    f"{stations_frame().capacity_kw.sum():,.0f} kW installed — "
    f"forecasting from **{NOW:%A %d %B %Y, %H:%M}**"
)

with st.sidebar:
    st.header("Settings")
    horizon = st.slider("Forecast horizon (hours)", 6, 48, 24, step=6)
    st.divider()
    st.markdown(
        f"**Service levels**\n\n"
        f"- 🟢 wait ≤ {config.WAIT_GREEN_MIN:.0f} min\n"
        f"- 🟡 wait ≤ {config.WAIT_AMBER_MIN:.0f} min\n"
        f"- 🔴 above that\n\n"
        f"**Grid ceiling** {config.GRID_SAFE_LOAD_KW:,.0f} kW"
    )
    st.divider()
    st.caption("Tariff (EGP/kWh): "
               + " · ".join(f"{k} {v:.2f}" for k, v in config.TARIFF_EGP_PER_KWH.items()))

fc_all = get_forecast(horizon)

tab_driver, tab_city, tab_opt, tab_model = st.tabs(
    ["🚗 Driver", "🏙️ City operator", "🔋 Smart charging", "📈 Model quality"])


# =========================================================== DRIVER ========= #
with tab_driver:
    st.subheader("Where should I charge?")
    left, right = st.columns([1, 2], gap="large")

    with left:
        anchor_name = st.selectbox(
            "I'm near", [s.name for s in STATIONS], index=11,
            help="Used as the driver's current position.")
        anchor = next(s for s in STATIONS if s.name == anchor_name)

        soc_now = st.slider("Battery now (%)", 2, 95, 22) / 100
        soc_target = st.slider("Charge to (%)", 30, 100, 85) / 100
        battery_kwh = st.select_slider("Battery size (kWh)", [40, 50, 60, 75, 82, 100], value=60)
        vehicle_max_kw = st.select_slider("Car max charging power (kW)",
                                          [11, 22, 50, 100, 150], value=100)
        hour_offset = st.slider("Start charging in (hours from now)", 0, horizon - 1, 18)
        objective = st.radio("Optimise for", ["balanced", "fastest", "cheapest"], horizontal=True)
        use_deadline = st.checkbox("I must leave by a fixed time", value=False)
        when = NOW + pd.Timedelta(hours=hour_offset)
        leave_by = None
        if use_deadline:
            leave_hours = st.slider("Leaving in (hours from now)", hour_offset + 1,
                                    horizon + 6, hour_offset + 2)
            leave_by = NOW + pd.Timedelta(hours=leave_hours)
        st.caption(f"Charging window starts **{when:%a %H:%M}**"
                   + (f", must be done by **{leave_by:%a %H:%M}**" if leave_by is not None else ""))

    req = DriverRequest(lat=anchor.lat, lon=anchor.lon, battery_kwh=float(battery_kwh),
                        soc_now=soc_now, soc_target=max(soc_target, soc_now + 0.02),
                        vehicle_max_kw=float(vehicle_max_kw), when=when, leave_by=leave_by,
                        radius_km=12.0, objective=objective)
    result = recommender.recommend(req, top_k=8)

    with right:
        if "error" in result:
            st.error(result["error"])
        else:
            best = result["recommendation"]
            top = result["options"][0]
            st.success(f"**{best['message']}**\n\n_{best['reason']}_")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Predicted queue", f"{top['predicted_wait_min']:.0f} min")
            k2.metric("Charging time", f"{top['charge_min']:.0f} min")
            k3.metric("Total time", f"{top['total_min']:.0f} min")
            k4.metric("Estimated cost", f"{top['cost_egp']:,.0f} EGP")

            opts = pd.DataFrame(result["options"])
            opts["state"] = opts.status.map(badge)
            show = opts[["name", "state", "distance_km", "drive_min", "predicted_wait_min",
                         "charge_min", "total_min", "cost_egp", "ready_at"]]
            show.columns = ["Station", "Queue", "km", "Drive", "Wait", "Charge",
                            "Total (min)", "Cost (EGP)", "Ready at"]
            st.dataframe(show, hide_index=True, width="stretch")

            fig = px.bar(opts.sort_values("total_min"), x="total_min", y="name",
                         orientation="h", color="status", color_discrete_map=STATUS_COLOUR,
                         labels={"total_min": "minutes until fully charged", "name": ""},
                         template=PLOT_TEMPLATE, height=340,
                         hover_data=["drive_min", "predicted_wait_min", "charge_min"])
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")

    st.markdown("##### Predicted queue at every station, hour by hour")
    pick = st.multiselect("Stations", [s.name for s in STATIONS],
                          default=[o["name"] for o in result.get("options", [])[:4]])
    if pick:
        sub = fc_all.loc[fc_all.name.isin(pick)]
        fig = px.line(sub, x="timestamp", y="avg_wait_min", color="name", markers=False,
                      template=PLOT_TEMPLATE, height=320,
                      labels={"avg_wait_min": "predicted wait (min)", "timestamp": "",
                              "name": ""})
        fig.add_hline(y=config.WAIT_AMBER_MIN, line_dash="dot", line_color="#dc2626",
                      annotation_text="red threshold")
        fig.add_vline(x=when, line_dash="dash", line_color="#64748b")
        st.plotly_chart(fig, width="stretch")


# ====================================================== CITY OPERATOR ======= #
with tab_city:
    st.subheader("Network outlook")
    summary = city_summary(fc_all)
    peak = summary.loc[summary.expected_load_kw.idxmax()]
    total_evs = summary.expected_evs.sum()
    red_hours = int((fc_all.status == "red").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"EVs expected ({horizon} h)", f"{total_evs:,.0f}")
    c2.metric("Peak load", f"{peak.expected_load_kw:,.0f} kW", f"at {peak.timestamp:%H:%M}")
    c3.metric("Grid headroom at peak", f"{peak.grid_headroom_kw:,.0f} kW",
              delta="over limit" if peak.overload_risk else "within limit",
              delta_color="inverse" if peak.overload_risk else "normal")
    c4.metric("Congested station-hours", f"{red_hours}",
              help="station-hours with a predicted queue above the red threshold")

    if peak.overload_risk:
        st.error(f"⚠️ Predicted load exceeds the {config.GRID_SAFE_LOAD_KW:,.0f} kW ceiling at "
                 f"{peak.timestamp:%H:%M}. Use the Smart charging tab to shave the peak.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=summary.timestamp, y=summary.expected_load_kw, name="predicted load",
                             fill="tozeroy", line=dict(color="#2563eb", width=2)))
    fig.add_hline(y=config.GRID_SAFE_LOAD_KW, line_dash="dash", line_color="#dc2626",
                  annotation_text="safe ceiling")
    fig.add_trace(go.Scatter(x=summary.timestamp, y=summary.expected_evs * 20, name="EV arrivals (×20)",
                             line=dict(color="#f59e0b", width=1.5, dash="dot"), yaxis="y"))
    fig.update_layout(template=PLOT_TEMPLATE, height=360, margin=dict(l=0, r=0, t=20, b=0),
                      yaxis_title="kW", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")

    m1, m2 = st.columns([3, 2], gap="large")
    with m1:
        st.markdown("##### Predicted waiting time — station × hour")
        pivot = fc_all.pivot_table(index="name", columns="timestamp",
                                   values="avg_wait_min", aggfunc="mean")
        pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=False).index)
        heat = px.imshow(pivot, aspect="auto", color_continuous_scale="RdYlGn_r",
                         labels=dict(color="min"), template=PLOT_TEMPLATE, height=520)
        heat.update_xaxes(tickformat="%H:%M", title="")
        heat.update_yaxes(title="")
        st.plotly_chart(heat, width="stretch")

    with m2:
        st.markdown("##### Network map at a chosen hour")
        hour_pick = st.select_slider(
            "Hour", options=list(summary.timestamp),
            value=summary.loc[summary.expected_load_kw.idxmax(), "timestamp"],
            format_func=lambda t: f"{t:%a %H:%M}")
        snap = fc_all.loc[fc_all.timestamp == hour_pick].copy()
        snap["size"] = snap.arrivals.clip(lower=0.6)
        mp = px.scatter_map(
            snap, lat="lat", lon="lon", color="status", size="size",
            color_discrete_map=STATUS_COLOUR, size_max=26, zoom=10.2,
            hover_name="name",
            hover_data={"arrivals": ":.1f", "avg_wait_min": ":.1f", "avg_load_kw": ":.0f",
                        "lat": False, "lon": False, "size": False},
            map_style="carto-positron", height=430)
        mp.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                         legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(mp, width="stretch")

        st.markdown("##### Stations to watch")
        watch = (fc_all.groupby(["station_id", "name"])
                 .agg(peak_wait=("avg_wait_min", "max"),
                      peak_kw=("avg_load_kw", "max"),
                      red_hours=("status", lambda s: int((s == "red").sum())))
                 .reset_index().sort_values("peak_wait", ascending=False).head(6))
        watch.columns = ["ID", "Station", "Peak wait (min)", "Peak kW", "Red hours"]
        st.dataframe(watch.round(1), hide_index=True, width="stretch")


# ======================================================= SMART CHARGING ===== #
with tab_opt:
    st.subheader("Smart charging: shifting load instead of adding cables")
    st.markdown(
        "Forecasting says *what will happen*. This tab decides *what to do about it*: "
        "a linear program spreads each car's energy over the hours it is parked, "
        "subject to its departure deadline and an hourly grid ceiling."
    )

    o1, o2 = st.columns(2)
    with o1:
        auto_cap = st.checkbox("Derive the ceiling from uncontrolled peak (80 %)", value=True)
        cap_kw = None
        if not auto_cap:
            cap_kw = st.slider("Grid ceiling (kW)", 500, 6000,
                               int(config.GRID_SAFE_LOAD_KW), step=100)
    with o2:
        peak_weight = st.slider("Peak-shaving weight (EGP per kW of peak)", 0.0, 40.0,
                                float(scheduler.PEAK_WEIGHT), step=1.0,
                                help="0 = minimise the energy bill only; higher = flatten harder.")

    sched = get_schedule(horizon, cap_kw, peak_weight)
    table = sched["table"]
    opt_row = table.loc[table.strategy == "optimised"].iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Vehicles scheduled", f"{sched['n_vehicles']:,}")
    k2.metric("Energy requested", f"{sched['total_energy_kwh']:,.0f} kWh")
    k3.metric("Peak reduction", f"{opt_row.peak_reduction_pct:.1f} %",
              f"{table.loc[0, 'peak_kw']:,.0f} → {opt_row.peak_kw:,.0f} kW")
    k4.metric("Energy delivered on time", f"{opt_row.energy_served_pct:.1f} %")

    prof = pd.DataFrame({"timestamp": sched["timestamps"], **sched["profiles"]})
    fig = go.Figure()
    colours = {"uncontrolled": "#dc2626", "greedy": "#f59e0b", "optimised": "#16a34a"}
    for name in ("uncontrolled", "greedy", "optimised"):
        fig.add_trace(go.Scatter(x=prof.timestamp, y=prof[name], name=name,
                                 line=dict(width=2.5, color=colours[name])))
    fig.add_hline(y=sched["cap_kw"], line_dash="dash", line_color="#334155",
                  annotation_text=f"ceiling {sched['cap_kw']:,.0f} kW")
    fig.update_layout(template=PLOT_TEMPLATE, height=380, yaxis_title="aggregate load (kW)",
                      margin=dict(l=0, r=0, t=20, b=0),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")

    t1, t2 = st.columns([2, 1], gap="large")
    with t1:
        show = table.copy()
        show.columns = ["Strategy", "Peak kW", "Peak ↓ %", "Cost EGP", "Cost ↓ %",
                        "Served %", "Hours over cap"]
        st.dataframe(show, hide_index=True, width="stretch")
    with t2:
        price_df = pd.DataFrame({"hour": [t.strftime("%H:%M") for t in sched["timestamps"]],
                                 "EGP/kWh": sched["prices"]})
        pf = px.bar(price_df, x="hour", y="EGP/kWh", template=PLOT_TEMPLATE, height=240,
                    title="Time-of-use tariff")
        pf.update_layout(margin=dict(l=0, r=0, t=40, b=0), xaxis_title="")
        st.plotly_chart(pf, width="stretch")


# ========================================================= MODEL QUALITY ==== #
with tab_model:
    st.subheader("How good are the predictions?")
    metrics = get_metrics()
    if not metrics:
        st.warning("Run `python -m src.models.train` to produce the metrics report.")
    else:
        st.caption(f"Time-based split — models trained on data up to "
                   f"{metrics['models']['arrivals']['cutoff'][:10]}, scored on the "
                   f"{config.TEST_MONTHS} months after it "
                   f"({metrics['models']['arrivals']['n_test']:,} unseen station-hours).")

        rows = []
        for target, m in metrics["models"].items():
            rows.append({
                "Target": target,
                "MAE": round(m["model"]["mae"], 3),
                "RMSE": round(m["model"]["rmse"], 3),
                "R²": round(m["model"]["r2"], 3),
                "Baseline MAE": round(m["baseline"]["mae"], 3),
                "Improvement": f"{m['uplift_mae_pct']:.1f} %",
                "Baseline used": m["baseline_name"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        wm = metrics["models"]["avg_wait_min"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Traffic-light accuracy", f"{wm['bucket_accuracy']:.1%}",
                  help="green / amber / red class predicted correctly")
        c2.metric("Predictions within 5 min", f"{wm['within_5min_pct']:.1f} %")
        c3.metric("Demand model R²", f"{metrics['models']['arrivals']['model']['r2']:.3f}")

        with st.expander("End-to-end cascade — arrivals predicted, not observed"):
            st.markdown(
                "The wait and load models take the number of arrivals as an input. "
                "In production that input is itself a forecast, so the errors compound. "
                "These are the honest end-to-end numbers:")
            casc = pd.DataFrame(metrics["cascade_end_to_end"]).T.round(3)
            casc.index.name = "target"
            st.dataframe(casc.reset_index(), hide_index=True, width="stretch")

        preds = get_test_predictions()
        if not preds.empty:
            target = st.selectbox("Inspect a target", ["arrivals", "avg_wait_min", "energy_kwh"])
            station = st.selectbox("Station", sorted(preds.name.unique()),
                                   index=sorted(preds.name.unique()).index("Downtown Mall Level B1")
                                   if "Downtown Mall Level B1" in preds.name.unique() else 0)
            days = st.slider("Days to show", 2, 14, 5)

            sub = preds.loc[preds.name == station].sort_values("timestamp")
            sub = sub.loc[sub.timestamp <= sub.timestamp.min() + pd.Timedelta(days=days)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sub.timestamp, y=sub[f"{target}_true"], name="actual",
                                     line=dict(color="#0f172a", width=1.8)))
            fig.add_trace(go.Scatter(x=sub.timestamp, y=sub[f"{target}_pred"], name="predicted",
                                     line=dict(color="#2563eb", width=1.8, dash="dot")))
            fig.update_layout(template=PLOT_TEMPLATE, height=340, title=f"{station} — {target}",
                              margin=dict(l=0, r=0, t=40, b=0),
                              legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, width="stretch")

            e1, e2 = st.columns(2)
            with e1:
                sc = px.scatter(preds.sample(min(6000, len(preds)), random_state=0),
                                x=f"{target}_true", y=f"{target}_pred", opacity=0.25,
                                template=PLOT_TEMPLATE, height=340,
                                labels={f"{target}_true": "actual", f"{target}_pred": "predicted"},
                                title="Predicted vs actual (held-out)")
                lim = float(preds[f"{target}_true"].quantile(0.999))
                sc.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                             line=dict(color="#dc2626", dash="dash"))
                sc.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(sc, width="stretch")
            with e2:
                err = preds[f"{target}_pred"] - preds[f"{target}_true"]
                hs = px.histogram(err, nbins=70, template=PLOT_TEMPLATE, height=340,
                                  title="Error distribution", labels={"value": "predicted − actual"})
                hs.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(hs, width="stretch")

        st.markdown("##### What the models actually rely on")
        imp_target = st.radio("Model", list(metrics["top_features"].keys()), horizontal=True)
        imp = pd.DataFrame(metrics["top_features"][imp_target], columns=["feature", "importance"])
        fig = px.bar(imp.head(12).iloc[::-1], x="importance", y="feature", orientation="h",
                     template=PLOT_TEMPLATE, height=380,
                     labels={"importance": "drop in R² when shuffled", "feature": ""})
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")

st.divider()
st.caption("Data is simulated by `src/data/simulate.py` — a discrete-event model of the "
           "New Administrative Capital network. Everything downstream (features, models, "
           "recommendations, optimisation) is production-shaped and works unchanged on real "
           "charge-point records with the same schema.")

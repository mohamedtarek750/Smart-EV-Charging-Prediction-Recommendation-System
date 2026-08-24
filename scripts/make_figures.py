"""Generate the figures used in the README / report.

    python -m scripts.make_figures     ->  artifacts/reports/figures/*.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.features.build_features import load_panel

FIG_DIR = config.REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.autolayout": True,
})
GREEN, AMBER, RED, BLUE = "#16a34a", "#d97706", "#dc2626", "#2563eb"


def fig_demand_profiles(panel: pd.DataFrame) -> None:
    p = panel.assign(hour=panel.timestamp.dt.hour)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for prof, grp in p.groupby("profile", observed=True):
        wd = grp.loc[grp.is_weekend == 0].groupby("hour").arrivals.mean()
        we = grp.loc[grp.is_weekend == 1].groupby("hour").arrivals.mean()
        axes[0].plot(wd.index, wd.to_numpy(), label=prof, lw=1.6)
        axes[1].plot(we.index, we.to_numpy(), label=prof, lw=1.6)
    axes[0].set_title("Weekday arrivals per station-hour")
    axes[1].set_title("Weekend (Fri–Sat) arrivals per station-hour")
    for ax in axes:
        ax.set_xlabel("hour of day")
        ax.set_xticks(range(0, 24, 3))
    axes[0].set_ylabel("mean EVs / hour")
    axes[1].legend(fontsize=7, ncol=2, frameon=False)
    fig.savefig(FIG_DIR / "demand_profiles.png")
    plt.close(fig)


def fig_station_heatmap(panel: pd.DataFrame) -> None:
    p = panel.assign(hour=panel.timestamp.dt.hour)
    piv = p.pivot_table(index="name", columns="hour", values="utilisation", aggfunc="mean")
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=piv.to_numpy().max())
    ax.set_xticks(range(0, 24, 2), [str(h) for h in range(0, 24, 2)])
    ax.set_yticks(range(len(piv)), piv.index, fontsize=7)
    ax.set_xlabel("hour of day")
    ax.set_title("Mean charger utilisation by station and hour")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85, label="utilisation")
    fig.savefig(FIG_DIR / "station_heatmap.png")
    plt.close(fig)


def fig_wait_vs_utilisation(panel: pd.DataFrame) -> None:
    p = panel.loc[panel.arrivals > 0].copy()
    p["bin"] = (p.utilisation * 20).round() / 20
    g = p.groupby("bin").avg_wait_min.agg(["mean", "count"])
    g = g.loc[g["count"] > 50]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(g.index * 100, g["mean"], color=BLUE, lw=2)
    ax.axhline(config.WAIT_GREEN_MIN, color=GREEN, ls="--", lw=1, label="green threshold")
    ax.axhline(config.WAIT_AMBER_MIN, color=RED, ls="--", lw=1, label="red threshold")
    ax.set_xlabel("charger utilisation (%)")
    ax.set_ylabel("mean waiting time (min)")
    ax.set_title("Queueing is non-linear in utilisation")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG_DIR / "wait_vs_utilisation.png")
    plt.close(fig)


def fig_forecast_vs_actual() -> None:
    path = config.REPORTS_DIR / "test_predictions.csv.gz"
    if not path.exists():
        print("skip forecast_vs_actual: run python -m src.models.train first")
        return
    pr = pd.read_csv(path, parse_dates=["timestamp"])
    station = "Downtown Mall Level B1"
    sub = pr.loc[pr.name == station].sort_values("timestamp")
    sub = sub.loc[sub.timestamp <= sub.timestamp.min() + pd.Timedelta(days=5)]

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
    for ax, target, label in ((axes[0], "arrivals", "EV arrivals / hour"),
                              (axes[1], "energy_kwh", "station load (kWh/h)")):
        ax.plot(sub.timestamp, sub[f"{target}_true"], color="#0f172a", lw=1.5, label="actual")
        ax.plot(sub.timestamp, sub[f"{target}_pred"], color=BLUE, lw=1.5, ls="--", label="predicted")
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=8, ncol=2)
    axes[0].set_title(f"{station} — held-out test period")
    fig.savefig(FIG_DIR / "forecast_vs_actual.png")
    plt.close(fig)


def fig_peak_shaving() -> None:
    from src.models.predict import Forecaster
    from src.optimization import scheduler

    fc = Forecaster().forecast(24)
    cmp = scheduler.compare(fc)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    hours = [t.strftime("%H") for t in cmp["timestamps"]]
    colours = {"uncontrolled": RED, "greedy": AMBER, "optimised": GREEN}
    for name, r in cmp["results"].items():
        ax.plot(hours, r.profile, label=name, lw=2, color=colours[name])
    ax.axhline(cmp["cap_kw"], color="#334155", ls="--", lw=1.2,
               label=f"grid ceiling {cmp['cap_kw']:,.0f} kW")
    ax.set_xlabel("hour")
    ax.set_ylabel("aggregate load (kW)")
    ax.set_title(f"Smart charging flattens the evening peak "
                 f"({cmp['n_vehicles']} vehicles, {cmp['total_energy_kwh']:,.0f} kWh)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG_DIR / "peak_shaving.png")
    plt.close(fig)

    print(scheduler.comparison_frame(cmp).to_string(index=False))


def fig_city_load(panel: pd.DataFrame) -> None:
    city = panel.groupby("timestamp").avg_load_kw.sum()
    week = city.loc["2025-06-09":"2025-06-16"]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.fill_between(week.index, week.to_numpy(), color=BLUE, alpha=0.25)
    ax.plot(week.index, week.to_numpy(), color=BLUE, lw=1.4)
    ax.axhline(config.GRID_SAFE_LOAD_KW, color=RED, ls="--", lw=1.2, label="safe ceiling")
    ax.set_ylabel("city EV load (kW)")
    ax.set_title("City-wide charging load, one week in June")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG_DIR / "city_load_week.png")
    plt.close(fig)


def main() -> None:
    panel = load_panel()
    print(f"panel: {len(panel):,} rows")
    fig_demand_profiles(panel)
    fig_station_heatmap(panel)
    fig_wait_vs_utilisation(panel)
    fig_city_load(panel)
    fig_forecast_vs_actual()
    fig_peak_shaving()
    print(f"\nfigures -> {FIG_DIR}")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"  {f.name}  ({f.stat().st_size/1024:.0f} kB)")


if __name__ == "__main__":
    main()

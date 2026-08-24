"""Train and evaluate the three forecasting models.

    1. arrivals      - how many EVs will plug in at this station next hour   (demand)
    2. avg_wait_min  - how long a driver arriving next hour will queue       (service)
    3. energy_kwh    - how much grid energy the station will draw next hour  (load)

Models 2 and 3 form a **cascade** on top of model 1: they take the number of
arrivals as an input, which at serving time is itself a prediction.  Training uses
the true value (so the structural relationship is learned cleanly) and the report
additionally scores the honest end-to-end path, where the demand model's output is
piped into them.

Validation is time-based: the last ``config.TEST_MONTHS`` months are held out.  A
random split would leak future information through the lag features.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from src import config
from src.features.build_features import FEATURE_SETS, load_features

# --------------------------------------------------------------------------- #
TARGET_SPECS = {
    "arrivals":     {"log": False, "clip_min": 0.0, "label": "EV arrivals / hour"},
    "avg_wait_min": {"log": True,  "clip_min": 0.0, "label": "average waiting time (min)"},
    "energy_kwh":   {"log": False, "clip_min": 0.0, "label": "station load (kWh/h)"},
}

BASELINE_COLS = {
    "arrivals": "arrivals_lag168",
    "avg_wait_min": "wait_profile",
    "energy_kwh": "energy_kwh_lag168",
}

HGB_PARAMS = dict(
    loss="squared_error",
    max_iter=500,
    learning_rate=0.06,
    max_leaf_nodes=63,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=0.12,
    n_iter_no_change=30,
    categorical_features="from_dtype",
    random_state=config.RANDOM_SEED,
)


# --------------------------------------------------------------------------- #
@dataclass
class TrainedModel:
    target: str
    model: object
    features: list[str]
    log_target: bool
    metrics: dict = field(default_factory=dict)
    importances: list = field(default_factory=list)
    trained_at: str = ""


# --------------------------------------------------------------------------- #
def time_split(df: pd.DataFrame, months: int = config.TEST_MONTHS):
    cutoff = df.timestamp.max() - pd.DateOffset(months=months)
    train = df.loc[df.timestamp <= cutoff]
    test = df.loc[df.timestamp > cutoff]
    return train, test, cutoff


def smape(y, yhat) -> float:
    denom = (np.abs(y) + np.abs(yhat)) / 2
    mask = denom > 1e-9
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / denom[mask]) * 100)


def score(y, yhat) -> dict:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    err = yhat - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "smape_pct": smape(y, yhat),
        "bias": float(np.mean(err)),
    }


def wait_bucket(minutes) -> np.ndarray:
    """Traffic-light class used by the driver app: 0 green, 1 amber, 2 red."""
    m = np.asarray(minutes, dtype=float)
    return np.where(m <= config.WAIT_GREEN_MIN, 0, np.where(m <= config.WAIT_AMBER_MIN, 1, 2))


# --------------------------------------------------------------------------- #
def train_target(df: pd.DataFrame, target: str, verbose: bool = True) -> TrainedModel:
    spec = TARGET_SPECS[target]
    feats = FEATURE_SETS[target]
    train, test, cutoff = time_split(df)

    X_tr, X_te = train[feats], test[feats]
    y_tr, y_te = train[target].to_numpy(float), test[target].to_numpy(float)
    fit_y = np.log1p(y_tr) if spec["log"] else y_tr

    t0 = time.time()
    model = HistGradientBoostingRegressor(**HGB_PARAMS)
    model.fit(X_tr, fit_y)
    fit_s = time.time() - t0

    pred = model.predict(X_te)
    if spec["log"]:
        pred = np.expm1(pred)
    pred = np.clip(pred, spec["clip_min"], None)

    metrics = {"model": score(y_te, pred)}
    base_col = BASELINE_COLS[target]
    base = test[base_col].fillna(train[target].mean()).to_numpy(float)
    metrics["baseline"] = score(y_te, base)
    metrics["baseline_name"] = base_col
    metrics["uplift_mae_pct"] = float(
        (metrics["baseline"]["mae"] - metrics["model"]["mae"]) / metrics["baseline"]["mae"] * 100)
    metrics["n_train"], metrics["n_test"] = int(len(train)), int(len(test))
    metrics["cutoff"] = str(cutoff)
    metrics["fit_seconds"] = round(fit_s, 1)
    metrics["n_iter"] = int(model.n_iter_)

    if target == "avg_wait_min":
        metrics["bucket_accuracy"] = float(np.mean(wait_bucket(y_te) == wait_bucket(pred)))
        metrics["within_5min_pct"] = float(np.mean(np.abs(pred - y_te) <= 5.0) * 100)
        busy = y_te > 0
        metrics["mae_busy_hours"] = float(np.mean(np.abs(pred[busy] - y_te[busy])))

    if verbose:
        m, b = metrics["model"], metrics["baseline"]
        print(f"\n=== {target} ({spec['label']}) ===")
        print(f"  train {len(train):,} rows  ->  test {len(test):,} rows   (cutoff {cutoff:%Y-%m-%d})")
        print(f"  model     MAE {m['mae']:7.3f}  RMSE {m['rmse']:7.3f}  R2 {m['r2']:6.3f}  sMAPE {m['smape_pct']:6.1f}%")
        print(f"  baseline  MAE {b['mae']:7.3f}  RMSE {b['rmse']:7.3f}  R2 {b['r2']:6.3f}   ({base_col})")
        print(f"  -> {metrics['uplift_mae_pct']:.1f}% lower MAE than the baseline, "
              f"{model.n_iter_} trees in {fit_s:.1f}s")
        if target == "avg_wait_min":
            print(f"  traffic-light accuracy {metrics['bucket_accuracy']:.1%}, "
                  f"within 5 min {metrics['within_5min_pct']:.1f}%")

    imp = top_importances(model, X_te, y_te, spec["log"], feats)
    return TrainedModel(target=target, model=model, features=list(feats), log_target=spec["log"],
                        metrics=metrics, importances=imp,
                        trained_at=pd.Timestamp.now().isoformat(timespec="seconds"))


def top_importances(model, X_te, y_te, log_target: bool, feats: list[str], k: int = 15) -> list:
    """Permutation importance on a test subsample (cheap but honest)."""
    n = min(4000, len(X_te))
    idx = np.random.default_rng(config.RANDOM_SEED).choice(len(X_te), n, replace=False)
    Xs = X_te.iloc[idx]
    ys = np.log1p(y_te[idx]) if log_target else y_te[idx]
    r = permutation_importance(model, Xs, ys, n_repeats=3,
                               random_state=config.RANDOM_SEED, scoring="r2")
    order = np.argsort(r.importances_mean)[::-1][:k]
    return [(feats[i], round(float(r.importances_mean[i]), 5)) for i in order]


# --------------------------------------------------------------------------- #
def evaluate_cascade(df: pd.DataFrame, models: dict[str, TrainedModel]) -> dict:
    """Score wait / energy when arrivals come from the demand model, not the truth."""
    _, test, _ = time_split(df)
    demand = models["arrivals"]
    pred_arrivals = np.clip(demand.model.predict(test[demand.features]), 0, None)

    out = {}
    for target in ("avg_wait_min", "energy_kwh"):
        tm = models[target]
        X = test[tm.features].copy()
        X["arrivals"] = pred_arrivals
        p = tm.model.predict(X)
        if tm.log_target:
            p = np.expm1(p)
        p = np.clip(p, 0, None)
        out[target] = score(test[target].to_numpy(float), p)
        if target == "avg_wait_min":
            out[target]["bucket_accuracy"] = float(
                np.mean(wait_bucket(test[target].to_numpy(float)) == wait_bucket(p)))
    return out


# --------------------------------------------------------------------------- #
def save_test_predictions(df: pd.DataFrame, models: dict[str, TrainedModel]) -> None:
    """Dump held-out predictions so the dashboard can show accuracy without refitting."""
    _, test, _ = time_split(df)
    out = test[["timestamp", "station_id", "name"]].copy()
    for t, m in models.items():
        p = m.model.predict(test[m.features])
        if m.log_target:
            p = np.expm1(p)
        out[f"{t}_true"] = test[t].to_numpy(float)
        out[f"{t}_pred"] = np.clip(p, 0, None)
    path = config.REPORTS_DIR / "test_predictions.csv.gz"
    out.to_csv(path, index=False)
    print(f"test predictions -> {path}")


# --------------------------------------------------------------------------- #
def write_report(models: dict[str, TrainedModel], cascade: dict, df: pd.DataFrame) -> None:
    import platform
    import sklearn

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "rows": int(len(df)),
        "period": [str(df.timestamp.min()), str(df.timestamp.max())],
        "models": {t: m.metrics for t, m in models.items()},
        "cascade_end_to_end": cascade,
        "top_features": {t: m.importances for t, m in models.items()},
    }
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Model report", "",
             f"Generated {payload['generated_at']} — {payload['rows']:,} station-hour rows "
             f"({df.timestamp.min():%Y-%m-%d} → {df.timestamp.max():%Y-%m-%d}).", "",
             "Validation is a **time split**: the final "
             f"{config.TEST_MONTHS} months are held out.", "",
             "## Headline accuracy (held-out test set)", "",
             "| target | MAE | RMSE | R² | sMAPE | baseline MAE | MAE uplift |",
             "|---|---|---|---|---|---|---|"]
    for t, m in models.items():
        mm, bb = m.metrics["model"], m.metrics["baseline"]
        lines.append(f"| `{t}` | {mm['mae']:.3f} | {mm['rmse']:.3f} | {mm['r2']:.3f} | "
                     f"{mm['smape_pct']:.1f}% | {bb['mae']:.3f} | "
                     f"**{m.metrics['uplift_mae_pct']:.1f}%** |")

    wm = models["avg_wait_min"].metrics
    lines += ["", "## Waiting-time model, driver's view", "",
              f"- traffic-light bucket accuracy (green ≤{config.WAIT_GREEN_MIN:.0f} / "
              f"amber ≤{config.WAIT_AMBER_MIN:.0f} / red): **{wm['bucket_accuracy']:.1%}**",
              f"- predictions within 5 minutes of truth: **{wm['within_5min_pct']:.1f}%**",
              f"- MAE restricted to hours that actually had a queue: {wm['mae_busy_hours']:.2f} min",
              "", "## End-to-end cascade (arrivals predicted, not observed)", "",
              "| target | MAE | RMSE | R² |", "|---|---|---|---|"]
    for t, s in cascade.items():
        lines.append(f"| `{t}` | {s['mae']:.3f} | {s['rmse']:.3f} | {s['r2']:.3f} |")

    lines += ["", "## Most important features (permutation importance)", ""]
    for t, m in models.items():
        lines.append(f"**{t}**")
        lines += [f"{i+1}. `{f}` — {v}" for i, (f, v) in enumerate(m.importances[:10])]
        lines.append("")
    (config.REPORTS_DIR / "model_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport -> {config.REPORTS_DIR / 'model_report.md'}")


# --------------------------------------------------------------------------- #
def train_all(save: bool = True) -> dict[str, TrainedModel]:
    df = load_features()
    models = {t: train_target(df, t) for t in TARGET_SPECS}

    print("\n=== end-to-end cascade (arrivals predicted by the demand model) ===")
    cascade = evaluate_cascade(df, models)
    for t, s in cascade.items():
        print(f"  {t:14s} MAE {s['mae']:7.3f}  RMSE {s['rmse']:7.3f}  R2 {s['r2']:6.3f}")

    save_test_predictions(df, models)

    if save:
        for t, m in models.items():
            joblib.dump({"target": m.target, "model": m.model, "features": m.features,
                         "log_target": m.log_target, "metrics": m.metrics,
                         "trained_at": m.trained_at},
                        config.MODELS_DIR / f"{t}.joblib")
        print(f"\nmodels -> {config.MODELS_DIR}")
        write_report(models, cascade, df)
    return models


if __name__ == "__main__":
    train_all()

# Model report

Generated 2026-08-24T11:42:55 — 171,840 station-hour rows (2025-01-08 → 2025-12-31).

Validation is a **time split**: the final 2 months are held out.

## Headline accuracy (held-out test set)

| target | MAE | RMSE | R² | sMAPE | baseline MAE | MAE uplift |
|---|---|---|---|---|---|---|
| `arrivals` | 0.980 | 1.505 | 0.746 | 113.4% | 1.290 | **24.0%** |
| `avg_wait_min` | 1.105 | 4.125 | 0.618 | 163.2% | 1.883 | **41.3%** |
| `energy_kwh` | 14.516 | 23.476 | 0.927 | 47.9% | 34.296 | **57.7%** |

## Waiting-time model, driver's view

- traffic-light bucket accuracy (green ≤5 / amber ≤15 / red): **93.7%**
- predictions within 5 minutes of truth: **93.0%**
- MAE restricted to hours that actually had a queue: 7.73 min

## End-to-end cascade (arrivals predicted, not observed)

| target | MAE | RMSE | R² |
|---|---|---|---|
| `avg_wait_min` | 1.314 | 4.693 | 0.506 |
| `energy_kwh` | 18.553 | 29.181 | 0.888 |

## Most important features (permutation importance)

**arrivals**
1. `arrivals_profile` — 0.66397
2. `arrivals_roll3` — 0.03067
3. `expected_pressure` — 0.02611
4. `arrivals_roll24` — 0.02123
5. `hour_cos` — 0.00732
6. `arrivals_lag2` — 0.00451
7. `city_arrivals_lag24` — 0.00426
8. `arrivals_lag168` — 0.00417
9. `arrivals_lag1` — 0.00369
10. `utilisation_roll24` — 0.00363

**avg_wait_min**
1. `arrivals` — 0.70763
2. `avg_wait_min_lag1` — 0.1698
3. `chargers_per_arrival_lag1` — 0.14818
4. `utilisation_lag1` — 0.06311
5. `avg_session_min_lag1` — 0.05926
6. `n_chargers` — 0.04333
7. `capacity_kw` — 0.02406
8. `headroom_lag1` — 0.0077
9. `expected_pressure` — 0.00469
10. `utilisation_roll3` — 0.00232

**energy_kwh**
1. `arrivals` — 0.26698
2. `arrivals_lag1` — 0.14329
3. `utilisation_lag1` — 0.05964
4. `arrivals_roll3` — 0.05036
5. `avg_session_min_lag1` — 0.02866
6. `n_chargers` — 0.02761
7. `energy_kwh_lag1` — 0.01873
8. `capacity_kw` — 0.00658
9. `district` — 0.00284
10. `energy_profile` — 0.00269

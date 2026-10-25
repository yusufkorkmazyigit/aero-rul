# aero-rul

Predict how many flights a turbofan engine has left, **with an interval whose coverage is
measured on engines the model has never seen**.

aero-rul works on NASA's N-CMAPSS dataset, which simulates full flights (climb, cruise,
descent) of a fleet of engines until failure. It is a small, tested Python package with one
command per step, and every number it reports can be reproduced from a config file.

## What it does

| Step | Command | Output |
| --- | --- | --- |
| Read the raw 1 Hz recordings, one row per flight | `aerorul prepare` | `cycles.parquet` |
| Ridge and LightGBM baselines, leave-one-engine-out CV | `aerorul evaluate` | `baselines.md` |
| Prediction intervals and their measured coverage | `aerorul uncertainty` | `intervals.md`, band plots |
| Which sensors the model relies on | `aerorul explain` | `explain.md`, SHAP plot |
| ABC vs random search vs TPE, same budget | `aerorul benchmark` | `benchmark.md`, convergence plot |

## Design choices

- **An engine is either in training or in evaluation, never both.** Consecutive flights
  of one engine are nearly identical; splitting by rows would let the model memorise.
- **Uncertainty is measured, not assumed.** Conformal intervals come with a guarantee
  under an exchangeability assumption; we check whether it holds on the test engines.
- **Comparisons are fair or not made.** Optimisers get the same budget, space and CV;
  every result is a mean over seeds.
- **No label leakage.** The simulator's hidden health parameters and health-state flag,
  and the cycle counter, are never model inputs.

See [Method](method.md) for the reasoning behind each choice and [Results](results.md) for
the numbers.

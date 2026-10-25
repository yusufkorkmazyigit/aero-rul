# Usage

## Install

```bash
pip install aerorul
```

or, for development:

```bash
git clone https://github.com/yusufkorkmazyigit/aero-rul
cd aero-rul
pip install -e ".[dev]"
```

## Data

Download the *Turbofan Engine Degradation Simulation Data Set-2* (N-CMAPSS) from the NASA
Prognostics Data Repository and place `N-CMAPSS_DS02-006.h5` in `data/raw/`. Only this one
file is needed. DS02 holds nine engines: units 2, 5, 10, 16, 18 and 20 for development and
units 11, 14 and 15 for testing.

## Run everything

```bash
aerorul prepare        # 1 Hz recordings -> one row per flight cycle
aerorul evaluate       # baselines, leave-one-engine-out CV + test score
aerorul uncertainty    # prediction intervals and coverage
aerorul explain        # SHAP importance per signal
aerorul benchmark      # ABC vs random search vs TPE (slowest step)
```

Results are written to `results/`. Each command reads `configs/default.yaml`; use
`--config my.yaml` to run another experiment.

## Configuration

```yaml
seed: 42
data:
  raw_path: data/raw/N-CMAPSS_DS02-006.h5
  processed_dir: data/processed
features:
  groups: [W, X_s]              # flight conditions + physical sensors
  stats: [mean, std, min, max]  # per-flight summaries of each signal
  rul_cap: null                 # set a number for a piecewise-linear target
models:
  names: [ridge, lgbm]
  params: {}                    # e.g. {lgbm: {num_leaves: 15}}
evaluation:
  seeds: [0, 1, 2, 3, 4]
  results_dir: results
uncertainty:
  model: lgbm
  methods: [split, cv, quantile]
  alphas: [0.2, 0.1]            # 80 % and 90 % intervals
  calibration_engines: 2
tuning:
  optimizers: [abc, random, tpe]
  budget: 60
  abc_pop_size: 10
```

Unknown keys are rejected, so a typo fails immediately instead of being ignored.

## Tests

```bash
pytest
```

The tests build a small synthetic file with the N-CMAPSS layout, so they need no download.

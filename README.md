# aero-rul

Remaining useful life (RUL) prediction for turbofan engines on NASA's
[N-CMAPSS](https://www.mdpi.com/2306-5729/6/1/5) dataset, with calibrated uncertainty intervals.

> Work in progress. Results, figures and full documentation arrive with v0.1.0.

## Why

Airlines and engine makers plan maintenance around how many flights an engine has left.
A point estimate is not enough for that decision: a planner needs to know how far the
prediction can be trusted. aero-rul predicts RUL **with an interval whose coverage is
measured**, evaluates every model on engines it has never seen, and is reproducible from
one config file.

## Install

```bash
git clone https://github.com/yusufkorkmazyigit/aero-rul
cd aero-rul
pip install -e ".[dev]"
```

## Data

Download `N-CMAPSS_DS02-006.h5` (Turbofan Engine Degradation Simulation Data Set-2) from
the NASA Prognostics Data Repository and place it at `data/raw/`. DS02 has nine engines:
units 2, 5, 10, 16, 18, 20 for training and 11, 14, 15 for testing.

## Usage

```bash
aerorul inspect                 # list the arrays in the raw file
aerorul prepare                 # 1 Hz recordings -> one row per flight cycle
```

Every command reads `configs/default.yaml`; pass `--config` to use another.

## Development

```bash
pytest          # runs on a small synthetic file, no download needed
ruff check .
```

## License

MIT. N-CMAPSS is released by NASA under CC0.

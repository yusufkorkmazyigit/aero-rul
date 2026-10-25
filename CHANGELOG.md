# Changelog

## [0.1.0] - unreleased

### Added
- `aerorul inspect` and `aerorul prepare`: read N-CMAPSS HDF5 one engine at a time and
  summarise each flight cycle into one row.
- `aerorul train` and `aerorul evaluate`: Ridge and LightGBM baselines, leave-one-engine-out
  cross-validation, RMSE, MAE and the NASA scoring function, mean and std over seeds.
- `aerorul uncertainty`: split conformal, cross-conformal and quantile intervals, with
  coverage measured overall and per test engine.
- `aerorul benchmark`: ABC (abcopt), random search and Optuna TPE under the same budget,
  search space and cross-validation.
- `aerorul explain`: SHAP importance per signal from LightGBM's TreeSHAP.
- Tests on a synthetic file with the N-CMAPSS layout; CI on Python 3.10 to 3.12.

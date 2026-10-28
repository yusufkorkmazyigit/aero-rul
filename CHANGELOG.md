# Changelog

## [0.2.0] - 2026-10-28

### Added
- Normalised conformal intervals (`norm_pred`, `norm_feat`): leave-one-engine-out errors
  scaled by a difficulty estimate, itself fitted without the scored engine.
- Results by phase of life: on DS02 the adaptive interval is narrower near failure but
  covers less there, so cross-conformal stays the default.
- `aerorul decide`: maintenance decision study. Each policy books a shop visit a lead
  time ahead from the point estimate, a tuned margin or an interval's lower edge;
  failures, wasted life and cost are measured with every engine held out in turn.

## [0.1.0] - 2026-10-25

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
- History features: trailing mean and slope over each engine's last flights.
- Conformalized quantile regression (CQR) as a fourth interval method.
- Per-flight NASA score, comparable across sets of different size; progress bars.
- Results on N-CMAPSS DS02 in the README and the documentation.
- Tests on a synthetic file with the N-CMAPSS layout; CI on Python 3.10 to 3.12.

# Results

N-CMAPSS DS02: trained on engines 2, 5, 10, 16, 18 and 20 (446 flights), tested on engines
11, 14 and 15 (202 flights). Every number is a mean ± standard deviation over five seeds.
Each table is produced by the command named above it.

## Point prediction

`aerorul evaluate`

| Model | Test RMSE | Test MAE | NASA score per flight | CV RMSE |
| --- | ---: | ---: | ---: | ---: |
| **LightGBM** | **10.05 ± 0.43** | **7.88 ± 0.39** | **1.66 ± 0.16** | 10.96 ± 0.09 |
| Ridge | 12.75 | 9.93 | 2.48 | 17.06 |

Test RMSE per engine (LightGBM): engine 11: 9.00, engine 14: 12.87, engine 15: 6.71.

### Effect of history features

| LightGBM | Without history | With history (10 flights) |
| --- | ---: | ---: |
| Test RMSE | 10.59 ± 0.25 | **10.05 ± 0.43** |
| CV RMSE | 11.52 ± 0.05 | **10.96 ± 0.09** |
| CV NASA score per flight | 1.91 ± 0.02 | **1.59 ± 0.03** |
| 90 % interval width (cross-conformal) | 37.2 | **34.1** |

The same features hurt Ridge (CV RMSE 12.91 to 17.06): 146 correlated features on 446
flights overwhelm a lightly regularised linear model.

## Prediction intervals

`aerorul uncertainty`

![Predicted RUL and 90 % cross-conformal interval for the test engines](assets/bands_cv_90.png)

| Target | Method | Coverage | Width (flights) |
| ---: | --- | ---: | ---: |
| 90 % | **Cross-conformal** | **90.4 % ± 1.5** | **34.1** |
| 90 % | Split conformal | 90.0 % ± 4.0 | 35.8 |
| 90 % | CQR | 98.5 % ± 0.9 | 50.5 |
| 90 % | Quantile regression | 82.2 % ± 0.9 | 36.8 |
| 80 % | **Cross-conformal** | **78.6 % ± 2.1** | **25.0** |
| 80 % | Split conformal | 75.0 % ± 4.2 | 25.8 |
| 80 % | CQR | 94.7 % ± 0.8 | 37.5 |
| 80 % | Quantile regression | 75.0 % ± 1.9 | 28.0 |

Coverage per test engine at the 90 % target:

| Method | Engine 11 | Engine 14 | Engine 15 |
| --- | ---: | ---: | ---: |
| Cross-conformal | 96.6 % | 80.3 % | 96.4 % |
| Split conformal | 92.5 % | 83.7 % | 94.9 % |
| CQR | 95.9 % | 99.5 % | 99.7 % |
| Quantile regression | 65.8 % | 88.9 % | 89.0 % |

!!! note "Reading the per-engine table"
    Conformal coverage is a guarantee on average over exchangeable points, not for each
    engine. Engine 14, which also has the largest point error, falls short of 90 %. Pooled
    coverage alone would hide it.

## What the model looks at

`aerorul explain`

![Mean absolute SHAP value per signal on the test engines](assets/shap_signals.png)

| Signal | Mean \|SHAP\| (flights of RUL) |
| --- | ---: |
| T50, LPT outlet temperature | 16.5 |
| Nc, core speed | 5.6 |
| T48, HPT outlet temperature | 3.3 |
| P15, bypass duct pressure | 2.0 |
| Nf, fan speed | 1.9 |

Fourteen of the fifteen most important features are ten-flight trailing means.

## Hyperparameter search benchmark

`aerorul benchmark`

![Best CV RMSE against candidate evaluations](assets/convergence.png)

| Optimiser | Best CV RMSE | Seeds won | Test RMSE of chosen config | Time per search |
| --- | ---: | ---: | ---: | ---: |
| TPE (Optuna) | **10.20 ± 0.09** | 4 / 5 | 11.04 ± 0.56 | 135 s |
| Random search | 10.31 ± 0.15 | 1 / 5 | 11.14 ± 0.94 | 56 s |
| ABC (abcopt) | 10.56 ± 0.17 | 0 / 5 | 11.12 ± 1.05 | 61 s |
| *Default parameters* | *10.96 ± 0.09* | | ***10.05 ± 0.43*** | |

Budget 60 evaluations, seven hyperparameters, leave-one-engine-out CV on the training
engines. TPE finds the lowest validation error, but every tuned configuration does worse
on the test engines than the defaults: with six training engines the search fits the
validation engines. The released configuration therefore keeps LightGBM's defaults.

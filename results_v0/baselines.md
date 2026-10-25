# Baselines

Seeds: [0, 1, 2, 3, 4]. RMSE and MAE in flight cycles; NASA score lower is better.

| stage   | model   |   rmse_mean |   rmse_std |   mae_mean |   mae_std |   nasa_mean |   nasa_std |   nasa_per_cycle_mean |   nasa_per_cycle_std |
|:--------|:--------|------------:|-----------:|-----------:|----------:|------------:|-----------:|----------------------:|---------------------:|
| cv      | lgbm    |       11.52 |       0.05 |       8.34 |      0.06 |      851.71 |      10.93 |                  1.91 |                 0.02 |
| cv      | ridge   |       12.91 |       0    |      10.61 |      0    |      954.55 |       0    |                  2.14 |                 0    |
| test    | lgbm    |       10.59 |       0.25 |       8.12 |      0.25 |      344.73 |      13.63 |                  1.71 |                 0.07 |
| test    | ridge   |       12.39 |       0    |      10.67 |      0    |      474.22 |       0    |                  2.35 |                 0    |

## Test RMSE per engine

| model   |    11 |    14 |    15 |
|:--------|------:|------:|------:|
| lgbm    | 10.12 | 12.36 |  8.63 |
| ridge   | 11.84 | 14.14 | 10.61 |

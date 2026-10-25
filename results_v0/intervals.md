# Prediction intervals (lgbm)

Coverage = share of test cycles whose true RUL falls inside the interval; it should be close to the target. Width in flight cycles; narrower is better at the same coverage.

|   target | method   |   coverage_mean |   coverage_std |   width_mean |   width_std |
|---------:|:---------|----------------:|---------------:|-------------:|------------:|
|      0.9 | cv       |           0.937 |          0.011 |       37.231 |       0.669 |
|      0.9 | quantile |           0.718 |          0.008 |       30.587 |       1.021 |
|      0.9 | split    |           0.904 |          0.034 |       36.901 |       3.17  |
|      0.8 | cv       |           0.802 |          0.022 |       26.619 |       0.547 |
|      0.8 | quantile |           0.586 |          0.013 |       21.855 |       0.539 |
|      0.8 | split    |           0.785 |          0.018 |       26.773 |       1.968 |

## Coverage per test engine

|                   |    11 |    14 |    15 |
|:------------------|------:|------:|------:|
| (0.8, 'cv')       | 0.803 | 0.705 | 0.91  |
| (0.8, 'quantile') | 0.546 | 0.608 | 0.597 |
| (0.8, 'split')    | 0.793 | 0.703 | 0.872 |
| (0.9, 'cv')       | 0.932 | 0.911 | 0.97  |
| (0.9, 'quantile') | 0.627 | 0.766 | 0.743 |
| (0.9, 'split')    | 0.895 | 0.861 | 0.961 |

## Figures

- bands_split_90.png
- bands_cv_90.png
- bands_quantile_90.png

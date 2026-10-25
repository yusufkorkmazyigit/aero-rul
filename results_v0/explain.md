# What drives the prediction

Mean |SHAP| on test engines, in cycles of RUL. Correlated sensors share credit; this explains the model, not the physics.

## Per signal

|           |   mean_abs_shap |
|:----------|----------------:|
| T50       |          16.743 |
| Nc        |           7.752 |
| T48       |           4.397 |
| Ps30      |           2.647 |
| Nf        |           2.072 |
| P40       |           2.059 |
| P50       |           2.046 |
| T30       |           1.566 |
| TRA       |           1.267 |
| T24       |           1.145 |
| Wf        |           1.134 |
| P24       |           1.057 |
| P15       |           0.931 |
| Mach      |           0.904 |
| alt       |           0.428 |
| T2        |           0.376 |
| P2        |           0.238 |
| P21       |           0.215 |
| n_seconds |           0.127 |
| Fc        |           0     |

## Top 15 features

|           |   mean_abs_shap |
|:----------|----------------:|
| T50_max   |          12.304 |
| Nc_max    |           4.796 |
| T50_mean  |           3.57  |
| T48_mean  |           2.71  |
| Nc_mean   |           1.845 |
| P40_max   |           1.425 |
| Ps30_max  |           1.373 |
| P50_max   |           1.157 |
| Nc_min    |           0.964 |
| Nf_max    |           0.957 |
| T48_min   |           0.826 |
| Ps30_mean |           0.763 |
| T48_max   |           0.701 |
| T24_max   |           0.63  |
| Nf_min    |           0.601 |

Figure: shap_signals.png

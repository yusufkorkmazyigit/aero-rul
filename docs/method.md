# Method

## Data

N-CMAPSS (Arias Chao et al., 2021) simulates a fleet of turbofan engines flying until
failure under real flight conditions. Each subset is an HDF5 file with a development split
and a test split. Every row is one second of flight:

| Group | Content |
| --- | --- |
| `W` | flight conditions: altitude, Mach number, throttle resolver angle, fan inlet temperature |
| `X_s` | 14 physical sensors: temperatures, pressures, shaft speeds, fuel flow |
| `X_v` | virtual sensors estimated by the engine model |
| `T` | health parameters, the simulator's hidden degradation state |
| `A` | engine id, flight cycle, flight class, health state |
| `Y` | remaining useful life in flight cycles (the label) |

We use DS02, the subset most often used for data-driven prognostics: six development
engines (2, 5, 10, 16, 18, 20) and three test engines (11, 14, 15).

## From seconds to flights

Degradation moves from one flight to the next, not from one second to the next. Each
flight cycle becomes one row: for every signal in `W` and `X_s` we take the mean, standard
deviation, minimum and maximum over the flight, plus the flight class and its length. This
removes sensor noise and turns millions of rows into a few hundred per engine.

The file is read one engine at a time, so memory use stays at the size of one engine.

## What the model may not see

| Column | Why it is excluded |
| --- | --- |
| `unit` | an identifier, not a property of the engine |
| `cycle` | with a fixed end of life, the flight count nearly gives away the answer |
| `hs` | derived from the simulator's hidden degradation |
| `T` group | the hidden degradation itself |

## Evaluation

Consecutive flights of one engine are nearly identical. A random split of rows would put
neighbours of every test row into training and report an error the model will never reach
on a new engine. We therefore:

1. cross-validate with **leave-one-engine-out** on the six development engines, and
2. report the final score on the three test engines, which no training step touches.

Every number is a mean and standard deviation over five seeds.

### Metrics

RMSE and MAE are in flight cycles. The NASA scoring function (Saxena et al., 2008) penalises
late predictions harder, because predicting more remaining life than there is moves
maintenance past the failure. With \(d = \hat{y} - y\):

$$
s = \sum_i \begin{cases} e^{-d_i/13} - 1 & d_i < 0 \\ e^{d_i/10} - 1 & d_i \ge 0 \end{cases}
$$

Ten cycles late costs about 1.72; ten cycles early about 1.16. Lower is better.

## Prediction intervals

A planner needs a range and a statement of how often such ranges are right. We build
intervals three ways and **measure coverage on the test engines**.

**Split conformal.** Hold out two development engines for calibration and train on the
other four. With absolute calibration errors \(r_1, \dots, r_n\), the radius is the
\(\lceil (n+1)(1-\alpha) \rceil\)-th smallest error, and the interval is
\(\hat{y} \pm q\). If calibration and test points are exchangeable, coverage is at least
\(1-\alpha\) for any model.

**Cross-conformal.** Take the errors from leave-one-engine-out predictions, so every error
comes from a model that never saw that engine, and train the final model on all development
engines. It uses the data better; its formal guarantee is weaker, which is one more reason
to measure coverage directly.

**Quantile regression.** LightGBM fit to the \(\alpha/2\) and \(1-\alpha/2\) quantiles.
Widths adapt to each input, but nothing guarantees coverage.

!!! note "Exchangeability"
    Test engines 14 and 15 fly a different mix of flights from the training engines. If
    coverage falls short on them, that is evidence of distribution shift, and we report it
    as a finding rather than tune it away.

## Hyperparameter search benchmark

We compare three optimisers for LightGBM: the canonical Artificial Bee Colony algorithm
(Karaboga, 2005) through [abcopt](https://github.com/yusufkorkmazyigit/abc-optimizer-lib),
random search, and Optuna's Tree-structured Parzen Estimator. Everything except the
optimiser is fixed: the same seven-parameter search space, the same budget of candidate
evaluations, the same leave-one-engine-out objective, five seeds.

abcopt counts every proposal against the budget, including a configuration it has already
tried; repeats are served from a cache and cost no compute. If anything this is strict on
ABC.

The chosen configuration is refit on all development engines and scored once on the test
engines. A better CV score that does not carry over to the test engines means the search
overfit the validation engines.

## Explanations

LightGBM computes exact TreeSHAP values. We report mean |SHAP| on the test engines, summed
per physical signal. SHAP explains the model, not the engine: correlated sensors share
credit, and a high value means the model relies on a signal, not that the signal causes
degradation.

## Limitations

- Three test engines are few; results are reported per engine as well as pooled.
- N-CMAPSS is a high-fidelity simulation, not fleet data.
- Interval width is constant for the conformal methods; adaptive (normalised) conformal
  scores are a natural next step.

## References

- M. Arias Chao, C. Kulkarni, K. Goebel, O. Fink. *Aircraft Engine Run-to-Failure Dataset
  under Real Flight Conditions for Prognostics and Diagnostics.* Data 6(1):5, 2021.
- A. Saxena, K. Goebel, D. Simon, N. Eklund. *Damage Propagation Modeling for Aircraft
  Engine Run-to-Failure Simulation.* PHM 2008.
- D. Karaboga. *An Idea Based on Honey Bee Swarm for Numerical Optimization.* Technical
  Report TR06, Erciyes University, 2005.
- V. Vovk, A. Gammerman, G. Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.
- S. M. Lundberg et al. *From local explanations to global understanding with explainable
  AI for trees.* Nature Machine Intelligence 2:56–67, 2020.

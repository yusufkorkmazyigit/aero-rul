# aero-rul

[![CI](https://github.com/yusufkorkmazyigit/aero-rul/actions/workflows/ci.yml/badge.svg)](https://github.com/yusufkorkmazyigit/aero-rul/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://yusufkorkmazyigit.github.io/aero-rul/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**How many flights does this engine have left, and how far can that answer be trusted?**

aero-rul predicts the remaining useful life (RUL) of turbofan engines on NASA's
[N-CMAPSS](https://www.mdpi.com/2306-5729/6/1/5) dataset and attaches a prediction interval
to every estimate. The interval's coverage is not assumed: it is **measured on engines the
model has never seen**. Then the question a planner actually asks is answered: when should
this engine go to the shop, and how many failures and wasted flights does that rule cost?

> **Result (DS02, three held-out engines, five seeds).** A 90 % conformal interval contains
> the true RUL in **90.4 %** of test flights. Plain quantile regression, asked for the same
> 90 %, delivers **82.2 %** with wider intervals. Point error: **10.1 flights RMSE**.
>
> **Decision (all nine engines held out in turn).** Booking a shop visit 10 flights ahead
> on the lower edge of a normalised conformal interval removed every engine before failure,
> wasting 10.5 flights of life per engine. Booking on the point estimate let 38 % of
> engines fail.

![Predicted RUL and 90 % conformal interval for the three test engines](https://raw.githubusercontent.com/yusufkorkmazyigit/aero-rul/main/docs/assets/bands_cv_90.png)

## Why this project

Airlines plan engine maintenance around the flights an engine has left. Too early wastes
a serviceable engine; too late risks an in-service failure. A point estimate is not enough
to plan with. A planner needs a range, and a statement of how often such ranges are right.

The project is built around four rules:

- **An engine is either in training or in evaluation, never both.** Consecutive flights
  of one engine are nearly identical; a row-level split lets the model memorise.
- **Uncertainty is measured, not assumed.** Coverage is reported pooled and per engine.
- **No label leakage.** The simulator's hidden health parameters, its health-state flag
  and the flight counter are never inputs.
- **Comparisons are fair or not made.** Same budget, same validation, several seeds.

## Results

All numbers: N-CMAPSS DS02, trained on engines 2, 5, 10, 16, 18, 20, tested on 11, 14, 15,
mean ± std over five seeds. Reproduce with the commands under [Usage](#usage).

### Point prediction

| Model | Test RMSE | Test MAE | NASA score per flight | CV RMSE |
| --- | ---: | ---: | ---: | ---: |
| **LightGBM** | **10.05 ± 0.43** | **7.88 ± 0.39** | **1.66 ± 0.16** | 10.96 ± 0.09 |
| Ridge | 12.75 | 9.93 | 2.48 | 17.06 |

RMSE and MAE in flights. The NASA score penalises late predictions (more life predicted
than remains) harder than early ones; lower is better. CV is leave-one-engine-out on the
six training engines.

**History features help.** Adding the trailing mean and slope over each engine's last ten
flights lowered LightGBM's test RMSE from 10.59 ± 0.25 to 10.05 ± 0.43 and its CV NASA
score per flight from 1.91 to 1.59. Ridge got worse with the same features (CV RMSE 12.91
to 17.06): 146 correlated features on 446 flights overwhelm a lightly regularised linear
model (alpha = 1), while the tree ensemble absorbs them.

### Prediction intervals

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

- **Conformal calibration works.** Cross-conformal lands on its 90 % target with the
  narrowest intervals. Uncalibrated quantile regression promises 90 % and delivers 82 %.
- **Coverage is marginal, not per engine.** At 90 %, cross-conformal covers 96.6 % of
  engine 11, 96.4 % of engine 15, but only **80.3 % of engine 14**, the engine with the
  largest point error. Conformal guarantees hold on average over engines; a planner should
  know which engine is the outlier.
- **CQR over-covers here.** Conformalized quantile regression is valid (coverage at least
  the target) but asked for 90 % it covers 98.5 %, and at both levels its intervals are
  about 50 % wider than cross-conformal's.
  Its calibration scores come from held-out training engines, which are harder than the
  test engines; with six engines that gap dominates.

### Adaptive intervals: tried, measured, not shipped

A conformal interval of constant width is too wide near failure and too narrow early in
life. Normalised conformal prediction (Lei et al., 2018) divides each calibration error by
a difficulty estimate sigma(x) and returns prediction +- q * sigma(x). Two versions were
tested, both with sigma fitted leave-one-engine-out so that no score uses a sigma that saw
its engine.

| 90 % target | Coverage | Width | Width / coverage at RUL <= 20 | at RUL > 50 | Engine 14 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Cross-conformal** (constant) | **90.4 %** | **34.1** | **30.8 / 81.9 %** | 35.6 / 93.1 % | 80.3 % |
| Normalised, sigma linear in prediction | 90.1 % | 33.2 | 21.9 / 74.0 % | 42.8 / 96.7 % | 77.9 % |
| Normalised, sigma from features (LightGBM) | 99.8 % | 56.9 | 40.4 / 99.4 % | 68.1 / 100 % | 100 % |

- **The linear version does what it promises on average:** target coverage, slightly
  narrower intervals, about 22 flights wide near failure instead of 31.
- **It pays for that where it matters most.** Below 20 flights of remaining life,
  coverage falls from 82 % to 74 %. That is where the maintenance decision is taken.
- **It does not fix engine 14.** Engine 14's shortfall is distribution shift, not a badly
  shaped interval.
- **The feature-based version over-covers**, like CQR: a difficulty model learned on five
  engines misjudges the sixth, calibration scores inflate, intervals grow by two thirds.
- **So the default interval stays cross-conformal.** Both variants remain available
  (`norm_pred`, `norm_feat`). Coverage is not the whole story, though: the next section
  asks which interval leads to the best maintenance decision, and the answer differs.

### From interval to maintenance decision

A planner books a shop visit some flights ahead. The rule tested here: after every
flight, book a removal 10 flights ahead as soon as a statistic of the prediction is at
most 10. If the engine had less than 10 flights left at booking, it **fails** before its
slot; otherwise the life left at removal is **wasted**. Cost per engine = waste + 100 x
failure, i.e. one unplanned failure is worth 100 flights of thrown-away life.

Every engine of DS02 (nine) is held out in turn; models, intervals and the tuned margin
come from the other eight only. Five seeds.

| Policy (book when ... <= 10) | Engines failing | Wasted flights per engine | Cost per engine |
| --- | ---: | ---: | ---: |
| Point estimate | 37.8 % | 1.7 | 39.5 |
| Point estimate - tuned margin (about 10) | 11.1 % | 9.6 | 20.7 |
| Lower edge, quantile regression | 26.7 % | 1.3 | 28.0 |
| Lower edge, cross-conformal | **0 %** | 17.9 | 17.9 |
| Lower edge, CQR | **0 %** | 10.7 | 10.7 |
| **Lower edge, normalised conformal** | **0 %** | **10.5** | **10.5** |

![Wasted life against failure rate for each policy](https://raw.githubusercontent.com/yusufkorkmazyigit/aero-rul/main/docs/assets/decision_frontier.png)

- **A point estimate is not a plan.** Booked on the raw prediction, more than a third of
  the engines fail before their slot: half the errors are optimistic, and an optimistic
  error is a failure.
- **Tuning a safety margin is not enough either.** A margin chosen to minimise cost on
  the training engines still loses engine 14 under every seed. The margin is fitted to
  engines that behave alike; engine 14 does not.
- **Calibrated lower edges lose no engine.** The normalised interval and CQR do it for
  about the waste of the best margin in hindsight (grey curve). Cross-conformal is just
  as safe but its constant margin is wider than needed: 7 extra flights per engine.
- **Coverage and decisions disagree.** The normalised interval lost two-sided coverage
  in the last 20 flights, but the decision is taken around 20 flights before failure and
  depends on the lower edge only. With a 5-flight lead time it does fail engines (8.9 %);
  cross-conformal never does. Which interval to use depends on the lead time.
- **Read with care.** Nine engines give nine decisions per seed; "0 %" means none of nine
  failed, not that the risk is zero. The cost ratio and lead time are swept in the
  [results](https://yusufkorkmazyigit.github.io/aero-rul/results/).

### What the model looks at

![Mean absolute SHAP value per signal on the test engines](https://raw.githubusercontent.com/yusufkorkmazyigit/aero-rul/main/docs/assets/shap_signals.png)

The exhaust-side temperature **T50** (low-pressure turbine outlet) dominates, followed by
core speed **Nc** and **T48** (high-pressure turbine outlet). Rising turbine temperature for
the same thrust is the classic sign of engine wear in practice (EGT margin). Fourteen of
the fifteen most important features are ten-flight averages: the model reads trends, not
single flights. SHAP explains the model, not the physics; correlated sensors share credit.

### Hyperparameter search: ABC vs random search vs TPE

Three optimisers tune LightGBM's seven main hyperparameters under identical conditions:
60 candidate evaluations each, the same search space, leave-one-engine-out CV on the
training engines, five seeds. ABC is the canonical Artificial Bee Colony algorithm
(Karaboga) from my own package [abcopt](https://github.com/yusufkorkmazyigit/abc-optimizer-lib).

| Optimiser | Best CV RMSE | Seeds won | Test RMSE of chosen config | Time per search |
| --- | ---: | ---: | ---: | ---: |
| TPE (Optuna) | **10.20 ± 0.09** | 4 / 5 | 11.04 ± 0.56 | 135 s |
| Random search | 10.31 ± 0.15 | 1 / 5 | 11.14 ± 0.94 | 56 s |
| ABC (abcopt) | 10.56 ± 0.17 | 0 / 5 | 11.12 ± 1.05 | 61 s |
| *Default parameters, no search* | *10.96 ± 0.09* | | ***10.05 ± 0.43*** | |

![Best CV RMSE against candidate evaluations, mean and range over five seeds](https://raw.githubusercontent.com/yusufkorkmazyigit/aero-rul/main/docs/assets/convergence.png)

- **On the search objective, TPE wins** and ABC comes last. With 60 evaluations and ten
  food sources ABC completes only a few employed/onlooker cycles; it is reported as is.
- **None of it carries over to new engines.** Every optimiser lowered CV RMSE below the
  defaults, and every tuned model is *worse* on the test engines than the untuned one
  (about 11.1 vs 10.05). With six training engines, the search fits the validation
  engines rather than engine wear. The test differences between optimisers are well
  inside one standard deviation.
- **So the shipped model uses default parameters.** On data this small, a better
  validation score is not evidence of a better model.

## Install

```bash
pip install aerorul
```

or for development:

```bash
git clone https://github.com/yusufkorkmazyigit/aero-rul
cd aero-rul
pip install -e ".[dev]"
```

## Data

Download *Turbofan Engine Degradation Simulation Data Set-2* from the NASA Prognostics Data
Repository and place `N-CMAPSS_DS02-006.h5` (2.4 GB) in `data/raw/`. Only this file is used.

## Usage

```bash
aerorul prepare        # 6.5 M one-second rows -> 648 flights (one row each)
aerorul evaluate       # baselines, leave-one-engine-out CV and test scores
aerorul uncertainty    # prediction intervals and their coverage
aerorul decide         # maintenance policies: failures and wasted life
aerorul explain        # SHAP importance per signal
aerorul benchmark      # ABC vs random search vs TPE, about 15 minutes on 16 threads
```

Results go to `results/` as Markdown tables, CSV files and figures. Every command reads
`configs/default.yaml`; `--config other.yaml` runs another experiment.

## How it works

1. **Per-flight features.** Each signal (4 flight conditions, 14 sensors) is summarised
   over every flight by its mean, standard deviation, minimum and maximum.
2. **History features.** For each engine, the trailing mean and slope of those summaries
   over its last ten flights. Only past flights of the same engine are used.
3. **Models.** Ridge and LightGBM, evaluated with leave-one-engine-out CV and on the test
   engines.
4. **Intervals.** Split conformal, cross-conformal, quantile regression, CQR and
   normalised conformal, with coverage checked pooled, per engine and by phase of life.
5. **Decisions.** Each interval's lower edge books a shop visit; failures and wasted life
   are counted with every engine held out in turn.

The [method page](https://yusufkorkmazyigit.github.io/aero-rul/method/) explains each
choice, with formulas and references.

## Limitations

- Three test engines are few. Results are given per engine as well as pooled.
- N-CMAPSS is a high-fidelity simulation, not fleet data.
- Conformal intervals have constant width. Normalised conformal scores narrow them near
  end of life but lose coverage exactly there on DS02 (see above); more engines, or a
  sigma model conditioned on life phase, are the next things to try.

## Development

```bash
pytest          # 41 tests on a synthetic file with the N-CMAPSS layout; no download
ruff check .
```

## Citation

See [CITATION.cff](CITATION.cff). The dataset: M. Arias Chao, C. Kulkarni, K. Goebel,
O. Fink, *Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for
Prognostics and Diagnostics*, Data 6(1):5, 2021.

## License

MIT. N-CMAPSS is released by NASA under CC0.

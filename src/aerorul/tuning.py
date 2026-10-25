"""A fair hyperparameter-search benchmark: ABC vs random search vs TPE.

Comparisons between optimisers are easy to rig by accident. This one fixes everything
except the optimiser:

* the same LightGBM model and the same search space,
* the same budget: every optimiser gets ``budget`` candidate evaluations,
* the same objective: RMSE under leave-one-engine-out CV on the dev engines,
* several seeds, reported as mean +- std, plus how often each optimiser wins.

The search only ever sees dev engines. The chosen configuration is then refit on all dev
engines and scored once on the test engines, which is the number that matters: a lower
CV score that does not carry over to new engines is overfitting the search.

A note on ABC's budget: abcopt counts every proposal, including a configuration it has
already tried (integer rounding makes repeats common). Repeats cost no compute because
they are cached, but they still use budget, so if anything this setup is strict on ABC.

Optimisers:
``abc``     canonical Artificial Bee Colony (Karaboga), via the ``abcopt`` package
``random``  scikit-learn ``RandomizedSearchCV``
``tpe``     Optuna's Tree-structured Parzen Estimator
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV, cross_val_score
from tqdm.auto import tqdm

from .evaluation import rmse, split_xy
from .features import feature_columns
from .models import make_model

# One search space, written in abcopt's format and translated for the others.
SPACE: dict[str, dict] = {
    "learning_rate": {"type": "float", "range": (0.01, 0.3), "log": True},
    "n_estimators": {"type": "int", "range": (100, 800)},
    "num_leaves": {"type": "int", "range": (8, 128)},
    "min_child_samples": {"type": "int", "range": (5, 100)},
    "subsample": {"type": "float", "range": (0.5, 1.0)},
    "colsample_bytree": {"type": "float", "range": (0.5, 1.0)},
    "reg_lambda": {"type": "float", "range": (1e-3, 10.0), "log": True},
}
SCORING = "neg_root_mean_squared_error"


def _scipy_distributions(space: dict) -> dict:
    out = {}
    for name, spec in space.items():
        low, high = spec["range"]
        if spec["type"] == "int":
            out[name] = randint(low, high + 1)
        elif spec.get("log"):
            out[name] = loguniform(low, high)
        else:
            out[name] = uniform(low, high - low)
    return out


def _optuna_suggest(trial, space: dict) -> dict:
    out = {}
    for name, spec in space.items():
        low, high = spec["range"]
        if spec["type"] == "int":
            out[name] = trial.suggest_int(name, low, high)
        else:
            out[name] = trial.suggest_float(name, low, high, log=bool(spec.get("log")))
    return out


def search(
    optimizer: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    budget: int,
    seed: int,
    space: dict | None = None,
    abc_pop_size: int = 10,
) -> dict:
    """Run one search. Returns best params, best CV RMSE, per-evaluation RMSE and time."""
    space = space or SPACE
    base = make_model("lgbm", seed=seed)
    cv = LeaveOneGroupOut()
    start = time.perf_counter()

    if optimizer == "abc":
        from abcopt import ABCSearchCV

        s = ABCSearchCV(base, space, cv=cv, scoring=SCORING, pop_size=abc_pop_size,
                        max_evals=budget, random_state=seed, refit=False)  # fmt: skip
        s.fit(X, y, groups=groups)
        trace = [-r["score"] for r in s.results_]
        best = s.best_params_
    elif optimizer == "random":
        s = RandomizedSearchCV(base, _scipy_distributions(space), n_iter=budget, cv=cv,
                               scoring=SCORING, random_state=seed, refit=False)  # fmt: skip
        s.fit(X, y, groups=groups)
        trace = list(-np.asarray(s.cv_results_["mean_test_score"], float))
        best = s.best_params_
    elif optimizer == "tpe":
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        splits = list(cv.split(X, y, groups))

        def objective(trial):
            model = base.set_params(**_optuna_suggest(trial, space))
            return -cross_val_score(model, X, y, cv=splits, scoring=SCORING).mean()

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))  # fmt: skip
        study.optimize(objective, n_trials=budget)
        trace = [t.value for t in study.trials]
        best = study.best_params
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")

    trace = np.asarray(trace, float)
    return {
        "best_params": dict(best),
        "cv_rmse": float(np.nanmin(trace)),
        "trace": trace,
        "seconds": time.perf_counter() - start,
    }


def run_benchmark(
    table: pd.DataFrame,
    optimizers: Sequence[str],
    budget: int,
    seeds: Sequence[int],
    abc_pop_size: int = 10,
    space: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every optimizer x seed. Returns (results, traces)."""
    dev = table[table["split"] == "dev"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)
    X, y, groups = split_xy(dev)
    X_test, y_test = test[feature_columns(test)], test["RUL"].to_numpy(float)

    results, traces = [], []
    runs = list(product(optimizers, seeds))
    bar = tqdm(runs, desc="benchmark", unit="search")
    for opt, seed in bar:
        bar.set_postfix(optimizer=opt, seed=seed)
        out = search(opt, X, y, groups, budget, seed, space, abc_pop_size)
        model = make_model("lgbm", out["best_params"], seed).fit(X, y)
        results.append(
            {
                "optimizer": opt,
                "seed": seed,
                "cv_rmse": out["cv_rmse"],
                "test_rmse": rmse(y_test, model.predict(X_test)),
                "seconds": out["seconds"],
                "best_params": out["best_params"],
            }
        )
        best_so_far = np.fmin.accumulate(out["trace"])
        traces.append(
            pd.DataFrame(
                {
                    "optimizer": opt,
                    "seed": seed,
                    "evaluation": np.arange(1, len(best_so_far) + 1),
                    "best_cv_rmse": best_so_far,
                }
            )
        )
    return pd.DataFrame(results), pd.concat(traces, ignore_index=True)


def summarize_benchmark(results: pd.DataFrame) -> pd.DataFrame:
    """Mean +- std per optimizer, and in how many seeds it had the lowest CV RMSE."""
    agg = results.groupby("optimizer")[["cv_rmse", "test_rmse", "seconds"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    winners = results.loc[results.groupby("seed")["cv_rmse"].idxmin(), "optimizer"]
    agg["cv_wins"] = winners.value_counts().reindex(agg.index, fill_value=0)
    return agg.reset_index().sort_values("cv_rmse_mean")


def plot_convergence(traces: pd.DataFrame, path) -> None:
    """Best CV RMSE so far against evaluations used: mean over seeds, band = min..max."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"abc": "#E45756", "random": "#9D9D9D", "tpe": "#4C78A8"}
    labels = {"abc": "ABC (abcopt)", "random": "Random search", "tpe": "TPE (Optuna)"}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for opt, g in traces.groupby("optimizer"):
        stats = g.groupby("evaluation")["best_cv_rmse"].agg(["mean", "min", "max"])
        c = colors.get(opt)
        ax.fill_between(stats.index, stats["min"], stats["max"], color=c, alpha=0.15, lw=0)
        ax.plot(stats.index, stats["mean"], color=c, lw=2, label=labels.get(opt, opt))
    ax.set_xlabel("Candidate evaluations")
    ax.set_ylabel("Best CV RMSE so far (cycles)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

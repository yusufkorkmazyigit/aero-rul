"""Baseline regressors for RUL.

Two deliberately different baselines:

* ``ridge``: a linear model on standardised features. If it is close to the boosted
  model, the signal is mostly linear and the complex model is not earning its keep.
* ``lgbm``: gradient-boosted trees, a strong default for tabular data.

Both are plain scikit-learn estimators, so everything downstream (cross-validation,
conformal intervals, hyperparameter search) works on either without special cases.
"""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULTS: dict[str, dict[str, Any]] = {
    "ridge": {"alpha": 1.0},
    "lgbm": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
    },
}


def make_model(
    name: str, params: dict[str, Any] | None = None, seed: int = 0
) -> RegressorMixin | Pipeline:
    """Build an unfitted model; ``params`` override the defaults above."""
    if name not in DEFAULTS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(DEFAULTS)}")
    kwargs = {**DEFAULTS[name], **(params or {})}
    if name == "ridge":
        # Ridge has no randomness; its seed-to-seed spread is zero by construction.
        return make_pipeline(StandardScaler(), Ridge(**kwargs))
    return LGBMRegressor(random_state=seed, verbose=-1, n_jobs=-1, **kwargs)

"""Honest evaluation.

The rule that matters most: **an engine is either in training or in evaluation, never
both.** Consecutive cycles of one engine are nearly identical, so a random row split lets
the model memorise each engine and reports an error far lower than it will ever see on a
new engine. We therefore

* cross-validate with leave-one-engine-out on the dev engines, and
* report the final score on the test engines, which no step of training has touched.

Metrics:

* RMSE and MAE, in flight cycles.
* NASA score (Saxena et al., PHM 2008): an exponential penalty that punishes a late
  prediction (predicted RUL above the truth, so maintenance comes after the failure)
  harder than an early one. Lower is better.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from tqdm.auto import tqdm

from .features import feature_columns
from .models import make_model


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(d**2)))


def mae(y_true, y_pred) -> float:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.mean(np.abs(d)))


def nasa_score(y_true, y_pred, early: float = 13.0, late: float = 10.0) -> float:
    """Sum of exp(|d|/a) - 1 with d = predicted - true; a = 13 if early, 10 if late."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    a = np.where(d < 0, early, late)
    return float(np.sum(np.expm1(np.abs(d) / a)))


def all_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa": nasa_score(y_true, y_pred),
        # The NASA score is a sum, so it grows with the number of cycles scored; the
        # per-cycle mean is what can be compared across sets of different size.
        "nasa_per_cycle": nasa_score(y_true, y_pred) / max(len(np.atleast_1d(y_true)), 1),
    }


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Features, label and engine ids of a cycle table."""
    return frame[feature_columns(frame)], frame["RUL"].to_numpy(float), frame["unit"].to_numpy()


def cross_validate(
    dev: pd.DataFrame, name: str, params: dict | None = None, seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray]:
    """Leave-one-engine-out CV. Returns per-fold metrics and out-of-fold predictions."""
    X, y, units = split_xy(dev)
    oof = np.full(len(y), np.nan)
    rows = []
    for train_idx, val_idx in LeaveOneGroupOut().split(X, y, units):
        model = make_model(name, params, seed).fit(X.iloc[train_idx], y[train_idx])
        oof[val_idx] = model.predict(X.iloc[val_idx])
        rows.append({"held_out": int(units[val_idx][0]), **all_metrics(y[val_idx], oof[val_idx])})
    return pd.DataFrame(rows), oof


def fit_and_test(
    dev: pd.DataFrame, test: pd.DataFrame, name: str, params: dict | None = None, seed: int = 0
):
    """Fit on all dev engines, predict the test engines."""
    X, y, _ = split_xy(dev)
    model = make_model(name, params, seed).fit(X, y)
    return model, model.predict(test[feature_columns(test)])


def run_baselines(
    table: pd.DataFrame,
    names: Sequence[str],
    seeds: Sequence[int],
    params: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every model x every seed: CV score on dev and final score on test.

    Returns (scores, predictions). ``scores`` has one row per model, seed and stage;
    ``predictions`` holds every test-cycle prediction for plots and later analysis.
    """
    params = params or {}
    dev = table[table["split"] == "dev"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)
    _, y_test, _ = split_xy(test)

    scores, preds = [], []
    for name, seed in tqdm(list(product(names, seeds)), desc="baselines", unit="run"):
        _, oof = cross_validate(dev, name, params.get(name), seed)
        scores.append({"model": name, "seed": seed, "stage": "cv",
                       **all_metrics(dev["RUL"], oof)})  # fmt: skip
        _, y_hat = fit_and_test(dev, test, name, params.get(name), seed)
        scores.append({"model": name, "seed": seed, "stage": "test",
                       **all_metrics(y_test, y_hat)})  # fmt: skip
        preds.append(test[["unit", "cycle", "RUL"]].assign(model=name, seed=seed, pred=y_hat))
    return pd.DataFrame(scores), pd.concat(preds, ignore_index=True)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds, one row per model and stage."""
    metrics = ["rmse", "mae", "nasa", "nasa_per_cycle"]
    agg = scores.groupby(["stage", "model"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index().sort_values(["stage", "rmse_mean"])


def per_unit(predictions: pd.DataFrame) -> pd.DataFrame:
    """Test RMSE for each engine, averaged over seeds; shows if one engine drives the result."""
    by = predictions.groupby(["model", "seed", "unit"])
    per = by.apply(lambda g: rmse(g["RUL"], g["pred"]), include_groups=False)
    per = per.rename("rmse").reset_index()
    return per.pivot_table(index="model", columns="unit", values="rmse", aggfunc="mean").round(2)

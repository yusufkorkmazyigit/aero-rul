"""Prediction intervals for RUL, and how often they are right.

A maintenance planner cannot act on "about 40 flights left". They can act on "between
28 and 52 flights, and intervals like this one contain the truth 90 % of the time". The
second half of that sentence is the point of this module: we **measure** coverage on
engines the model never saw, instead of trusting the method's promise.

Four methods:

``split`` (split conformal prediction)
    Set some dev engines aside for calibration and train on the rest. The absolute
    errors on the calibration engines give a radius q: the ceil((n+1)(1-alpha))-th
    smallest error. The interval is prediction +- q. If calibration and test cycles are
    exchangeable, coverage is at least 1 - alpha, whatever the model. Cost: fewer
    engines to train on.

``cv`` (cross-conformal style)
    Use every dev engine for both jobs: the errors come from leave-one-engine-out
    predictions, so each error is made by a model that never saw that engine. The final
    model is trained on all dev engines. Uses the data better; the formal guarantee is
    slightly weaker, which is why we check coverage empirically.

``quantile`` (LightGBM quantile regression)
    Fit two models for the alpha/2 and 1 - alpha/2 quantiles. Widths adapt to each
    input, but there is no coverage guarantee at all; a useful contrast.

``cqr`` (conformalized quantile regression, Romano et al., 2019)
    The quantile models' interval, widened or narrowed by a conformal correction. The
    score of a cycle is how far its true RUL lies outside [lo, hi] (negative if inside);
    the conformal quantile q of those scores gives [lo - q, hi + q]. Widths still adapt
    to each input, and coverage regains the conformal guarantee. Scores come from
    leave-one-engine-out predictions, as in ``cv``.

Exchangeability is the assumption to watch. N-CMAPSS test engines 14 and 15 fly a
different mix of flights than the training engines; if coverage drops there, that is a
finding about distribution shift, not a bug.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import LeaveOneGroupOut
from tqdm.auto import tqdm

from .evaluation import cross_validate, split_xy
from .features import feature_columns
from .models import DEFAULTS, make_model


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest score; inf if there are too few scores."""
    scores = np.sort(np.asarray(scores, float))
    n = scores.size
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(scores[k - 1])


def _interval_frame(test: pd.DataFrame, pred, lo, hi) -> pd.DataFrame:
    # RUL cannot be negative; clipping at zero only removes impossible values
    # (and can only improve coverage and width, never hurt them).
    def nonneg(a):
        return np.maximum(np.asarray(a, float), 0.0)

    return test[["unit", "cycle", "RUL"]].assign(pred=nonneg(pred), lo=nonneg(lo), hi=nonneg(hi))


def split_conformal(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    name: str,
    alpha: float,
    params: dict | None = None,
    seed: int = 0,
    n_calibration: int = 2,
) -> pd.DataFrame:
    units = np.sort(dev["unit"].unique())
    if n_calibration >= units.size:
        raise ValueError("need at least one training engine besides the calibration engines")
    calib_units = np.random.default_rng(seed).choice(units, n_calibration, replace=False)
    is_cal = dev["unit"].isin(calib_units)
    X_tr, y_tr, _ = split_xy(dev[~is_cal])
    X_cal, y_cal, _ = split_xy(dev[is_cal])
    model = make_model(name, params, seed).fit(X_tr, y_tr)
    q = conformal_quantile(np.abs(y_cal - model.predict(X_cal)), alpha)
    pred = model.predict(test[feature_columns(test)])
    return _interval_frame(test, pred, pred - q, pred + q)


def cv_conformal(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    name: str,
    alpha: float,
    params: dict | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    _, oof = cross_validate(dev, name, params, seed)
    q = conformal_quantile(np.abs(dev["RUL"].to_numpy(float) - oof), alpha)
    X, y, _ = split_xy(dev)
    pred = make_model(name, params, seed).fit(X, y).predict(test[feature_columns(test)])
    return _interval_frame(test, pred, pred - q, pred + q)


def _quantile_predict(X, y, X_new, alpha: float, params: dict | None, seed: int):
    """Fit LightGBM at alpha/2, 0.5 and 1 - alpha/2; return (lo, mid, hi) on X_new."""
    kw = {**DEFAULTS["lgbm"], **(params or {}), "random_state": seed, "verbose": -1}
    edges = [
        LGBMRegressor(objective="quantile", alpha=q, **kw).fit(X, y).predict(X_new)
        for q in (alpha / 2, 0.5, 1 - alpha / 2)
    ]
    # Separately fitted quantiles can cross; sort them so lo <= mid <= hi.
    lo, mid, hi = np.sort(np.vstack(edges), axis=0)
    return lo, mid, hi


def quantile_intervals(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    alpha: float,
    params: dict | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    X, y, _ = split_xy(dev)
    lo, mid, hi = _quantile_predict(X, y, test[feature_columns(test)], alpha, params, seed)
    return _interval_frame(test, mid, lo, hi)


def cqr_intervals(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    alpha: float,
    params: dict | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    X, y, units = split_xy(dev)
    scores = np.empty(len(y))
    for train_idx, val_idx in LeaveOneGroupOut().split(X, y, units):
        lo, _, hi = _quantile_predict(
            X.iloc[train_idx], y[train_idx], X.iloc[val_idx], alpha, params, seed
        )
        scores[val_idx] = np.maximum(lo - y[val_idx], y[val_idx] - hi)
    q = conformal_quantile(scores, alpha)
    lo, mid, hi = _quantile_predict(X, y, test[feature_columns(test)], alpha, params, seed)
    lo, hi = lo - q, hi + q
    # A negative q narrows the interval; never let it turn inside out.
    centre = (lo + hi) / 2
    lo, hi = np.minimum(lo, centre), np.maximum(hi, centre)
    return _interval_frame(test, mid, lo, hi)


def interval_metrics(y, lo, hi) -> dict[str, float]:
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return {"coverage": float(np.mean((y >= lo) & (y <= hi))), "width": float(np.mean(hi - lo))}


def run_uncertainty(
    table: pd.DataFrame,
    name: str,
    methods: Sequence[str],
    alphas: Sequence[float],
    seeds: Sequence[int],
    params: dict | None = None,
    n_calibration: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every method x alpha x seed on the test engines. Returns (scores, intervals)."""
    dev = table[table["split"] == "dev"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)
    scores, frames = [], []
    runs = list(product(methods, alphas, seeds))
    for method, alpha, seed in tqdm(runs, desc="intervals", unit="run"):
        if method == "split":
            iv = split_conformal(dev, test, name, alpha, params, seed, n_calibration)
        elif method == "cv":
            iv = cv_conformal(dev, test, name, alpha, params, seed)
        elif method == "quantile":
            iv = quantile_intervals(dev, test, alpha, params, seed)
        elif method == "cqr":
            iv = cqr_intervals(dev, test, alpha, params, seed)
        else:
            raise ValueError(f"unknown method {method!r}")
        iv = iv.assign(method=method, alpha=alpha, target=1 - alpha, seed=seed)
        frames.append(iv)
        scores.append({"method": method, "target": 1 - alpha, "seed": seed,
                       **interval_metrics(iv["RUL"], iv["lo"], iv["hi"])})  # fmt: skip
    return pd.DataFrame(scores), pd.concat(frames, ignore_index=True)


def summarize_intervals(scores: pd.DataFrame) -> pd.DataFrame:
    agg = scores.groupby(["target", "method"])[["coverage", "width"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index().sort_values(["target", "method"], ascending=[False, True])


def coverage_per_unit(intervals: pd.DataFrame) -> pd.DataFrame:
    """Coverage for each test engine; where an engine falls short, look for shift."""
    hit = (intervals["RUL"] >= intervals["lo"]) & (intervals["RUL"] <= intervals["hi"])
    cov = intervals.assign(hit=hit).groupby(["target", "method", "unit"])["hit"].mean()
    return cov.reset_index().pivot_table(
        index=["target", "method"], columns="unit", values="hit", aggfunc="mean"
    )


def plot_bands(intervals: pd.DataFrame, method: str, alpha: float, path) -> None:
    """True RUL, prediction and interval per test engine (first seed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = intervals[(intervals["method"] == method) & np.isclose(intervals["alpha"], alpha)]
    sel = sel[sel["seed"] == sel["seed"].min()]
    units = sorted(sel["unit"].unique())
    fig, axes = plt.subplots(1, len(units), figsize=(4.2 * len(units), 3.4), squeeze=False)
    for ax, unit in zip(axes[0], units, strict=True):
        g = sel[sel["unit"] == unit].sort_values("cycle")
        ax.fill_between(g["cycle"], g["lo"], g["hi"], color="#4C78A8", alpha=0.25,
                        label=f"{1 - alpha:.0%} interval")  # fmt: skip
        ax.plot(g["cycle"], g["pred"], color="#4C78A8", lw=1.5, label="predicted")
        ax.plot(g["cycle"], g["RUL"], color="#222222", lw=1.5, ls="--", label="true")
        ax.set_title(f"Engine {unit}")
        ax.set_xlabel("Flight cycle")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Remaining useful life (cycles)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

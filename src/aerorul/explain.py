"""Which signals drive the prediction?

SHAP values split each prediction into one contribution per feature, so that the
contributions plus a base value add up to the prediction. LightGBM computes exact
TreeSHAP values itself (``pred_contrib=True``), so no extra dependency is needed.

We report mean |SHAP| on the test engines, per feature and summed per physical signal
(``T48_mean``, ``T48_std``, ... all count towards ``T48``). Read it as "how much the
model leans on this signal", not "this signal causes degradation": correlated sensors
share credit, and SHAP explains the model, not the engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import split_xy
from .features import feature_columns
from .models import make_model

STAT_SUFFIXES = ("_mean", "_std", "_min", "_max", "_median", "_first", "_last")


def signal_of(feature: str) -> str:
    for suffix in STAT_SUFFIXES:
        if feature.endswith(suffix):
            return feature[: -len(suffix)]
    return feature


def shap_importance(
    table: pd.DataFrame, params: dict | None = None, seed: int = 0
) -> tuple[pd.Series, pd.Series]:
    """Mean |SHAP| on test engines, per feature and per signal (both sorted, largest first)."""
    dev = table[table["split"] == "dev"]
    test = table[table["split"] == "test"]
    X, y, _ = split_xy(dev)
    model = make_model("lgbm", params, seed).fit(X, y)
    contrib = model.predict(test[feature_columns(test)], pred_contrib=True)
    # last column is the base value, not a feature
    per_feature = pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=X.columns)
    per_signal = per_feature.groupby(per_feature.index.map(signal_of)).sum()
    return (
        per_feature.sort_values(ascending=False),
        per_signal.sort_values(ascending=False),
    )


def plot_importance(per_signal: pd.Series, path, top: int = 15) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = per_signal.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 0.32 * len(s) + 1.0))
    ax.barh(s.index, s.values, color="#4C78A8")
    ax.set_xlabel("Mean |SHAP| on test engines (cycles of RUL)")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

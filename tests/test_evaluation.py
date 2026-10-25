import math

import numpy as np
import pandas as pd
import pytest

from aerorul.evaluation import (
    cross_validate,
    mae,
    nasa_score,
    per_unit,
    rmse,
    run_baselines,
    summarize,
)
from aerorul.models import make_model


def test_point_metrics():
    assert rmse([0, 0], [3, 4]) == pytest.approx(math.sqrt(12.5))
    assert mae([0, 0], [3, -4]) == pytest.approx(3.5)


def test_nasa_score_punishes_late_more_than_early():
    late = nasa_score([50], [60])  # predicted 10 cycles too many
    early = nasa_score([50], [40])  # predicted 10 cycles too few
    assert late == pytest.approx(math.e - 1)
    assert early == pytest.approx(math.exp(10 / 13) - 1)
    assert late > early
    assert nasa_score([5, 5], [5, 5]) == 0.0


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        make_model("xgb")


def _toy_table(n_units=4, n_cycles=30, seed=0) -> pd.DataFrame:
    """Cycle table with a feature that tracks degradation, plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, unit in enumerate(range(1, n_units + 2)):
        split = "test" if i == n_units else "dev"
        for cycle in range(1, n_cycles + 1):
            rul = n_cycles - cycle
            rows.append({"split": split, "unit": unit, "cycle": cycle, "Fc": 1,
                         "s_mean": -rul + rng.normal(scale=2), "noise_mean": rng.normal(),
                         "RUL": float(rul)})  # fmt: skip
    return pd.DataFrame(rows)


def test_cross_validation_holds_out_whole_engines():
    dev = _toy_table().query("split == 'dev'").reset_index(drop=True)
    folds, oof = cross_validate(dev, "ridge")
    assert sorted(folds["held_out"]) == sorted(dev["unit"].unique())
    assert not np.isnan(oof).any()


def test_baselines_learn_the_signal_and_cover_every_seed():
    scores, preds = run_baselines(_toy_table(), ["ridge", "lgbm"], seeds=[0, 1])
    assert len(scores) == 2 * 2 * 2  # models x seeds x stages
    summary = summarize(scores)
    test_ridge = summary.query("stage == 'test' and model == 'ridge'")
    assert test_ridge["rmse_mean"].item() < 4  # signal is linear, noise sd 2
    assert test_ridge["rmse_std"].item() == pytest.approx(0)  # ridge is deterministic
    assert set(preds["model"]) == {"ridge", "lgbm"}
    assert list(per_unit(preds).columns) == [5]


def test_nasa_per_cycle_is_comparable_across_sizes():
    from aerorul.evaluation import all_metrics

    small = all_metrics([50, 50], [60, 60])
    large = all_metrics([50] * 10, [60] * 10)
    assert large["nasa"] == pytest.approx(5 * small["nasa"])
    assert large["nasa_per_cycle"] == pytest.approx(small["nasa_per_cycle"])

import math

import numpy as np
import pandas as pd
import pytest

from aerorul.uncertainty import (
    conformal_quantile,
    coverage_per_unit,
    interval_metrics,
    quantile_intervals,
    run_uncertainty,
    split_conformal,
    summarize_intervals,
)


def test_conformal_quantile_rank():
    scores = np.arange(1, 10)  # n = 9
    # alpha 0.1: k = ceil(10 * 0.9) = 9 -> 9th smallest
    assert conformal_quantile(scores, 0.1) == 9
    # alpha 0.5: k = ceil(10 * 0.5) = 5
    assert conformal_quantile(scores, 0.5) == 5
    # too few scores for alpha 0.05: k = 10 > 9 -> infinite radius, honest not optimistic
    assert math.isinf(conformal_quantile(scores, 0.05))


def test_interval_metrics():
    m = interval_metrics([1, 5, 9], [0, 0, 0], [2, 4, 10])
    assert m["coverage"] == pytest.approx(2 / 3)
    assert m["width"] == pytest.approx(16 / 3)


def _table(n_dev=6, n_test=3, n_cycles=60, seed=0) -> pd.DataFrame:
    """Engines with the same noise level everywhere: conformal should hit its target."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_dev + n_test + 1):
        split = "dev" if unit <= n_dev else "test"
        for cycle in range(1, n_cycles + 1):
            rul = float(n_cycles - cycle)
            rows.append({"split": split, "unit": unit, "cycle": cycle, "Fc": 1,
                         "s_mean": rul + rng.normal(scale=4), "RUL": rul})  # fmt: skip
    return pd.DataFrame(rows)


def test_conformal_methods_reach_target_coverage_without_shift():
    scores, _ = run_uncertainty(_table(), "ridge", ["split", "cv"], alphas=[0.1], seeds=[0, 1, 2])
    summary = summarize_intervals(scores)
    for cov in summary["coverage_mean"]:
        assert cov == pytest.approx(0.9, abs=0.06)


def test_lower_edge_never_negative_and_ordered():
    t = _table()
    dev, test = t[t["split"] == "dev"], t[t["split"] == "test"]
    iv = split_conformal(dev, test, "ridge", alpha=0.1)
    assert (iv["lo"] >= 0).all()
    assert (iv["lo"] <= iv["hi"]).all()
    q = quantile_intervals(dev, test, alpha=0.2, params={"n_estimators": 50})
    assert (q["lo"] <= q["pred"]).all() and (q["pred"] <= q["hi"]).all()


def test_split_needs_a_training_engine():
    t = _table(n_dev=2)
    with pytest.raises(ValueError, match="training engine"):
        split_conformal(t[t["split"] == "dev"], t[t["split"] == "test"], "ridge", 0.1,
                        n_calibration=2)  # fmt: skip


def test_coverage_per_unit_has_every_test_engine():
    _, iv = run_uncertainty(_table(), "ridge", ["cv"], alphas=[0.2], seeds=[0])
    assert list(coverage_per_unit(iv).columns) == [7, 8, 9]


def test_cqr_reaches_target_and_stays_ordered():
    from aerorul.uncertainty import cqr_intervals

    t = _table()
    dev, test = t[t["split"] == "dev"], t[t["split"] == "test"]
    iv = cqr_intervals(dev, test, alpha=0.1, params={"n_estimators": 60})
    assert (iv["lo"] <= iv["hi"]).all()
    assert interval_metrics(iv["RUL"], iv["lo"], iv["hi"])["coverage"] == pytest.approx(
        0.9, abs=0.08
    )


def _hetero_table(n_dev=6, n_test=3, n_cycles=80, seed=0) -> pd.DataFrame:
    """Noise grows with RUL, like real engines: easy near failure, hard early on."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_dev + n_test + 1):
        split = "dev" if unit <= n_dev else "test"
        for cycle in range(1, n_cycles + 1):
            rul = float(n_cycles - cycle)
            rows.append({"split": split, "unit": unit, "cycle": cycle, "Fc": 1,
                         "s_mean": rul + rng.normal(scale=1 + 0.15 * rul), "RUL": rul})  # fmt: skip
    return pd.DataFrame(rows)


@pytest.mark.parametrize("method", ["norm_pred", "norm_feat"])
def test_normalized_conformal_reaches_target(method):
    scores, iv = run_uncertainty(_hetero_table(), "ridge", [method], alphas=[0.1], seeds=[0, 1])
    assert summarize_intervals(scores)["coverage_mean"].iloc[0] == pytest.approx(0.9, abs=0.07)
    assert (iv["lo"] <= iv["pred"]).all() and (iv["pred"] <= iv["hi"]).all()


def test_normalized_intervals_adapt_where_constant_ones_cannot():
    scores, iv = run_uncertainty(_hetero_table(), "ridge", ["cv", "norm_pred"], alphas=[0.1],
                                 seeds=[0])  # fmt: skip
    width = scores.set_index("method")["width"]
    assert width["norm_pred"] < width["cv"]
    # Near failure the adaptive interval is much tighter than early in life.
    norm = iv[iv["method"] == "norm_pred"]
    late, early = norm[norm["RUL"] < 15], norm[norm["RUL"] > 60]
    assert (late["hi"] - late["lo"]).mean() < 0.5 * (early["hi"] - early["lo"]).mean()


def test_unknown_difficulty_is_rejected():
    from aerorul.uncertainty import normalized_conformal

    t = _table()
    with pytest.raises(ValueError, match="difficulty"):
        normalized_conformal(t[t["split"] == "dev"], t[t["split"] == "test"], "ridge", 0.1,
                             difficulty="magic")  # fmt: skip

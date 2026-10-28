import numpy as np
import pandas as pd
import pytest

from aerorul.decision import (
    cost,
    decide,
    frontier,
    policy_outcomes,
    run_decision_study,
    summarize_decisions,
    tune_margin,
)


def test_decide_on_time_wastes_the_remainder():
    # RUL 20..0; statistic = truth: books when RUL hits 10, removed exactly at end of life
    rul = np.arange(20, -1, -1.0)
    o = decide(np.arange(21), rul, rul, lead=10)
    assert o["failed"] == 0 and o["wasted"] == 0 and o["rul_at_trigger"] == 10


def test_decide_optimistic_statistic_fails():
    rul = np.arange(20, -1, -1.0)
    o = decide(np.arange(21), rul, rul + 5, lead=10)  # predicts 5 flights too many
    assert o["failed"] == 1 and o["wasted"] == 0


def test_decide_pessimistic_statistic_wastes():
    rul = np.arange(20, -1, -1.0)
    o = decide(np.arange(21), rul, rul - 4, lead=10)
    assert o["failed"] == 0 and o["wasted"] == 4


def test_decide_never_triggered_is_a_failure():
    o = decide([1, 2, 3], [2.0, 1.0, 0.0], [50.0, 50.0, 50.0], lead=5)
    assert o["failed"] == 1 and np.isnan(o["trigger_cycle"])


def test_decide_sorts_by_cycle():
    rul = np.arange(20, -1, -1.0)
    cyc = np.arange(21)
    perm = np.random.default_rng(0).permutation(21)
    assert decide(cyc[perm], rul[perm], rul[perm] - 4, 10) == decide(cyc, rul, rul - 4, 10)


def _engines(bias: float, n: int = 4, life: int = 40) -> pd.DataFrame:
    rows = []
    for u in range(n):
        for c in range(life):
            r = float(life - 1 - c)
            rows.append({"unit": u, "cycle": c, "RUL": r, "pred": r + bias})
    return pd.DataFrame(rows)


def test_tune_margin_cancels_a_known_bias():
    frame = _engines(bias=7.0)
    assert tune_margin(frame, lead=5, ratio=100, margins=np.arange(0, 20)) == 7


def test_cost_counts_failures_by_ratio():
    o = policy_outcomes(_engines(bias=3.0).assign(stat=lambda d: d["pred"]), "stat", 5)
    assert o["failed"].all() and cost(o, ratio=50) == 50


def test_frontier_is_monotone_in_margin():
    preds = _engines(bias=2.0).assign(seed=0)
    preds["pred"] += np.random.default_rng(1).normal(scale=3, size=len(preds))
    f = frontier(preds, lead=5, margins=np.arange(0, 15))
    assert (np.diff(f["wasted_mean"]) >= -1e-9).sum() >= len(f) - 3  # waste grows with margin
    assert f["failure_rate"].iloc[-1] <= f["failure_rate"].iloc[0]


def _table(n=7, life=50, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n + 1):
        for c in range(1, life + 1):
            r = float(life - c)
            rows.append({"split": "dev", "unit": u, "cycle": c, "Fc": 1,
                         "s_mean": r + rng.normal(scale=3), "RUL": r})  # fmt: skip
    return pd.DataFrame(rows)


def test_study_runs_every_policy_and_ratio():
    out, preds = run_decision_study(
        _table(), "ridge", alpha=0.1, seeds=[0], lead=5, ratios=[20, 100],
        policies=["point", "point_margin", "cv", "norm_pred"], max_margin=20,
    )  # fmt: skip
    assert len(out) == 7 * 4 * 2
    assert set(preds.columns) >= {"pred", "lo_cv", "lo_norm_pred"}
    s = summarize_decisions(out)
    # A calibrated lower bound is more cautious than the raw point estimate.
    at = s[s["ratio"] == 100].set_index("policy")
    assert at.loc["cv", "failure_rate"] <= at.loc["point", "failure_rate"]


def test_study_rejects_unknown_policy():
    with pytest.raises(ValueError, match="unknown policies"):
        run_decision_study(_table(), "ridge", 0.1, [0], 5, [20], policies=["magic"])

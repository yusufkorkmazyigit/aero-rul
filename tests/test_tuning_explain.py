import numpy as np
import pandas as pd
import pytest

from aerorul.explain import shap_importance, signal_of
from aerorul.tuning import (
    SPACE,
    _scipy_distributions,
    plot_convergence,
    run_benchmark,
    search,
    summarize_benchmark,
)

SMALL_SPACE = {
    "learning_rate": {"type": "float", "range": (0.05, 0.3), "log": True},
    "n_estimators": {"type": "int", "range": (20, 60)},
    "num_leaves": {"type": "int", "range": (4, 16)},
}


def _table(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, 6):
        split = "dev" if unit <= 4 else "test"
        for cycle in range(1, 41):
            rul = float(40 - cycle)
            rows.append({"split": split, "unit": unit, "cycle": cycle, "Fc": 1,
                         "T48_mean": rul + rng.normal(scale=3), "T48_std": rng.normal(),
                         "Nf_mean": rng.normal(), "RUL": rul})  # fmt: skip
    return pd.DataFrame(rows)


def test_spaces_translate_to_scipy():
    dists = _scipy_distributions(SPACE)
    assert set(dists) == set(SPACE)
    draws = dists["num_leaves"].rvs(size=200, random_state=0)
    assert draws.min() >= 8 and draws.max() <= 128


@pytest.mark.parametrize("optimizer", ["abc", "random", "tpe"])
def test_every_optimizer_uses_exactly_the_budget(optimizer):
    t = _table().query("split == 'dev'")
    X = t[["Fc", "T48_mean", "T48_std", "Nf_mean"]]
    out = search(optimizer, X, t["RUL"].to_numpy(), t["unit"].to_numpy(),
                 budget=6, seed=0, space=SMALL_SPACE, abc_pop_size=2)  # fmt: skip
    assert len(out["trace"]) == 6
    assert out["cv_rmse"] == pytest.approx(np.nanmin(out["trace"]))
    assert set(out["best_params"]) == set(SMALL_SPACE)


def test_unknown_optimizer_rejected():
    t = _table().query("split == 'dev'")
    with pytest.raises(ValueError, match="unknown optimizer"):
        search("grid", t[["T48_mean"]], t["RUL"].to_numpy(), t["unit"].to_numpy(), 2, 0)


def test_benchmark_traces_never_get_worse(tmp_path):
    results, traces = run_benchmark(_table(), ["abc", "random"], budget=4, seeds=[0, 1],
                                    abc_pop_size=2, space=SMALL_SPACE)  # fmt: skip
    assert len(results) == 4
    for _, g in traces.groupby(["optimizer", "seed"]):
        assert np.all(np.diff(g["best_cv_rmse"].to_numpy()) <= 1e-12)
    summary = summarize_benchmark(results)
    assert summary["cv_wins"].sum() == 2  # one winner per seed
    plot_convergence(traces, tmp_path / "c.png")
    assert (tmp_path / "c.png").exists()


def test_signal_of_strips_statistic():
    assert signal_of("T48_mean") == "T48"
    assert signal_of("X_s_std") == "X_s"
    assert signal_of("n_seconds") == "n_seconds"


def test_shap_finds_the_informative_signal():
    per_feature, per_signal = shap_importance(_table(), {"n_estimators": 50})
    assert per_signal.index[0] == "T48"
    assert per_feature.index[0] == "T48_mean"

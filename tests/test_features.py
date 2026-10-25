import numpy as np
import pandas as pd
import pytest

from aerorul.config import Config, DataConfig, FeatureConfig
from aerorul.features import build_split, cycle_features, feature_columns, prepare

from .conftest import DEV_UNITS, TEST_UNITS


def _toy() -> pd.DataFrame:
    # unit 1: cycle 1 has 3 seconds, cycle 2 has 1 second
    return pd.DataFrame(
        {
            "unit": [1, 1, 1, 1],
            "cycle": [1, 1, 1, 2],
            "Fc": [3, 3, 3, 3],
            "hs": [1, 1, 0, 0],
            "s": [1.0, 2.0, 3.0, 10.0],
            "RUL": [1, 1, 1, 0],
        }
    )


def test_cycle_features_aggregates_per_cycle():
    out = cycle_features(_toy(), ["s"], ["mean", "std", "max"])
    assert list(out["cycle"]) == [1, 2]
    assert out.loc[0, "s_mean"] == pytest.approx(2.0)
    assert out.loc[0, "s_max"] == 3.0
    assert out.loc[1, "s_std"] == 0.0  # one sample: no spread, not NaN
    assert list(out["n_seconds"]) == [3, 1]
    assert list(out["hs"]) == [0, 0]  # any degraded second marks the cycle degraded
    assert list(out["RUL"]) == [1.0, 0.0]


def test_rul_cap_clips_the_label():
    out = cycle_features(_toy(), ["s"], ["mean"], rul_cap=0.5)
    assert out["RUL"].max() == 0.5


def test_missing_column_is_reported():
    with pytest.raises(KeyError, match="nope"):
        cycle_features(_toy(), ["nope"])


def _cfg(fake_h5, tmp_path) -> Config:
    return Config(
        data=DataConfig(raw_path=fake_h5, processed_dir=tmp_path),
        features=FeatureConfig(stats=["mean", "std"]),
    )


def test_build_split_has_one_row_per_cycle(fake_h5, tmp_path):
    table = build_split(_cfg(fake_h5, tmp_path), "dev")
    assert len(table) == sum(DEV_UNITS.values())
    assert not table[feature_columns(table)].isna().any().any()
    # RUL falls by exactly one per cycle within each engine
    for _, eng in table.groupby("unit"):
        assert np.all(np.diff(eng["RUL"].to_numpy()) == -1)


def test_train_and_test_engines_never_overlap(fake_h5, tmp_path):
    table = pd.read_parquet(prepare(_cfg(fake_h5, tmp_path)))
    dev = set(table.loc[table["split"] == "dev", "unit"])
    test = set(table.loc[table["split"] == "test", "unit"])
    assert dev == set(DEV_UNITS)
    assert test == set(TEST_UNITS)
    assert dev.isdisjoint(test)


def test_feature_columns_exclude_label_and_ids(fake_h5, tmp_path):
    cols = feature_columns(build_split(_cfg(fake_h5, tmp_path), "dev"))
    assert "RUL" not in cols
    assert "unit" not in cols
    assert "hs" not in cols  # health state would leak the label
    assert "T24_mean" in cols


def _history_table() -> pd.DataFrame:
    rows = []
    for unit, offset in ((1, 0.0), (2, 100.0)):
        for cycle in range(1, 7):
            rows.append({"split": "dev", "unit": unit, "cycle": cycle, "Fc": 1,
                         "T50_max": offset + 2.0 * cycle, "T50_std": 0.0,
                         "RUL": float(6 - cycle)})  # fmt: skip
    return pd.DataFrame(rows)


def test_history_mean_and_slope_use_only_the_past():
    from aerorul.features import add_history

    out = add_history(_history_table(), window=3, on_stats=["max"])
    e1 = out[out["unit"] == 1].reset_index(drop=True)
    # trailing mean of 2, 4, 6, ... over up to 3 flights
    assert list(e1["T50_max_roll3"]) == pytest.approx([2, 3, 4, 6, 8, 10])
    # the signal rises 2 per flight; first flight has no slope yet
    assert list(e1["T50_max_slope3"]) == pytest.approx([0, 2, 2, 2, 2, 2])
    # only *_max columns get history
    assert "T50_std_roll3" not in out.columns


def test_history_never_crosses_engines_or_looks_ahead():
    from aerorul.features import add_history

    base = _history_table()
    out = add_history(base, window=3, on_stats=["max"])
    e2 = out[out["unit"] == 2].reset_index(drop=True)
    assert e2.loc[0, "T50_max_roll3"] == pytest.approx(102.0)  # not mixed with engine 1
    changed = base.copy()
    changed.loc[(changed["unit"] == 1) & (changed["cycle"] == 6), "T50_max"] = 999.0
    out2 = add_history(changed, window=3, on_stats=["max"])
    early = (out2["unit"] == 1) & (out2["cycle"] < 6)
    pd.testing.assert_frame_equal(out2[early], out[early])  # the future changed nothing


def test_history_window_zero_is_a_no_op():
    from aerorul.features import add_history

    base = _history_table()
    assert add_history(base, window=0).equals(base)

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

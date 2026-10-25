import numpy as np
import pytest

from aerorul.data import iter_units, unit_blocks

from .conftest import DEV_UNITS, TEST_UNITS, XS_VAR


def test_unit_blocks_finds_contiguous_runs():
    assert unit_blocks(np.array([2, 2, 5, 5, 5, 10])) == [(2, 0, 2), (5, 2, 5), (10, 5, 6)]
    assert unit_blocks(np.array([])) == []


@pytest.mark.parametrize(("split", "units"), [("dev", DEV_UNITS), ("test", TEST_UNITS)])
def test_iter_units_yields_one_frame_per_engine(fake_h5, split, units):
    frames = list(iter_units(fake_h5, split, groups=("W", "X_s")))
    assert [f.attrs["unit"] for f in frames] == list(units)
    for frame in frames:
        assert frame["unit"].nunique() == 1
        assert set(XS_VAR) <= set(frame.columns)
        assert "RUL" in frame.columns
        assert frame["T24"].dtype == np.float32


def test_bad_split_is_rejected(fake_h5):
    with pytest.raises(ValueError, match="split"):
        next(iter_units(fake_h5, "train"))

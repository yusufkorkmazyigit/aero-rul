"""Reading N-CMAPSS HDF5 files.

An N-CMAPSS file holds two splits, ``dev`` (training) and ``test``. Each split stores
the same groups as separate 2-D arrays, one row per second of flight:

    W    flight conditions      alt, Mach, TRA, T2
    X_s  physical sensors       T24, T30, ..., Wf  (14 channels)
    X_v  virtual sensors        model-estimated quantities
    T    health parameters      the simulator's hidden degradation state
    Y    label                  RUL, remaining flight cycles
    A    auxiliary              unit, cycle, Fc (flight class), hs (health state)

Column names live next to the data in ``<group>_var`` arrays. Rows are ordered by unit,
so each engine is one contiguous block; we read one engine at a time to keep memory low.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SPLITS = ("dev", "test")
AUX_INT_COLUMNS = ("unit", "cycle", "Fc", "hs")


def _decode(names: np.ndarray) -> list[str]:
    return [n.decode() if isinstance(n, bytes) else str(n) for n in names]


def column_names(f: h5py.File, group: str) -> list[str]:
    """Column names of one group; Y has no ``_var`` entry and is called RUL."""
    key = f"{group}_var"
    if key in f:
        return _decode(f[key][()])
    if group == "Y":
        return ["RUL"]
    width = f[f"{group}_dev"].shape[1]
    return [f"{group}_{i}" for i in range(width)]


def unit_blocks(units: np.ndarray) -> list[tuple[int, int, int]]:
    """(unit id, start row, stop row) for each contiguous run of the same unit."""
    units = np.asarray(units).ravel()
    if units.size == 0:
        return []
    change = np.flatnonzero(np.diff(units)) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [units.size]))
    return [(int(units[s]), int(s), int(e)) for s, e in zip(starts, stops, strict=True)]


def iter_units(
    path: str | Path,
    split: str = "dev",
    groups: Sequence[str] = ("W", "X_s"),
    dtype: type = np.float32,
) -> Iterator[pd.DataFrame]:
    """Yield one DataFrame per engine: the requested groups plus A and Y columns."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    with h5py.File(path, "r") as f:
        a_names = column_names(f, "A")
        a_all = f[f"A_{split}"][()]
        unit_col = a_names.index("unit")
        names = {g: column_names(f, g) for g in (*groups, "Y")}
        for unit, start, stop in unit_blocks(a_all[:, unit_col]):
            parts = {name: a_all[start:stop, i].astype(np.int64) for i, name in enumerate(a_names)}
            for g in (*groups, "Y"):
                block = f[f"{g}_{split}"][start:stop].astype(dtype)
                for i, name in enumerate(names[g]):
                    parts[name] = block[:, i]
            df = pd.DataFrame(parts)
            df.attrs["unit"] = unit
            yield df


def describe_file(path: str | Path) -> dict:
    """Shapes and column names of every array, without loading the data."""
    with h5py.File(path, "r") as f:
        return {
            key: {"shape": list(f[key].shape), "dtype": str(f[key].dtype)}
            for key in f
            if not key.endswith("_var")
        }

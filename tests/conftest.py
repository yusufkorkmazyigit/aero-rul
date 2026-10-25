"""A tiny synthetic N-CMAPSS file with the real layout, so tests and CI need no download."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

W_VAR = ["alt", "Mach", "TRA", "T2"]
XS_VAR = ["T24", "T30", "T48", "T50", "P15", "P2", "P21",
          "P24", "Ps30", "P40", "P50", "Nf", "Nc", "Wf"]  # fmt: skip
A_VAR = ["unit", "cycle", "Fc", "hs"]

# unit -> number of cycles; dev and test units are disjoint, as in DS02
DEV_UNITS = {2: 6, 5: 5}
TEST_UNITS = {11: 4}


def _split(units: dict[int, int], rng: np.random.Generator):
    w, xs, a, y = [], [], [], []
    for unit, n_cycles in units.items():
        for cycle in range(1, n_cycles + 1):
            seconds = int(rng.integers(20, 40))  # flights have different lengths
            rul = n_cycles - cycle
            hs = 1 if cycle <= n_cycles // 2 else 0
            w.append(rng.normal(size=(seconds, len(W_VAR))))
            # a slow drift with cycle number, like real degradation
            xs.append(rng.normal(loc=cycle * 0.1, size=(seconds, len(XS_VAR))))
            a.append(np.tile([unit, cycle, 1, hs], (seconds, 1)))
            y.append(np.full((seconds, 1), rul))
    return (np.vstack(w), np.vstack(xs), np.vstack(a).astype(np.float64), np.vstack(y))


@pytest.fixture(scope="session")
def fake_h5(tmp_path_factory) -> str:
    rng = np.random.default_rng(0)
    path = tmp_path_factory.mktemp("raw") / "fake_DS02.h5"
    with h5py.File(path, "w") as f:
        for name, units in (("dev", DEV_UNITS), ("test", TEST_UNITS)):
            w, xs, a, y = _split(units, rng)
            f[f"W_{name}"] = w
            f[f"X_s_{name}"] = xs
            f[f"A_{name}"] = a
            f[f"Y_{name}"] = y
        f["W_var"] = np.array(W_VAR, dtype="S")
        f["X_s_var"] = np.array(XS_VAR, dtype="S")
        f["A_var"] = np.array(A_VAR, dtype="S")
    return str(path)


@pytest.fixture
def config_file(tmp_path, fake_h5) -> str:
    path = tmp_path / "test.yaml"
    path.write_text(
        f"seed: 0\n"
        f"data:\n  raw_path: {fake_h5}\n  processed_dir: {tmp_path / 'processed'}\n"
        f"features:\n  groups: [W, X_s]\n  stats: [mean, std]\n",
        encoding="utf-8",
    )
    return str(path)

"""From 1 Hz flight recordings to one row per flight cycle.

A flight cycle is one flight: take-off, climb, cruise, descent. Degradation moves from
cycle to cycle, not from second to second, so the model sees one row per cycle. Each
signal is summarised by a few statistics over the flight. This cuts DS02 from millions
of rows to a few hundred per engine and averages out sensor noise.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import AUX_INT_COLUMNS, iter_units

KEYS = ("unit", "cycle")


def cycle_features(
    df: pd.DataFrame,
    signals: Sequence[str],
    stats: Sequence[str] = ("mean", "std", "min", "max"),
    rul_cap: float | None = None,
) -> pd.DataFrame:
    """Aggregate one or more engines' 1 Hz rows into one row per (unit, cycle).

    Output columns: unit, cycle, Fc, hs, n_seconds, ``<signal>_<stat>`` for every pair,
    and RUL (clipped at ``rul_cap`` when given).
    """
    missing = [c for c in (*KEYS, "RUL", *signals) if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")

    grouped = df.groupby(list(KEYS), sort=True)
    feats = grouped[list(signals)].agg(list(stats))
    feats.columns = [f"{sig}_{stat}" for sig, stat in feats.columns]
    # std over a single sample is NaN; a one-second flight carries no spread.
    std_cols = [c for c in feats.columns if c.endswith("_std")]
    feats[std_cols] = feats[std_cols].fillna(0.0)

    meta = pd.DataFrame(index=feats.index)
    meta["n_seconds"] = grouped.size()
    if "Fc" in df.columns:
        meta["Fc"] = grouped["Fc"].first()
    if "hs" in df.columns:
        # hs: 1 healthy, 0 degraded. A cycle counts as degraded if any second is.
        meta["hs"] = grouped["hs"].min()
    # RUL is constant within a cycle in N-CMAPSS; min() is robust if it is not.
    rul = grouped["RUL"].min().astype("float64")
    if rul_cap is not None:
        rul = rul.clip(upper=rul_cap)

    out = pd.concat([meta, feats], axis=1)
    out["RUL"] = rul
    return out.reset_index()


# Never model inputs:
#   unit   an engine id, not a property of the engine
#   cycle  flights flown so far; with a fixed end of life it nearly gives away the answer,
#          and we want the model to read degradation from the sensors
#   hs     the simulator's health-state flag, derived from the hidden degradation
#   RUL    the label itself
EXCLUDED = frozenset({"unit", "cycle", "hs", "RUL", "split"})


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model input columns: sensor and flight-condition summaries, Fc and flight length."""
    return [c for c in frame.columns if c not in EXCLUDED]


def build_split(cfg: Config, split: str) -> pd.DataFrame:
    """Cycle table for one split, built one engine at a time."""
    frames: list[pd.DataFrame] = []
    signals: list[str] | None = None
    for unit_df in iter_units(cfg.data.raw_path, split, cfg.features.groups):
        if signals is None:
            signals = [c for c in unit_df.columns if c not in (*AUX_INT_COLUMNS, "RUL")]
        frames.append(cycle_features(unit_df, signals, cfg.features.stats, cfg.features.rul_cap))
    if not frames:
        raise ValueError(f"no rows found for split {split!r}")
    table = pd.concat(frames, ignore_index=True)
    table.insert(0, "split", split)
    return table


def prepare(cfg: Config, splits: Iterable[str] = ("dev", "test")) -> Path:
    """Build the cycle table for all splits and write it to Parquet."""
    table = pd.concat([build_split(cfg, s) for s in splits], ignore_index=True)
    out_dir = Path(cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "cycles.parquet"
    table.to_parquet(out, index=False)
    return out


# --------------------------------------------------------------------------------------
# History features
#
# A single flight is noisy: weather, payload and route change every reading. The trend
# over the last few flights is what carries degradation. For each chosen column we add
#
#   <col>_roll<w>   mean over the current and previous w-1 flights of the same engine
#   <col>_slope<w>  least-squares slope over the same window, per flight
#
# Only the current and earlier flights of the same engine are used, so nothing from the
# future and nothing from another engine reaches a row. In the first w-1 flights the
# window is shorter; that is also what a deployed system would have.
# --------------------------------------------------------------------------------------


def _rolling_mean_and_slope(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing-window mean and slope down the rows of ``values`` (n x k), in O(n k)."""
    n = values.shape[0]
    t = np.arange(n, dtype=float)[:, None]

    def trailing_sum(a: np.ndarray) -> np.ndarray:
        c = np.cumsum(np.vstack([np.zeros((1, a.shape[1])), a]), axis=0)
        lag = np.maximum(np.arange(1, n + 1) - window, 0)
        return c[1:] - c[lag]

    m = np.minimum(np.arange(1, n + 1), window).astype(float)[:, None]
    s_x = trailing_sum(values)
    s_t = trailing_sum(t)
    s_tt = trailing_sum(t * t)
    s_tx = trailing_sum(t * values)
    mean = s_x / m
    denom = m * s_tt - s_t**2
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(denom > 0, (m * s_tx - s_t * s_x) / denom, 0.0)
    return mean, slope


def add_history(
    table: pd.DataFrame, window: int, on_stats: Sequence[str] = ("mean", "max")
) -> pd.DataFrame:
    """Append trailing-window mean and slope of every ``*_<stat>`` column, per engine."""
    if window <= 1:
        return table
    suffixes = tuple(f"_{s}" for s in on_stats)
    cols = [c for c in feature_columns(table) if c.endswith(suffixes)]
    if not cols:
        return table
    table = table.sort_values(["split", "unit", "cycle"], kind="stable").reset_index(drop=True)
    means = np.empty((len(table), len(cols)))
    slopes = np.empty_like(means)
    for _, idx in table.groupby(["split", "unit"], sort=False).indices.items():
        m, s = _rolling_mean_and_slope(table[cols].to_numpy(float)[idx], window)
        means[idx], slopes[idx] = m, s
    extra = pd.DataFrame(
        np.hstack([means, slopes]),
        columns=[f"{c}_roll{window}" for c in cols] + [f"{c}_slope{window}" for c in cols],
        index=table.index,
    )
    return pd.concat([table, extra], axis=1)


def load_cycles(cfg: Config) -> pd.DataFrame:
    """The prepared cycle table plus the history features the config asks for."""
    path = Path(cfg.data.processed_dir) / "cycles.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `aerorul prepare` first")
    table = pd.read_parquet(path)
    return add_history(table, cfg.features.history_window, cfg.features.history_on)

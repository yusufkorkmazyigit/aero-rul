"""From 1 Hz flight recordings to one row per flight cycle.

A flight cycle is one flight: take-off, climb, cruise, descent. Degradation moves from
cycle to cycle, not from second to second, so the model sees one row per cycle. Each
signal is summarised by a few statistics over the flight. This cuts DS02 from millions
of rows to a few hundred per engine and averages out sensor noise.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

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

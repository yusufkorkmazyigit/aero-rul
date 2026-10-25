"""Experiment configuration.

Every run is described by one YAML file. Loading it through pydantic means a typo or a
wrong type fails loudly at the start, not halfway through a long run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Stat = Literal["mean", "std", "min", "max", "median", "first", "last"]
Group = Literal["W", "X_s", "X_v", "T"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    raw_path: Path = Path("data/raw/N-CMAPSS_DS02-006.h5")
    processed_dir: Path = Path("data/processed")


class FeatureConfig(_Strict):
    # Which HDF5 groups become model inputs. W = flight conditions, X_s = physical sensors.
    # X_v (virtual sensors) and T (health parameters) are left out by default: T is the
    # simulator's hidden degradation state, so using it would leak the answer.
    groups: list[Group] = Field(default_factory=lambda: ["W", "X_s"])
    stats: list[Stat] = Field(default_factory=lambda: ["mean", "std", "min", "max"])
    # Optional upper clip on the RUL label (piecewise-linear target). None = raw RUL.
    rul_cap: float | None = None


class Config(_Strict):
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)


def load_config(path: str | Path) -> Config:
    """Read a YAML file into a validated Config."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)

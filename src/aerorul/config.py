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


ModelName = Literal["ridge", "lgbm"]


class ModelConfig(_Strict):
    names: list[ModelName] = Field(default_factory=lambda: ["ridge", "lgbm"])
    # Per-model overrides of the defaults in aerorul.models.DEFAULTS.
    params: dict[ModelName, dict] = Field(default_factory=dict)


class EvalConfig(_Strict):
    # Every number we report is a mean +- std over these seeds.
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    results_dir: Path = Path("results")


Method = Literal["split", "cv", "quantile"]


class UncertaintyConfig(_Strict):
    model: ModelName = "lgbm"
    methods: list[Method] = Field(default_factory=lambda: ["split", "cv", "quantile"])
    # Miscoverage levels: 0.1 -> a 90 % interval.
    alphas: list[float] = Field(default_factory=lambda: [0.2, 0.1])
    # Dev engines set aside for calibration by the split method.
    calibration_engines: int = Field(2, ge=1)


Optimizer = Literal["abc", "random", "tpe"]


class TuningConfig(_Strict):
    optimizers: list[Optimizer] = Field(default_factory=lambda: ["abc", "random", "tpe"])
    # Candidate evaluations per search; the same for every optimizer.
    budget: int = Field(60, ge=2)
    # ABC food sources (abcopt pop_size).
    abc_pop_size: int = Field(10, ge=2)


class Config(_Strict):
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)


def load_config(path: str | Path) -> Config:
    """Read a YAML file into a validated Config."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)

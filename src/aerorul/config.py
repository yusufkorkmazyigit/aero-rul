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
    # History features: trailing mean and slope over the last `history_window` flights of
    # the same engine, for every `<signal>_<stat>` column with a stat in `history_on`.
    # 0 or 1 turns them off.
    history_window: int = Field(10, ge=0)
    history_on: list[Stat] = Field(default_factory=lambda: ["mean", "max"])


ModelName = Literal["ridge", "lgbm"]


class ModelConfig(_Strict):
    names: list[ModelName] = Field(default_factory=lambda: ["ridge", "lgbm"])
    # Per-model overrides of the defaults in aerorul.models.DEFAULTS.
    params: dict[ModelName, dict] = Field(default_factory=dict)


class EvalConfig(_Strict):
    # Every number we report is a mean +- std over these seeds.
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    results_dir: Path = Path("results")


Method = Literal["split", "cv", "quantile", "cqr", "norm_pred", "norm_feat"]


class UncertaintyConfig(_Strict):
    model: ModelName = "lgbm"
    methods: list[Method] = Field(
        default_factory=lambda: ["split", "cv", "quantile", "cqr", "norm_pred", "norm_feat"]
    )
    # Miscoverage levels: 0.1 -> a 90 % interval.
    alphas: list[float] = Field(default_factory=lambda: [0.2, 0.1])
    # Dev engines set aside for calibration by the split method.
    calibration_engines: int = Field(2, ge=1)


Policy = Literal["point", "point_margin", "cv", "quantile", "cqr", "norm_pred"]


class DecisionConfig(_Strict):
    model: ModelName = "lgbm"
    policies: list[Policy] = Field(
        default_factory=lambda: ["point", "point_margin", "cv", "quantile", "cqr", "norm_pred"]
    )
    # Interval policies book a removal on the lower edge of a (1 - alpha) interval.
    alpha: float = Field(0.1, gt=0, lt=1)
    # A shop slot must be booked this many flights ahead.
    lead: int = Field(10, ge=0)
    # How many flights of wasted life one unplanned failure costs.
    cost_ratios: list[float] = Field(default_factory=lambda: [20.0, 50.0, 100.0, 200.0])
    # Largest safety margin (flights) tried when tuning the point_margin policy.
    max_margin: int = Field(60, ge=0)


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
    decision: DecisionConfig = Field(default_factory=DecisionConfig)


def load_config(path: str | Path) -> Config:
    """Read a YAML file into a validated Config."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)

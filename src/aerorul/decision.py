"""From a prediction to a maintenance decision.

A planner does not consume "38 flights left, 90 % interval [21, 55]". They decide, after
every flight, whether to book the engine into the shop. A shop slot has to be booked
``lead`` flights ahead, so the decision rule is:

    after each flight, book a removal ``lead`` flights from now as soon as a statistic of
    the prediction (the point estimate, or the lower edge of an interval) is <= ``lead``.

Two things can go wrong:

* **Failure.** The engine's true remaining life at the moment of booking is shorter than
  ``lead``: it fails before its slot. If nothing ever triggers, it flies to failure too.
* **Waste.** The engine is removed with life left: ``RUL at booking - lead`` flights of a
  serviceable engine are thrown away.

The cost of one engine is ``waste + ratio * failure``; ``ratio`` says how many flights of
wasted life one unplanned failure is worth. Real ratios are large (an in-service failure
means an unscheduled removal, a grounded aircraft, possibly a safety event); we sweep it.

Policies compared, all with the same model and the same lead time:

``point``
    Book when the predicted RUL is <= lead. The naive use of a point estimate.
``point_margin``
    Book when predicted RUL <= lead + m. The margin m is tuned to minimise cost on
    out-of-fold predictions of the training engines, separately for every cost ratio.
    The engineer's usual fix, and the baseline to beat.
``cv``, ``quantile``, ``cqr``, ``norm_pred``
    Book when the lower edge of that method's interval is <= lead. No tuning: the target
    coverage alone sets how cautious the policy is.

Note what this means for ``cv``: its interval is pred +- q with constant q, so it *is* a
margin policy. The only difference from ``point_margin`` is who chooses the margin: a
coverage target or a cost function.

**Protocol.** One engine per subset gives one decision, and three test engines give three.
The decision study therefore runs leave-one-engine-out over all engines: each engine in
turn is held out completely, and every model, interval and margin is built from the other
engines alone (with an inner leave-one-engine-out where calibration needs it).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .evaluation import cross_validate, split_xy
from .features import feature_columns
from .models import make_model
from .uncertainty import (
    conformal_quantile,
    cqr_intervals,
    normalized_radius,
    quantile_intervals,
)

POLICIES = ("point", "point_margin", "cv", "quantile", "cqr", "norm_pred")
INTERVAL_POLICIES = ("cv", "quantile", "cqr", "norm_pred")


def decide(cycle, rul, stat, lead: float) -> dict[str, float]:
    """Outcome of booking a removal when ``stat <= lead`` first holds, for one engine."""
    order = np.argsort(np.asarray(cycle))
    rul, stat = np.asarray(rul, float)[order], np.asarray(stat, float)[order]
    hits = np.flatnonzero(stat <= lead)
    if hits.size == 0:  # never booked: flies to failure
        return {"trigger_cycle": np.nan, "rul_at_trigger": np.nan, "failed": 1, "wasted": 0.0}
    i = hits[0]
    r = float(rul[i])
    failed = int(r < lead)
    return {"trigger_cycle": float(np.asarray(cycle)[order][i]), "rul_at_trigger": r,
            "failed": failed, "wasted": 0.0 if failed else r - lead}  # fmt: skip


def policy_outcomes(frame: pd.DataFrame, stat: str, lead: float) -> pd.DataFrame:
    """``decide`` for every engine in ``frame`` (columns unit, cycle, RUL and ``stat``)."""
    rows = [{"unit": unit, **decide(g["cycle"], g["RUL"], g[stat], lead)}
            for unit, g in frame.groupby("unit")]  # fmt: skip
    return pd.DataFrame(rows)


def cost(outcomes: pd.DataFrame, ratio: float) -> float:
    """Mean cost per engine: wasted flights + ratio * failures."""
    return float(np.mean(outcomes["wasted"] + ratio * outcomes["failed"]))


def tune_margin(frame: pd.DataFrame, lead: float, ratio: float, margins) -> float:
    """The margin with the lowest mean cost on ``frame`` (ties: the smallest margin)."""
    costs = [cost(policy_outcomes(frame.assign(stat=frame["pred"] - m), "stat", lead), ratio)
             for m in margins]  # fmt: skip
    return float(margins[int(np.argmin(costs))])


def held_out_predictions(
    train: pd.DataFrame,
    held: pd.DataFrame,
    name: str,
    alpha: float,
    params: dict | None,
    seed: int,
    policies: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Point prediction and interval lower edges for the held-out engine.

    Returns (held-out frame with ``pred`` and ``lo_<method>`` columns, training-engine
    frame with out-of-fold ``pred`` for margin tuning).
    """
    X, y, _ = split_xy(train)
    _, oof = cross_validate(train, name, params, seed)
    X_held = held[feature_columns(held)]
    pred = make_model(name, params, seed).fit(X, y).predict(X_held)
    out = held[["unit", "cycle", "RUL"]].assign(pred=pred)
    if "cv" in policies:
        out["lo_cv"] = pred - conformal_quantile(np.abs(y - oof), alpha)
    if "norm_pred" in policies:
        out["lo_norm_pred"] = pred - normalized_radius(train, oof, X_held, pred, alpha, "pred",
                                                       seed)  # fmt: skip
    if "quantile" in policies:
        out["lo_quantile"] = quantile_intervals(train, held, alpha, params, seed)["lo"].to_numpy()
    if "cqr" in policies:
        out["lo_cqr"] = cqr_intervals(train, held, alpha, params, seed)["lo"].to_numpy()
    tune = train[["unit", "cycle", "RUL"]].assign(pred=oof)
    return out, tune


def run_decision_study(
    table: pd.DataFrame,
    name: str,
    alpha: float,
    seeds: Sequence[int],
    lead: float,
    ratios: Sequence[float],
    policies: Sequence[str] = POLICIES,
    params: dict | None = None,
    max_margin: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-engine-out over every engine in ``table``, for each seed.

    Returns (outcomes, predictions): one outcome row per seed, held-out engine, policy and
    cost ratio; and every held-out prediction, for the waste/failure frontier.
    """
    unknown = set(policies) - set(POLICIES)
    if unknown:
        raise ValueError(f"unknown policies {sorted(unknown)}; choose from {POLICIES}")
    margins = np.arange(0, max_margin + 1)
    units = np.sort(table["unit"].unique())
    rows, preds = [], []
    for seed, unit in tqdm(list(product(seeds, units)), desc="decisions", unit="engine"):
        train = table[table["unit"] != unit].reset_index(drop=True)
        held = table[table["unit"] == unit].reset_index(drop=True)
        out, tune = held_out_predictions(train, held, name, alpha, params, seed, policies)
        preds.append(out.assign(seed=seed))
        fixed = [("point", "pred")] if "point" in policies else []
        fixed += [(p, f"lo_{p}") for p in INTERVAL_POLICIES if p in policies]
        for policy, col in fixed:
            o = decide(out["cycle"], out["RUL"], out[col], lead)
            rows += [{"seed": seed, "unit": unit, "policy": policy, "ratio": r, "margin": np.nan,
                      **o} for r in ratios]  # fmt: skip
        if "point_margin" in policies:
            for r in ratios:
                m = tune_margin(tune, lead, r, margins)
                o = decide(out["cycle"], out["RUL"], out["pred"] - m, lead)
                rows.append({"seed": seed, "unit": unit, "policy": "point_margin", "ratio": r,
                             "margin": m, **o})  # fmt: skip
    outcomes = pd.DataFrame(rows)
    outcomes["cost"] = outcomes["wasted"] + outcomes["ratio"] * outcomes["failed"]
    return outcomes, pd.concat(preds, ignore_index=True)


def summarize_decisions(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Per policy and cost ratio: failure rate, mean waste and mean cost per engine."""
    g = outcomes.groupby(["ratio", "policy"])
    out = pd.DataFrame({
        "failure_rate": g["failed"].mean(),
        "wasted_mean": g["wasted"].mean(),
        "cost_mean": g["cost"].mean(),
        # spread over seeds of the per-seed mean cost
        "cost_std": outcomes.groupby(["ratio", "policy", "seed"])["cost"].mean()
        .groupby(["ratio", "policy"]).std(),
        "margin_mean": g["margin"].mean(),
    })  # fmt: skip
    order = {p: i for i, p in enumerate(POLICIES)}
    out = out.reset_index()
    return out.sort_values(
        ["ratio", "policy"], key=lambda c: c.map(order) if c.name == "policy" else c
    )


def frontier(predictions: pd.DataFrame, lead: float, margins) -> pd.DataFrame:
    """Failure rate and mean waste of the constant-margin policy for every margin.

    Evaluated on held-out predictions, for display only: it shows the trade-off every
    margin buys, not a margin anyone could have chosen in advance.
    """
    rows = []
    for m in margins:
        o = pd.concat([policy_outcomes(g.assign(stat=g["pred"] - m), "stat", lead)
                       for _, g in predictions.groupby("seed")])  # fmt: skip
        rows.append({"margin": m, "failure_rate": o["failed"].mean(),
                     "wasted_mean": o["wasted"].mean()})  # fmt: skip
    return pd.DataFrame(rows)


def plot_frontier(front: pd.DataFrame, summary: pd.DataFrame, ratio: float, path) -> None:
    """Waste against failure rate: the margin curve, and where each policy lands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(front["wasted_mean"], 100 * front["failure_rate"], color="#999999", lw=1.2,
            label="point estimate - constant margin")  # fmt: skip
    for m in (0, 10, 20, 30):
        r = front[front["margin"] == m]
        if len(r):
            xy = (r["wasted_mean"].iloc[0], 100 * r["failure_rate"].iloc[0])
            ax.annotate(f"m={m}", xy, textcoords="offset points", xytext=(4, 4), fontsize=7,
                        color="#777777")  # fmt: skip
    markers = {"point": "o", "point_margin": "D", "cv": "s", "quantile": "^", "cqr": "v",
               "norm_pred": "P"}  # fmt: skip
    sel = summary[np.isclose(summary["ratio"], ratio)]
    for _, row in sel.iterrows():
        ax.scatter(row["wasted_mean"], 100 * row["failure_rate"], s=46, zorder=3,
                   marker=markers.get(row["policy"], "o"), label=row["policy"])  # fmt: skip
    # Beyond a few times the policies' waste the curve is flat at zero failures.
    ax.set_xlim(0, max(10.0, 2.2 * float(sel["wasted_mean"].max())))
    ax.set_xlabel("Wasted life per engine (flights)")
    ax.set_ylabel("Engines failing before removal (%)")
    ax.set_title(f"Maintenance policies (failure = {ratio:g} flights of waste)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

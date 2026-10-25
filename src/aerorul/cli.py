"""Command line interface: ``aerorul <command> --config configs/default.yaml``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from . import __version__
from .config import load_config
from .data import describe_file
from .features import feature_columns, prepare

app = typer.Typer(
    help="Remaining useful life prediction for turbofan engines (NASA N-CMAPSS).",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOpt = typer.Option(Path("configs/default.yaml"), "--config", "-c", help="YAML config.")


@app.callback()
def main() -> None:
    """aerorul command line tool."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def inspect(config: Path = ConfigOpt) -> None:
    """Show the arrays inside the raw HDF5 file without loading them."""
    cfg = load_config(config)
    typer.echo(json.dumps(describe_file(cfg.data.raw_path), indent=2))


@app.command(name="prepare")
def prepare_cmd(config: Path = ConfigOpt) -> None:
    """Turn the raw 1 Hz HDF5 file into one row per flight cycle (Parquet)."""
    cfg = load_config(config)
    out = prepare(cfg)
    table = pd.read_parquet(out)
    for split, part in table.groupby("split", sort=False):
        units = sorted(part["unit"].unique().tolist())
        typer.echo(f"{split:>5}: {len(part):>5} cycles, units {units}")
    typer.echo(f"features: {len(feature_columns(table))}")
    typer.echo(f"wrote {out}")


def _load_table(cfg) -> pd.DataFrame:
    from .features import load_cycles

    try:
        return load_cycles(cfg)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def train(config: Path = ConfigOpt) -> None:
    """Fit each model on all dev engines and save it (seed = config `seed`)."""
    import joblib

    from .evaluation import split_xy
    from .models import make_model

    cfg = load_config(config)
    table = _load_table(cfg)
    X, y, _ = split_xy(table[table["split"] == "dev"])
    out_dir = Path(cfg.evaluation.results_dir) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in cfg.models.names:
        model = make_model(name, cfg.models.params.get(name), cfg.seed).fit(X, y)
        joblib.dump({"model": model, "features": list(X.columns)}, out_dir / f"{name}.joblib")
        typer.echo(f"saved {out_dir / f'{name}.joblib'}")


@app.command()
def evaluate(config: Path = ConfigOpt) -> None:
    """Leave-one-engine-out CV on dev and final test score, over every seed."""
    from .evaluation import per_unit, run_baselines, summarize

    cfg = load_config(config)
    table = _load_table(cfg)
    scores, preds = run_baselines(table, cfg.models.names, cfg.evaluation.seeds, cfg.models.params)
    out_dir = Path(cfg.evaluation.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "baseline_scores.csv", index=False)
    preds.to_parquet(out_dir / "baseline_predictions.parquet", index=False)

    summary = summarize(scores).round(2)
    units = per_unit(preds)
    report = (
        f"# Baselines\n\nSeeds: {list(cfg.evaluation.seeds)}. "
        "RMSE and MAE in flight cycles; NASA score lower is better.\n\n"
        f"{summary.to_markdown(index=False)}\n\n## Test RMSE per engine\n\n"
        f"{units.to_markdown()}\n"
    )
    (out_dir / "baselines.md").write_text(report, encoding="utf-8")
    typer.echo(report)


@app.command()
def uncertainty(config: Path = ConfigOpt) -> None:
    """Prediction intervals on the test engines: coverage and width per method."""
    from .uncertainty import coverage_per_unit, plot_bands, run_uncertainty, summarize_intervals

    cfg = load_config(config)
    uc = cfg.uncertainty
    table = _load_table(cfg)
    scores, intervals = run_uncertainty(
        table,
        uc.model,
        uc.methods,
        uc.alphas,
        cfg.evaluation.seeds,
        cfg.models.params.get(uc.model),
        uc.calibration_engines,
    )
    out_dir = Path(cfg.evaluation.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / "interval_scores.csv", index=False)
    intervals.to_parquet(out_dir / "intervals.parquet", index=False)

    figures = []
    for method in uc.methods:
        alpha = min(uc.alphas)
        fig = out_dir / f"bands_{method}_{round(100 * (1 - alpha))}.png"
        plot_bands(intervals, method, alpha, fig)
        figures.append(fig.name)

    report = (
        f"# Prediction intervals ({uc.model})\n\n"
        "Coverage = share of test cycles whose true RUL falls inside the interval; "
        "it should be close to the target. Width in flight cycles; narrower is better "
        "at the same coverage.\n\n"
        f"{summarize_intervals(scores).round(3).to_markdown(index=False)}\n\n"
        "## Coverage per test engine\n\n"
        f"{coverage_per_unit(intervals).round(3).to_markdown()}\n\n"
        "## Figures\n\n" + "\n".join(f"- {f}" for f in figures) + "\n"
    )
    (out_dir / "intervals.md").write_text(report, encoding="utf-8")
    typer.echo(report)


@app.command()
def benchmark(config: Path = ConfigOpt) -> None:
    """ABC vs random search vs TPE for LightGBM, same budget, same CV, several seeds."""
    from .tuning import plot_convergence, run_benchmark, summarize_benchmark

    cfg = load_config(config)
    tc = cfg.tuning
    table = _load_table(cfg)
    results, traces = run_benchmark(
        table, tc.optimizers, tc.budget, cfg.evaluation.seeds, tc.abc_pop_size
    )
    out_dir = Path(cfg.evaluation.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "benchmark_results.csv", index=False)
    traces.to_csv(out_dir / "benchmark_traces.csv", index=False)
    plot_convergence(traces, out_dir / "convergence.png")

    report = (
        f"# Hyperparameter search benchmark\n\n"
        f"LightGBM, budget {tc.budget} evaluations per search, leave-one-engine-out CV "
        f"on dev engines, seeds {list(cfg.evaluation.seeds)}. RMSE in flight cycles. "
        "`cv_wins`: seeds in which the optimizer found the lowest CV RMSE.\n\n"
        f"{summarize_benchmark(results).round(3).to_markdown(index=False)}\n\n"
        "Figure: convergence.png\n"
    )
    (out_dir / "benchmark.md").write_text(report, encoding="utf-8")
    typer.echo(report)


@app.command()
def explain(config: Path = ConfigOpt) -> None:
    """SHAP importance of each signal for the LightGBM model on the test engines."""
    from .explain import plot_importance, shap_importance

    cfg = load_config(config)
    table = _load_table(cfg)
    per_feature, per_signal = shap_importance(table, cfg.models.params.get("lgbm"), cfg.seed)
    out_dir = Path(cfg.evaluation.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_importance(per_signal, out_dir / "shap_signals.png")
    report = (
        "# What drives the prediction\n\n"
        "Mean |SHAP| on test engines, in cycles of RUL. Correlated sensors share "
        "credit; this explains the model, not the physics.\n\n"
        f"## Per signal\n\n{per_signal.round(3).rename('mean_abs_shap').to_markdown()}\n\n"
        f"## Top 15 features\n\n"
        f"{per_feature.head(15).round(3).rename('mean_abs_shap').to_markdown()}\n\n"
        "Figure: shap_signals.png\n"
    )
    (out_dir / "explain.md").write_text(report, encoding="utf-8")
    typer.echo(report)


if __name__ == "__main__":
    app()

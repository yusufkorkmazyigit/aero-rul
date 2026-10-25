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


if __name__ == "__main__":
    app()

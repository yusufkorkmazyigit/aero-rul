from pathlib import Path

import pytest
from typer.testing import CliRunner

from aerorul import __version__
from aerorul.cli import app
from aerorul.config import load_config

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_prepare_writes_parquet(config_file):
    result = runner.invoke(app, ["prepare", "--config", config_file])
    assert result.exit_code == 0, result.output
    out = Path(load_config(config_file).data.processed_dir) / "cycles.parquet"
    assert out.exists()
    assert "dev:" in result.output


def test_inspect_lists_arrays(config_file):
    result = runner.invoke(app, ["inspect", "-c", config_file])
    assert result.exit_code == 0
    assert "X_s_dev" in result.output


def test_unknown_config_key_fails(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("features:\n  rul_capp: 100\n", encoding="utf-8")
    with pytest.raises(Exception, match="rul_capp"):
        load_config(bad)

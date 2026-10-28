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


def test_train_and_evaluate_end_to_end(config_file):
    assert runner.invoke(app, ["prepare", "-c", config_file]).exit_code == 0
    trained = runner.invoke(app, ["train", "-c", config_file])
    assert trained.exit_code == 0, trained.output
    evaluated = runner.invoke(app, ["evaluate", "-c", config_file])
    assert evaluated.exit_code == 0, evaluated.output
    results = Path(load_config(config_file).evaluation.results_dir)
    assert (results / "models" / "lgbm.joblib").exists()
    assert (results / "baselines.md").exists()
    assert "rmse_mean" in evaluated.output
    intervals = runner.invoke(app, ["uncertainty", "-c", config_file])
    assert intervals.exit_code == 0, intervals.output
    assert (results / "intervals.md").exists()
    assert (results / "bands_split_90.png").exists()
    assert "coverage_mean" in intervals.output
    decided = runner.invoke(app, ["decide", "-c", config_file])
    assert decided.exit_code == 0, decided.output
    assert (results / "decisions.md").exists()
    assert (results / "decision_frontier.png").exists()
    assert "failure_rate" in decided.output
    bench = runner.invoke(app, ["benchmark", "-c", config_file])
    assert bench.exit_code == 0, bench.output
    assert (results / "convergence.png").exists()
    explained = runner.invoke(app, ["explain", "-c", config_file])
    assert explained.exit_code == 0, explained.output
    assert (results / "shap_signals.png").exists()


def test_evaluate_without_prepare_explains_what_to_do(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"data:\n  processed_dir: {tmp_path / 'none'}\n", encoding="utf-8")
    result = runner.invoke(app, ["evaluate", "-c", str(cfg)])
    assert result.exit_code != 0
    assert "aerorul prepare" in result.output


def test_inspect_lists_arrays(config_file):
    result = runner.invoke(app, ["inspect", "-c", config_file])
    assert result.exit_code == 0
    assert "X_s_dev" in result.output


def test_unknown_config_key_fails(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("features:\n  rul_capp: 100\n", encoding="utf-8")
    with pytest.raises(Exception, match="rul_capp"):
        load_config(bad)

"""Tests for CLI command structure and argument parsing."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from inferall.cli.app import app


runner = CliRunner()


class TestCLIAppStructure:
    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # Typer exits with code 2 for no_args_is_help
        assert result.exit_code in (0, 2)
        assert "model_engine" in result.output.lower() or "Usage" in result.output

    def test_help_flag(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "pull" in result.output
        assert "serve" in result.output
        assert "run" in result.output
        assert "list" in result.output
        assert "status" in result.output
        assert "remove" in result.output
        assert "login" in result.output


class TestPullCommand:
    def test_pull_help(self):
        result = runner.invoke(app, ["pull", "--help"])
        assert result.exit_code == 0
        assert "--variant" in result.output
        assert "--trust-remote-code" in result.output
        assert "--force" in result.output


class TestServeCommand:
    def test_serve_help(self):
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--host" in result.output
        assert "--api-key" in result.output
        assert "--compat-mode" in result.output
        assert "--workers" in result.output


class TestRunCommand:
    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--system" in result.output
        assert "--temperature" in result.output
        assert "--max-tokens" in result.output
        assert "--top-p" in result.output
        assert "--top-k" in result.output
        assert "--repetition-penalty" in result.output


class TestListCommand:
    def test_list_help(self):
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0


class TestStatusCommand:
    @patch("inferall.cli.commands.status.GPUManager")
    def test_status_no_gpus(self, mock_gpu_cls):
        mock_mgr = MagicMock()
        mock_mgr.n_gpus = 0
        mock_mgr.gpu_assignments = {}
        mock_gpu_cls.return_value = mock_mgr

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "No GPUs" in result.output or "No models" in result.output


class TestRemoveCommand:
    def test_remove_help(self):
        result = runner.invoke(app, ["remove", "--help"])
        assert result.exit_code == 0
        assert "--yes" in result.output

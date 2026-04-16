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


class TestTuiCommand:
    def test_tui_is_registered_in_root_help(self):
        """`inferall --help` should list the new tui subcommand."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "tui" in result.output

    def test_tui_help(self):
        result = runner.invoke(app, ["tui", "--help"])
        assert result.exit_code == 0
        assert "--url" in result.output
        assert "-u" in result.output
        assert "dashboard" in result.output.lower()

    @patch("inferall.cli.commands.tui.run_dashboard")
    def test_tui_default_url(self, mock_run_dashboard):
        result = runner.invoke(app, ["tui"])
        assert result.exit_code == 0
        mock_run_dashboard.assert_called_once_with(server_url="http://127.0.0.1:8000")

    @patch("inferall.cli.commands.tui.run_dashboard")
    def test_tui_custom_url_long_flag(self, mock_run_dashboard):
        result = runner.invoke(app, ["tui", "--url", "http://10.0.0.5:9000"])
        assert result.exit_code == 0
        mock_run_dashboard.assert_called_once_with(server_url="http://10.0.0.5:9000")

    @patch("inferall.cli.commands.tui.run_dashboard")
    def test_tui_custom_url_short_flag(self, mock_run_dashboard):
        result = runner.invoke(app, ["tui", "-u", "http://inferall.local:8001"])
        assert result.exit_code == 0
        mock_run_dashboard.assert_called_once_with(server_url="http://inferall.local:8001")

    @patch("inferall.cli.commands.tui.run_dashboard")
    def test_tui_surfaces_dashboard_errors(self, mock_run_dashboard):
        """If the dashboard itself raises, the CLI should not silently succeed."""
        mock_run_dashboard.side_effect = RuntimeError("cannot connect to server")
        result = runner.invoke(app, ["tui"])
        assert result.exit_code != 0

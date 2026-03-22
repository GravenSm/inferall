"""Tests for inferall.config — EngineConfig loading and layered resolution."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from inferall.config import EngineConfig, _apply_dict


class TestEngineConfigDefaults:
    """Test built-in defaults."""

    def test_default_values(self):
        config = EngineConfig()
        assert config.default_port == 8000
        assert config.default_host == "127.0.0.1"
        assert config.idle_timeout == 300
        assert config.vram_buffer_mb == 512
        assert config.max_loaded_models == 3
        assert config.inference_workers == 2
        assert config.api_key is None
        assert config.trust_remote_code is False

    def test_derived_paths(self):
        config = EngineConfig()
        assert config.models_dir == config.base_dir / "models"
        assert config.registry_path == config.base_dir / "registry.db"

    def test_custom_base_dir(self, tmp_path):
        config = EngineConfig(base_dir=tmp_path)
        assert config.models_dir == tmp_path / "models"
        assert config.registry_path == tmp_path / "registry.db"

    def test_explicit_models_dir(self, tmp_path):
        models = tmp_path / "custom_models"
        config = EngineConfig(base_dir=tmp_path, models_dir=models)
        assert config.models_dir == models

    def test_ensure_dirs(self, tmp_path):
        config = EngineConfig(base_dir=tmp_path / "new_dir")
        config.ensure_dirs()
        assert config.base_dir.exists()
        assert config.models_dir.exists()


class TestEngineConfigEnvVars:
    """Test environment variable overrides."""

    def test_env_port(self):
        with patch.dict(os.environ, {"INFERALL_PORT": "9999"}):
            config = EngineConfig.load()
        assert config.default_port == 9999

    def test_env_host(self):
        with patch.dict(os.environ, {"INFERALL_HOST": "0.0.0.0"}):
            config = EngineConfig.load()
        assert config.default_host == "0.0.0.0"

    def test_env_api_key(self):
        with patch.dict(os.environ, {"INFERALL_API_KEY": "secret123"}):
            config = EngineConfig.load()
        assert config.api_key == "secret123"

    def test_invalid_env_value_ignored(self):
        with patch.dict(os.environ, {"INFERALL_PORT": "not_a_number"}):
            config = EngineConfig.load()
        assert config.default_port == 8000  # falls back to default

    def test_env_base_dir_re_derives_paths(self, tmp_path):
        with patch.dict(os.environ, {"INFERALL_BASE_DIR": str(tmp_path)}):
            config = EngineConfig.load()
        assert config.base_dir == tmp_path
        assert config.models_dir == tmp_path / "models"
        assert config.registry_path == tmp_path / "registry.db"


class TestEngineConfigCLIOverrides:
    """Test CLI override layer (highest priority)."""

    def test_cli_overrides_port(self):
        config = EngineConfig.load(cli_overrides={"default_port": 4242})
        assert config.default_port == 4242

    def test_cli_overrides_beat_env(self):
        with patch.dict(os.environ, {"INFERALL_PORT": "9999"}):
            config = EngineConfig.load(cli_overrides={"default_port": 1111})
        assert config.default_port == 1111

    def test_cli_overrides_none_ignored(self):
        config = EngineConfig.load(cli_overrides={"default_port": None})
        assert config.default_port == 8000


class TestApplyDict:
    """Test the _apply_dict helper."""

    def test_coerces_path(self):
        config = EngineConfig()
        _apply_dict(config, {"base_dir": "/tmp/test"})
        assert isinstance(config.base_dir, Path)
        assert config.base_dir == Path("/tmp/test")

    def test_coerces_int(self):
        config = EngineConfig()
        _apply_dict(config, {"default_port": "3000"})
        assert config.default_port == 3000

    def test_skips_none(self):
        config = EngineConfig()
        original_port = config.default_port
        _apply_dict(config, {"default_port": None})
        assert config.default_port == original_port

    def test_skips_unknown_keys(self):
        config = EngineConfig()
        _apply_dict(config, {"nonexistent_field": 42})
        assert not hasattr(config, "nonexistent_field")

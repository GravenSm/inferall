"""
Engine Configuration
--------------------
EngineConfig dataclass with path setup, defaults, and config file loading.

Config resolution order (highest priority first):
1. CLI flags
2. Environment variables (INFERALL_*)
3. Config file (~/.inferall/config.yaml)
4. Built-in defaults
"""

import logging
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_BASE_DIR = Path.home() / ".inferall"


@dataclass
class EngineConfig:
    """Central configuration for the model engine."""

    base_dir: Path = field(default_factory=lambda: _DEFAULT_BASE_DIR)
    models_dir: Path = field(default=None)  # type: ignore[assignment]
    files_dir: Path = field(default=None)  # type: ignore[assignment]
    registry_path: Path = field(default=None)  # type: ignore[assignment]
    default_port: int = 8000
    default_host: str = "127.0.0.1"
    idle_timeout: int = 300
    vram_buffer_mb: int = 512
    max_loaded_models: int = 3
    inference_workers: int = 2
    max_concurrent_requests: int = 16
    concurrency_per_model: int = 1
    model_queue_size: int = 64
    api_key: Optional[str] = None
    ollama_api_key: Optional[str] = None
    trust_remote_code: bool = False

    def __post_init__(self):
        # Derive paths from base_dir if not explicitly set
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
        if self.files_dir is None:
            self.files_dir = self.base_dir / "files"
        if self.registry_path is None:
            self.registry_path = self.base_dir / "registry.db"

    def ensure_dirs(self) -> None:
        """Create base and model directories if they don't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, cli_overrides: Optional[dict] = None) -> "EngineConfig":
        """
        Build config from defaults → config file → env vars → CLI overrides.

        Args:
            cli_overrides: Dict of field_name → value from CLI flags.
                           Only non-None values are applied.
        """
        # Start with defaults
        config = cls()

        # Layer 2: config file
        config_file = config.base_dir / "config.yaml"
        if config_file.exists():
            file_values = _load_yaml_config(config_file)
            _apply_dict(config, file_values)

        # Layer 1: environment variables
        env_map = {
            "INFERALL_PORT": ("default_port", int),
            "INFERALL_HOST": ("default_host", str),
            "INFERALL_API_KEY": ("api_key", str),
            "INFERALL_IDLE_TIMEOUT": ("idle_timeout", int),
            "INFERALL_VRAM_BUFFER_MB": ("vram_buffer_mb", int),
            "INFERALL_MAX_LOADED": ("max_loaded_models", int),
            "INFERALL_WORKERS": ("inference_workers", int),
            "INFERALL_BASE_DIR": ("base_dir", Path),
            "OLLAMA_API_KEY": ("ollama_api_key", str),
        }
        for env_var, (field_name, type_fn) in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                try:
                    setattr(config, field_name, type_fn(val))
                except (ValueError, TypeError):
                    logger.warning("Invalid value for %s: %r, ignoring", env_var, val)

        # Re-derive paths if base_dir changed
        if os.environ.get("INFERALL_BASE_DIR"):
            config.models_dir = config.base_dir / "models"
            config.files_dir = config.base_dir / "files"
            config.registry_path = config.base_dir / "registry.db"

        # Layer 0: CLI overrides (highest priority)
        if cli_overrides:
            _apply_dict(config, cli_overrides)

        return config


def _apply_dict(config: EngineConfig, values: dict) -> None:
    """Apply a dict of field_name → value to config, skipping None values."""
    for key, val in values.items():
        if val is not None and hasattr(config, key):
            current = getattr(config, key)
            # Coerce types
            if isinstance(current, Path) and not isinstance(val, Path):
                val = Path(val)
            elif isinstance(current, int) and not isinstance(val, int):
                val = int(val)
            setattr(config, key, val)


def _load_yaml_config(path: Path) -> dict:
    """Load config.yaml and check file permissions."""
    # Check permissions — warn if world-readable
    try:
        file_stat = path.stat()
        mode = file_stat.st_mode
        if mode & stat.S_IROTH:
            logger.warning(
                "Config file %s is world-readable (mode %o). "
                "Consider: chmod 600 %s",
                path, stat.S_IMODE(mode), path,
            )
    except OSError:
        pass

    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            logger.warning("Config file %s is not a YAML mapping, ignoring", path)
            return {}
        return data
    except ImportError:
        logger.warning("PyYAML not installed, skipping config file %s", path)
        return {}
    except Exception as e:
        logger.warning("Failed to load config file %s: %s", path, e)
        return {}

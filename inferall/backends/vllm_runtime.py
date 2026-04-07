"""
vLLM Runtime Discovery + Bootstrap
-----------------------------------
vLLM is intentionally NOT installed in the main inferall venv. Reasons:

1. vLLM 0.19.x pins ``transformers<5`` while inferall uses transformers 5.x.
2. vLLM pulls in ~3 GB of extra dependencies (flashinfer, quack-kernels,
   compressed-tensors, lm-format-enforcer, xgrammar, etc.) that are unrelated
   to the rest of inferall.
3. vLLM ships frequent breaking changes between releases. Pinning it inside
   inferall would couple two unrelated release cadences.

Instead, vLLM lives in its own isolated venv. We talk to it as a subprocess
over its OpenAI-compatible HTTP server. This is the same way most production
deployments run vLLM, including chandra's reference setup.

Detection order:
1. ``INFERALL_VLLM_PYTHON`` env var (absolute path to a Python interpreter)
2. ``~/.cache/inferall/vllm-venv/bin/python`` (the bootstrap location)
3. Raises ``VLLMNotInstalled`` with install instructions
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


DEFAULT_VENV_PATH = Path.home() / ".cache" / "inferall" / "vllm-venv"
DEFAULT_VLLM_VERSION = "0.19.0"
PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


class VLLMNotInstalled(RuntimeError):
    """vLLM runtime not found and could not be auto-located."""


def find_vllm_python() -> Path:
    """
    Locate a Python interpreter that has vllm installed.

    Returns:
        Path to the python binary.

    Raises:
        VLLMNotInstalled: if no interpreter could be found.
    """
    # 1. Explicit env var override
    env_path = os.environ.get("INFERALL_VLLM_PYTHON")
    if env_path:
        p = Path(env_path)
        if p.exists() and os.access(p, os.X_OK):
            return p
        logger.warning(
            "INFERALL_VLLM_PYTHON=%s does not exist or is not executable", env_path,
        )

    # 2. Default bootstrap location
    default = DEFAULT_VENV_PATH / "bin" / "python"
    if default.exists() and os.access(default, os.X_OK):
        return default

    raise VLLMNotInstalled(
        "vLLM is not installed in any known location.\n"
        "\n"
        "Install it with:\n"
        "  inferall vllm install\n"
        "\n"
        "Or set INFERALL_VLLM_PYTHON to a Python interpreter that has vllm:\n"
        "  export INFERALL_VLLM_PYTHON=/path/to/vllm-venv/bin/python\n"
    )


def is_vllm_available() -> bool:
    """Cheap check — does a vllm runtime exist?"""
    try:
        find_vllm_python()
        return True
    except VLLMNotInstalled:
        return False


def install_vllm_venv(
    venv_path: Path = DEFAULT_VENV_PATH,
    vllm_version: str = DEFAULT_VLLM_VERSION,
    extra_index_url: Optional[str] = PYTORCH_CUDA_INDEX,
) -> Path:
    """
    Bootstrap an isolated venv with vllm installed.

    This is slow (~5 minutes — vllm pulls a lot of CUDA wheels) and downloads
    several GB. Intended to be invoked from a CLI command, not at runtime.

    Args:
        venv_path: Where to create the venv. Default ~/.cache/inferall/vllm-venv.
        vllm_version: Pinned vllm version to install.
        extra_index_url: PyTorch wheel index for matching CUDA build. Pass
            ``None`` to use only PyPI defaults.

    Returns:
        Path to the python binary inside the new venv.
    """
    venv_path = Path(venv_path).expanduser()
    venv_path.parent.mkdir(parents=True, exist_ok=True)

    py_bin = venv_path / "bin" / "python"

    # Create the venv if it doesn't exist
    if not py_bin.exists():
        logger.info("Creating vllm venv at %s", venv_path)
        # Use the system python that we're running under, since we know it
        # works on this machine. ``venv`` from stdlib avoids depending on
        # virtualenv being installed.
        subprocess.check_call(
            [sys.executable, "-m", "venv", str(venv_path)],
        )

    if not py_bin.exists():
        raise RuntimeError(f"venv creation failed: {py_bin} does not exist")

    # Upgrade pip first — old pip can hit dependency-resolution corner cases
    # with vllm's complex requirements.
    logger.info("Upgrading pip in vllm venv")
    subprocess.check_call(
        [str(py_bin), "-m", "pip", "install", "--upgrade", "pip"],
    )

    # Install vllm at the requested version
    install_cmd = [
        str(py_bin), "-m", "pip", "install", f"vllm=={vllm_version}",
    ]
    if extra_index_url:
        install_cmd.extend(["--extra-index-url", extra_index_url])

    logger.info("Installing vllm==%s into %s (this can take several minutes)",
                vllm_version, venv_path)
    subprocess.check_call(install_cmd)

    # Verify
    subprocess.check_call(
        [str(py_bin), "-c", "import vllm; print('vllm', vllm.__version__)"],
    )

    return py_bin


def remove_vllm_venv(venv_path: Path = DEFAULT_VENV_PATH) -> bool:
    """Delete the bootstrap venv. Returns True if it existed."""
    venv_path = Path(venv_path).expanduser()
    if venv_path.exists():
        shutil.rmtree(venv_path)
        return True
    return False

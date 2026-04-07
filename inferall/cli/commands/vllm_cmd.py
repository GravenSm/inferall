"""
vllm CLI subcommands
--------------------
Manage the isolated vLLM runtime + opt models into the vLLM backend.

Subcommands:
  inferall vllm install [--version 0.19.0]
      Bootstrap an isolated venv with vllm. Slow on first run.

  inferall vllm status
      Show whether vllm is installed and where.

  inferall vllm uninstall
      Delete the vllm venv.

  inferall vllm enable <model-id>
      Mark a model to use the vLLM backend on next load.

  inferall vllm disable <model-id>
      Revert a model to its format-default backend.
"""

from pathlib import Path

import typer
from rich.console import Console

from inferall.backends.vllm_runtime import (
    DEFAULT_VENV_PATH,
    DEFAULT_VLLM_VERSION,
    VLLMNotInstalled,
    find_vllm_python,
    install_vllm_venv,
    remove_vllm_venv,
)
from inferall.config import EngineConfig
from inferall.registry.registry import ModelRegistry

console = Console()


def vllm_install(
    version: str = typer.Option(
        DEFAULT_VLLM_VERSION, "--version", "-v",
        help="vLLM version to install (e.g. 0.19.0)",
    ),
    venv_path: Path = typer.Option(
        DEFAULT_VENV_PATH, "--venv",
        help="Where to create the isolated venv",
    ),
) -> None:
    """Bootstrap an isolated venv with vllm. Takes several minutes on first run."""
    console.print(f"[cyan]Installing vllm=={version} into {venv_path}[/cyan]")
    console.print(
        "[dim]This downloads several GB and can take 5-10 minutes. "
        "Logs stream below.[/dim]"
    )
    try:
        py_bin = install_vllm_venv(venv_path=venv_path, vllm_version=version)
    except Exception as e:
        console.print(f"[red]Install failed:[/red] {e}")
        raise typer.Exit(code=1)
    console.print(f"[green]vllm installed at:[/green] {py_bin}")
    console.print(
        "[dim]Use [bold]inferall vllm enable <model-id>[/bold] to opt a model "
        "into the vLLM backend.[/dim]"
    )


def vllm_status() -> None:
    """Show vLLM runtime status."""
    try:
        py_bin = find_vllm_python()
        console.print(f"[green]vLLM available at:[/green] {py_bin}")
    except VLLMNotInstalled as e:
        console.print(f"[yellow]vLLM not installed.[/yellow]")
        console.print(f"[dim]{e}[/dim]")


def vllm_uninstall(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete the bootstrap vllm venv. Does not affect models or settings."""
    if not yes:
        confirm = typer.confirm(f"Delete {DEFAULT_VENV_PATH}?")
        if not confirm:
            raise typer.Abort()
    if remove_vllm_venv():
        console.print(f"[green]Removed {DEFAULT_VENV_PATH}[/green]")
    else:
        console.print(f"[yellow]{DEFAULT_VENV_PATH} did not exist[/yellow]")


def vllm_enable(
    model_id: str = typer.Argument(..., help="Model ID to opt into vLLM backend"),
) -> None:
    """Mark a model to use the vLLM backend on next load."""
    config = EngineConfig.load()
    registry = ModelRegistry(config.registry_path)
    try:
        record = registry.get(model_id)
        if record is None:
            console.print(f"[red]Model '{model_id}' not found in registry.[/red]")
            raise typer.Exit(code=1)
        registry.set_preferred_engine(record.model_id, "vllm")
        console.print(
            f"[green]'{record.model_id}' will now use the vLLM backend.[/green]"
        )
    finally:
        registry.close()


def vllm_disable(
    model_id: str = typer.Argument(..., help="Model ID to revert to default backend"),
) -> None:
    """Revert a model to its format-default backend."""
    config = EngineConfig.load()
    registry = ModelRegistry(config.registry_path)
    try:
        record = registry.get(model_id)
        if record is None:
            console.print(f"[red]Model '{model_id}' not found in registry.[/red]")
            raise typer.Exit(code=1)
        registry.set_preferred_engine(record.model_id, None)
        console.print(
            f"[green]'{record.model_id}' will now use its default backend.[/green]"
        )
    finally:
        registry.close()

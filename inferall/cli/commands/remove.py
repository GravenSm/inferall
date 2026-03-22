"""
Remove Command
--------------
Remove a model from the registry and delete its local files.

Strategy (from plan):
1. Delete the ModelRecord from SQLite
2. Delete the directory at record.local_path (our copy)
3. Do NOT touch ~/.cache/huggingface/ (user's global HF cache)
4. Print the freed disk space
"""

import shutil

import typer
from rich.console import Console

from inferall.config import EngineConfig
from inferall.registry.registry import ModelRegistry

console = Console()


def remove(
    model_id: str = typer.Argument(help="Model to remove (HuggingFace ID)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Remove a model from the registry and delete local files."""
    config = EngineConfig.load()
    registry = ModelRegistry(config.registry_path)

    try:
        record = registry.get(model_id)
    except Exception:
        registry.close()
        raise

    if record is None:
        console.print(f"[red]Model '{model_id}' not found in registry.[/red]")
        registry.close()
        raise typer.Exit(1)

    # Show what will be removed
    size_gb = record.file_size_bytes / (1024 ** 3)
    console.print(f"\nModel:  [bold]{record.model_id}[/bold]")
    console.print(f"Format: {record.format.value}")
    console.print(f"Size:   {size_gb:.1f} GB")
    console.print(f"Path:   {record.local_path}")

    if not yes:
        confirm = typer.confirm("\nRemove this model?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            registry.close()
            return

    # Calculate actual disk usage before deletion
    freed_bytes = 0
    local_path = record.local_path
    if local_path.exists():
        freed_bytes = _dir_size(local_path)
        try:
            shutil.rmtree(local_path)
            console.print(f"[green]Deleted files:[/green] {local_path}")
        except OSError as e:
            console.print(f"[yellow]Warning: Could not delete files:[/yellow] {e}")
            console.print(f"[dim]You may need to manually delete: {local_path}[/dim]")
    else:
        console.print(f"[dim]Local files not found (already deleted?): {local_path}[/dim]")

    # Remove from registry
    registry.remove(model_id)
    registry.close()

    # Print freed space
    if freed_bytes > 0:
        freed_gb = freed_bytes / (1024 ** 3)
        if freed_gb >= 1:
            console.print(f"\n[green]Freed {freed_gb:.1f} GB[/green]")
        else:
            freed_mb = freed_bytes / (1024 ** 2)
            console.print(f"\n[green]Freed {freed_mb:.0f} MB[/green]")

    console.print(f"[green]Removed {model_id} from registry.[/green]")


def _dir_size(path) -> int:
    """Calculate total size of a directory in bytes."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except OSError:
        pass
    return total

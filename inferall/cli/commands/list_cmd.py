"""
List Command
------------
Show all pulled models in a Rich table.

Columns: Model ID, Format, Size, Revision, Last Used
"""

import typer
from rich.console import Console
from rich.table import Table

from inferall.config import EngineConfig
from inferall.registry.registry import ModelRegistry

console = Console()


def list_models() -> None:
    """List all pulled models."""
    config = EngineConfig.load()
    registry = ModelRegistry(config.registry_path)

    try:
        records = registry.list_all()
    finally:
        registry.close()

    if not records:
        console.print("[dim]No models pulled yet.[/dim]")
        console.print("Run: [bold]inferall pull <model_id>[/bold]")
        return

    table = Table(title="Pulled Models")
    table.add_column("Model", style="bold cyan", no_wrap=True)
    table.add_column("Task", style="magenta")
    table.add_column("Format", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Params", justify="right")
    table.add_column("Revision", style="dim")
    table.add_column("Last Used", style="dim")

    for r in records:
        # Format size
        size_gb = r.file_size_bytes / (1024 ** 3)
        size_str = f"{size_gb:.1f} GB" if size_gb >= 1 else f"{r.file_size_bytes / (1024**2):.0f} MB"

        # Format param count
        if r.param_count:
            if r.param_count >= 1_000_000_000:
                params_str = f"{r.param_count / 1_000_000_000:.1f}B"
            else:
                params_str = f"{r.param_count / 1_000_000:.0f}M"
        else:
            params_str = "-"

        # Format revision (first 7 chars)
        rev_str = r.revision[:7] if r.revision else "-"

        # Format last used
        if r.last_used_at:
            last_used_str = r.last_used_at.strftime("%Y-%m-%d %H:%M")
        else:
            last_used_str = "never"

        # Format name with variant
        name = r.model_id
        if r.gguf_variant:
            name += f" ({r.gguf_variant})"

        table.add_row(
            name,
            r.task.value,
            r.format.value,
            size_str,
            params_str,
            rev_str,
            last_used_str,
        )

    console.print(table)
    console.print(f"\n[dim]{len(records)} model(s)[/dim]")

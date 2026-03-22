"""
Status Command
--------------
Show GPU status and currently loaded models.

Displays:
- Per-GPU: name, VRAM usage, temperature, utilization, power
- Loaded models: model ID, backend, VRAM estimate, ref count, idle time
"""

import typer
from rich.console import Console
from rich.table import Table

from inferall.gpu.manager import GPUManager

console = Console()


def status() -> None:
    """Show GPU status and loaded model information."""
    gpu_manager = GPUManager()

    # GPU Info
    if gpu_manager.n_gpus == 0:
        console.print("[yellow]No GPUs detected.[/yellow]")
    else:
        gpu_table = Table(title="GPU Status")
        gpu_table.add_column("GPU", style="bold")
        gpu_table.add_column("Name", style="cyan")
        gpu_table.add_column("VRAM Used", justify="right")
        gpu_table.add_column("VRAM Free", justify="right", style="green")
        gpu_table.add_column("VRAM Total", justify="right")
        gpu_table.add_column("Temp", justify="right")
        gpu_table.add_column("Util", justify="right")
        gpu_table.add_column("Power", justify="right")

        for i in range(gpu_manager.n_gpus):
            try:
                stats = gpu_manager.get_gpu_stats(i)
                used_gb = stats.used_memory / (1024 ** 3)
                free_gb = stats.free_memory / (1024 ** 3)
                total_gb = stats.total_memory / (1024 ** 3)

                temp_str = f"{stats.temperature}°C" if stats.temperature is not None else "-"
                util_str = f"{stats.utilization}%" if stats.utilization is not None else "-"
                power_str = f"{stats.power_usage:.0f}W" if stats.power_usage is not None else "-"

                gpu_table.add_row(
                    str(i),
                    stats.name,
                    f"{used_gb:.1f} GB",
                    f"{free_gb:.1f} GB",
                    f"{total_gb:.1f} GB",
                    temp_str,
                    util_str,
                    power_str,
                )
            except Exception as e:
                gpu_table.add_row(str(i), f"[red]Error: {e}[/red]", "", "", "", "", "", "")

        console.print(gpu_table)

    # Model Allocations
    allocations = gpu_manager.gpu_assignments
    if allocations:
        console.print()
        alloc_table = Table(title="GPU Allocations")
        alloc_table.add_column("Model", style="bold cyan")
        alloc_table.add_column("GPU", justify="center")
        alloc_table.add_column("Estimated VRAM", justify="right")

        for name, alloc in sorted(allocations.items()):
            vram_gb = alloc.allocated_memory / (1024 ** 3)
            alloc_table.add_row(
                name,
                str(alloc.device_id),
                f"{vram_gb:.1f} GB",
            )

        console.print(alloc_table)
    else:
        console.print("\n[dim]No models currently loaded.[/dim]")

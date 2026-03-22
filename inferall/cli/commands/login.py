"""
Login Command
--------------
HuggingFace token management. Thin wrapper around huggingface_hub.login().

Usage:
    inferall login
"""

import typer
from rich.console import Console

console = Console()


def login():
    """Log in to HuggingFace Hub (for gated models like Llama)."""
    try:
        from huggingface_hub import login as hf_login
        console.print(
            "[bold]HuggingFace Login[/bold]\n"
            "This stores your token at ~/.cache/huggingface/token\n"
            "Get your token at: https://huggingface.co/settings/tokens\n"
        )
        hf_login()
        console.print("[bold green]Login successful![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Login failed:[/bold red] {e}")
        raise typer.Exit(1)

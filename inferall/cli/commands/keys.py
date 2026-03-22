"""
Keys Command
--------------
Manage API keys for multi-user access.

Usage:
    model_engine keys create --name "user1" --priority 1 --rpm 120
    model_engine keys list
    model_engine keys revoke me-abc123abcd
    model_engine keys usage me-abc123abcd
"""

import typer
from rich.console import Console
from rich.table import Table

from inferall.config import EngineConfig

console = Console()


def keys_create(
    name: str = typer.Option(..., "--name", "-n", help="Key name/label"),
    priority: int = typer.Option(0, "--priority", "-p", help="Priority level (0=normal, 1=high, 2=critical)"),
    rpm: int = typer.Option(60, "--rpm", help="Rate limit: requests per minute"),
    rpd: int = typer.Option(10000, "--rpd", help="Rate limit: requests per day"),
):
    """Create a new API key."""
    from inferall.auth.key_store import KeyStore

    config = EngineConfig.load()
    config.ensure_dirs()
    db_path = config.base_dir / "auth.db"
    store = KeyStore(str(db_path))

    raw_key = store.create_key(name=name, priority=priority,
                                rate_limit_rpm=rpm, rate_limit_rpd=rpd)
    store.close()

    console.print(f"\n[bold green]API Key Created[/bold green]")
    console.print(f"  Name:     {name}")
    console.print(f"  Priority: {priority}")
    console.print(f"  RPM:      {rpm}")
    console.print(f"  RPD:      {rpd}")
    console.print(f"\n  [bold]Key: {raw_key}[/bold]")
    console.print(f"\n  [yellow]Save this key — it won't be shown again![/yellow]\n")


def keys_list():
    """List all API keys."""
    from inferall.auth.key_store import KeyStore

    config = EngineConfig.load()
    db_path = config.base_dir / "auth.db"
    store = KeyStore(str(db_path))
    keys = store.list_keys()
    store.close()

    if not keys:
        console.print("[dim]No API keys created yet.[/dim]")
        return

    table = Table(title="API Keys")
    table.add_column("Prefix", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Priority")
    table.add_column("RPM", justify="right")
    table.add_column("RPD", justify="right")
    table.add_column("Status")

    for k in keys:
        status = "[green]active[/green]" if k["enabled"] else "[red]revoked[/red]"
        table.add_row(
            k["key_prefix"], k["name"], str(k["priority"]),
            str(k["rate_limit_rpm"]), str(k["rate_limit_rpd"]), status,
        )

    console.print(table)


def keys_revoke(
    key_prefix: str = typer.Argument(help="Key prefix to revoke (e.g., me-abc123abcd)"),
):
    """Revoke an API key."""
    from inferall.auth.key_store import KeyStore

    config = EngineConfig.load()
    db_path = config.base_dir / "auth.db"
    store = KeyStore(str(db_path))
    revoked = store.revoke_key(key_prefix)
    store.close()

    if revoked:
        console.print(f"[green]Revoked key: {key_prefix}[/green]")
    else:
        console.print(f"[red]Key not found: {key_prefix}[/red]")


def keys_usage(
    key_prefix: str = typer.Argument(help="Key prefix to check usage"),
    hours: int = typer.Option(24, "--hours", help="Lookback period in hours"),
):
    """Show usage stats for an API key."""
    from inferall.auth.key_store import KeyStore

    config = EngineConfig.load()
    db_path = config.base_dir / "auth.db"
    store = KeyStore(str(db_path))
    usage = store.get_usage(key_prefix, hours=hours)
    store.close()

    if "error" in usage:
        console.print(f"[red]{usage['error']}[/red]")
        return

    console.print(f"\nUsage for [bold]{key_prefix}[/bold] (last {hours}h):")
    console.print(f"  Requests: {usage['requests']}")
    console.print(f"  Tokens:   {usage['total_tokens']}")

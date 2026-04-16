"""
TUI Command
-----------
Launch the InferAll dashboard (Textual TUI) against a running server.

Same entry point as the standalone `inferall_dashboard` script, exposed as
`inferall tui` so users can discover it from the main CLI help.
"""

import typer

from inferall.tui.app import run_dashboard


def tui(
    url: str = typer.Option(
        "http://127.0.0.1:8000",
        "--url", "-u",
        help="URL of the running inferall server the dashboard should connect to.",
    ),
) -> None:
    """Launch the dashboard TUI connected to a running inferall server."""
    run_dashboard(server_url=url)

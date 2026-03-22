"""CLI root — Typer application."""

import typer

from inferall.cli.commands.list_cmd import list_models
from inferall.cli.commands.login import login
from inferall.cli.commands.pull import pull
from inferall.cli.commands.remove import remove
from inferall.cli.commands.run import run
from inferall.cli.commands.serve import serve
from inferall.cli.commands.status import status
from inferall.cli.commands.keys import keys_create, keys_list, keys_revoke, keys_usage

app = typer.Typer(
    name="inferall",
    help="Personal inference engine for HuggingFace models — chat, embeddings, vision, ASR, diffusion, TTS.",
    no_args_is_help=True,
)

app.command()(pull)
app.command()(login)
app.command()(run)
app.command()(serve)
app.command(name="list")(list_models)
app.command()(remove)
app.command()(status)

# Key management subcommands
keys_app = typer.Typer(name="keys", help="Manage API keys for multi-user access.")
keys_app.command(name="create")(keys_create)
keys_app.command(name="list")(keys_list)
keys_app.command(name="revoke")(keys_revoke)
keys_app.command(name="usage")(keys_usage)
app.add_typer(keys_app)


def main():
    app()

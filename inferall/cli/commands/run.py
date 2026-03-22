"""
Run Command — Interactive REPL
-------------------------------
Loads a model and starts an interactive chat session.

Features:
- Conversation history accumulates across turns
- /clear — reset conversation history
- /system <prompt> — set/change system prompt
- /params — show current generation parameters
- /exit or Ctrl+D — quit
- Rich-formatted output with markdown rendering
"""

import logging
import signal
import sys
import threading
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from inferall.backends.base import GenerationParams
from inferall.config import EngineConfig
from inferall.gpu.allocator import GPUAllocator
from inferall.gpu.manager import GPUManager
from inferall.orchestrator import ModelNotFoundError, Orchestrator
from inferall.registry.metadata import ModelTask
from inferall.registry.registry import ModelRegistry

logger = logging.getLogger(__name__)
console = Console()


def run(
    model_id: str = typer.Argument(help="Model to run (HuggingFace ID)"),
    system_prompt: Optional[str] = typer.Option(
        None, "--system", "-s", help="Initial system prompt",
    ),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    max_tokens: int = typer.Option(2048, "--max-tokens", help="Maximum tokens to generate"),
    top_p: float = typer.Option(0.9, "--top-p", help="Top-p sampling"),
    top_k: int = typer.Option(50, "--top-k", help="Top-k sampling"),
    repetition_penalty: float = typer.Option(1.1, "--repetition-penalty", help="Repetition penalty"),
) -> None:
    """Chat interactively with a model."""
    config = EngineConfig.load()
    config.ensure_dirs()

    # Build dependencies
    registry = ModelRegistry(config.registry_path)
    gpu_manager = GPUManager()
    allocator = GPUAllocator(gpu_manager, vram_buffer_mb=config.vram_buffer_mb)
    orchestrator = Orchestrator(config, registry, gpu_manager, allocator)

    # Generation params
    params = GenerationParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    # Check task type — interactive REPL is chat-only
    record = registry.get(model_id)
    if record and record.task not in (ModelTask.CHAT, ModelTask.VISION_LANGUAGE):
        task_hints = {
            ModelTask.EMBEDDING: "Use the API: POST /v1/embeddings",
            ModelTask.ASR: "Use the API: POST /v1/audio/transcriptions",
            ModelTask.DIFFUSION: "Use the API: POST /v1/images/generations",
            ModelTask.TTS: "Use the API: POST /v1/audio/speech",
        }
        hint = task_hints.get(record.task, "Use the API server instead.")
        console.print(
            f"\n[red]Model '{model_id}' is a {record.task.value} model, "
            f"not a chat model.[/red]\n{hint}"
        )
        registry.close()
        raise typer.Exit(1)

    # Load the model
    console.print(f"\n[bold]Loading {model_id}...[/bold]")
    try:
        loaded = orchestrator.get_or_load(model_id)
        orchestrator.release(model_id)  # Release the load ref; stream will acquire its own
    except ModelNotFoundError:
        console.print(
            f"\n[red]Model '{model_id}' not found in registry.[/red]\n"
            f"Run: [bold]inferall pull {model_id}[/bold]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Failed to load model:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[green]Model loaded.[/green] Type your message, or /help for commands.\n")

    # Conversation state
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        console.print(f"[dim]System prompt set.[/dim]\n")

    # REPL loop
    try:
        _repl_loop(orchestrator, model_id, messages, params)
    except KeyboardInterrupt:
        console.print("\n")
    finally:
        orchestrator.shutdown()
        registry.close()


def _repl_loop(
    orchestrator: Orchestrator,
    model_id: str,
    messages: list,
    params: GenerationParams,
) -> None:
    """Main REPL loop."""
    while True:
        try:
            user_input = _get_user_input()
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break

        if user_input is None:
            continue

        # Handle commands
        if user_input.startswith("/"):
            result = _handle_command(user_input, messages, params)
            if result == "exit":
                break
            continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Stream response
        console.print()
        response_text = _stream_response(orchestrator, model_id, messages, params)

        if response_text is not None:
            messages.append({"role": "assistant", "content": response_text})
        else:
            # Remove the user message if generation failed
            messages.pop()

        console.print()


def _get_user_input() -> Optional[str]:
    """Get user input, supporting multi-line with backslash continuation."""
    try:
        line = console.input("[bold blue]>>> [/bold blue]")
    except EOFError:
        raise

    lines = [line]

    # Multi-line continuation with backslash
    while lines[-1].endswith("\\"):
        lines[-1] = lines[-1][:-1]  # Remove trailing backslash
        try:
            continuation = console.input("[dim]... [/dim]")
        except EOFError:
            break
        lines.append(continuation)

    text = "\n".join(lines).strip()
    if not text:
        return None
    return text


def _handle_command(command: str, messages: list, params: GenerationParams) -> Optional[str]:
    """
    Handle REPL commands. Returns "exit" to quit, None otherwise.
    """
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit", "/q"):
        console.print("[dim]Goodbye![/dim]")
        return "exit"

    elif cmd == "/clear":
        # Keep system prompt if present
        system = [m for m in messages if m["role"] == "system"]
        messages.clear()
        messages.extend(system)
        console.print("[yellow]Conversation cleared.[/yellow]\n")

    elif cmd == "/system":
        if not arg:
            # Show current system prompt
            system = [m for m in messages if m["role"] == "system"]
            if system:
                console.print(f"[dim]Current system prompt:[/dim] {system[-1]['content']}\n")
            else:
                console.print("[dim]No system prompt set. Usage: /system <prompt>[/dim]\n")
        else:
            # Remove existing system messages
            messages[:] = [m for m in messages if m["role"] != "system"]
            messages.insert(0, {"role": "system", "content": arg})
            console.print(f"[yellow]System prompt updated.[/yellow]\n")

    elif cmd == "/params":
        console.print("[bold]Generation Parameters:[/bold]")
        console.print(f"  temperature:      {params.temperature}")
        console.print(f"  max_tokens:       {params.max_tokens}")
        console.print(f"  top_p:            {params.top_p}")
        console.print(f"  top_k:            {params.top_k}")
        console.print(f"  repetition_penalty: {params.repetition_penalty}")
        console.print()

    elif cmd == "/help":
        console.print("[bold]Commands:[/bold]")
        console.print("  /clear            Reset conversation history")
        console.print("  /system <prompt>  Set/show system prompt")
        console.print("  /params           Show generation parameters")
        console.print("  /exit             Quit (also Ctrl+D)")
        console.print("  /help             Show this help")
        console.print()
        console.print("[dim]Tip: End a line with \\ for multi-line input.[/dim]")
        console.print()

    else:
        console.print(f"[red]Unknown command: {cmd}[/red]. Type /help for available commands.\n")

    return None


def _stream_response(
    orchestrator: Orchestrator,
    model_id: str,
    messages: list,
    params: GenerationParams,
) -> Optional[str]:
    """Stream a response and return the full text, or None on error."""
    cancel = threading.Event()
    full_text = []

    # Handle Ctrl+C during generation
    original_handler = signal.getsignal(signal.SIGINT)

    def cancel_handler(sig, frame):
        cancel.set()
        console.print("\n[yellow]Generation cancelled.[/yellow]")

    try:
        signal.signal(signal.SIGINT, cancel_handler)

        for token in orchestrator.stream(model_id, messages, params, cancel):
            full_text.append(token)
            console.print(token, end="", highlight=False)

        # Print final newline
        console.print()

        text = "".join(full_text)
        if cancel.is_set():
            return text if text else None
        return text

    except ModelNotFoundError:
        console.print(f"\n[red]Model '{model_id}' was unloaded unexpectedly.[/red]")
        return None
    except Exception as e:
        console.print(f"\n[red]Generation error:[/red] {e}")
        logger.error("Generation error", exc_info=True)
        return None
    finally:
        signal.signal(signal.SIGINT, original_handler)

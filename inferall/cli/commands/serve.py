"""
Serve Command
-------------
Start the OpenAI-compatible API server.

Binds to localhost by default. Use --host 0.0.0.0 to expose to network.
API key can be set via INFERALL_API_KEY env var or --api-key flag.
"""

import logging
import os
from typing import Optional

import typer
from rich.console import Console

from inferall.config import EngineConfig

logger = logging.getLogger(__name__)
console = Console()


def serve(
    port: int = typer.Option(
        None, "--port", "-p",
        help="Port to listen on. Default: 8000 (override with INFERALL_PORT).",
    ),
    host: str = typer.Option(
        None, "--host",
        help=(
            "Address to bind to. Default: 127.0.0.1 (local-only). "
            "Use 0.0.0.0 to expose the server on your LAN — set an API key first."
        ),
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key",
        help=(
            "Require this API key on every request. Prefer the INFERALL_API_KEY "
            "env var so the key isn't visible in `ps`. Without this flag (and "
            "without the env var) the server runs without auth — fine for "
            "127.0.0.1, dangerous on 0.0.0.0."
        ),
    ),
    compat_mode: str = typer.Option(
        "strict", "--compat-mode",
        help=(
            "How to handle OpenAI request fields InferAll doesn't implement. "
            "'strict' returns 400 (recommended for development), 'lenient' "
            "silently strips them (use only if a third-party client sends "
            "fields you can't control)."
        ),
    ),
    workers: Optional[int] = typer.Option(
        None, "--workers",
        help=(
            "Number of inference worker threads. Default: 2. Increase for "
            "more concurrent requests across different models — within a "
            "single model concurrency is still capped by concurrency_per_model."
        ),
    ),
) -> None:
    """
    Start the OpenAI-compatible API server.

    By default the server binds to 127.0.0.1:8000 with no authentication —
    safe for local-only use. Models load on demand the first time a request
    asks for them, and unload after the idle timeout.

    [bold]Common patterns:[/bold]

      [cyan]inferall serve[/cyan]                              [dim]# local-only, no auth, port 8000[/dim]
      [cyan]inferall serve --port 8080[/cyan]                  [dim]# pick a different port[/dim]
      [cyan]INFERALL_API_KEY=secret inferall serve[/cyan]      [dim]# require auth (recommended)[/dim]
      [cyan]inferall serve --host 0.0.0.0[/cyan]               [dim]# expose to LAN — set a key first![/dim]
      [cyan]inferall serve --workers 4[/cyan]                  [dim]# more concurrent inference[/dim]

    The server stays in the foreground; press [bold]Ctrl+C[/bold] to stop.
    All loaded models are unloaded cleanly on shutdown.

    Once running, point any OpenAI SDK at [cyan]http://<host>:<port>/v1[/cyan]
    and your existing client code works without changes. Try it with:

      [cyan]curl http://localhost:8000/v1/models[/cyan]

    First time? Pull a model first with [cyan]inferall pull <model-id>[/cyan].
    """
    # Build config with CLI overrides
    overrides = {}
    if port is not None:
        overrides["default_port"] = port
    if host is not None:
        overrides["default_host"] = host
    if workers is not None:
        overrides["inference_workers"] = workers

    config = EngineConfig.load(cli_overrides=overrides if overrides else None)
    config.ensure_dirs()

    # Resolve API key: CLI flag > env var > config
    resolved_api_key = api_key or config.api_key

    if api_key and not os.environ.get("INFERALL_API_KEY"):
        console.print(
            "[yellow]Warning:[/yellow] API key passed via --api-key flag is visible "
            "in process list. Prefer: INFERALL_API_KEY=<key> inferall serve"
        )

    # Network exposure warning
    if config.default_host == "0.0.0.0":
        console.print(
            "[yellow]Warning:[/yellow] Binding to 0.0.0.0 exposes the server to your network. "
            "Ensure API key authentication is enabled for security."
        )

    # Build dependencies
    from inferall.gpu.allocator import GPUAllocator
    from inferall.gpu.manager import GPUManager
    from inferall.orchestrator import Orchestrator
    from inferall.registry.registry import ModelRegistry

    registry = ModelRegistry(config.registry_path)
    gpu_manager = GPUManager()
    allocator = GPUAllocator(gpu_manager, vram_buffer_mb=config.vram_buffer_mb)
    orchestrator = Orchestrator(config, registry, gpu_manager, allocator)

    # File store for Files API
    from inferall.registry.file_store import FileStore
    from inferall.registry.assistants_store import AssistantsStore
    from inferall.registry.jobs_store import FineTuningStore, BatchStore
    from inferall.scheduling.dispatcher import ModelDispatcher
    file_store = FileStore(registry.conn)
    assistants_store = AssistantsStore(registry.conn)
    fine_tuning_store = FineTuningStore(registry.conn)
    batch_store = BatchStore(registry.conn)

    # Multi-key auth (if auth.db exists)
    from inferall.auth.key_store import KeyStore
    key_store = None
    auth_db = config.base_dir / "auth.db"
    if auth_db.exists():
        key_store = KeyStore(str(auth_db))
        key_count = len(key_store.list_keys())
        if key_count > 0:
            console.print(f"  API Keys: [green]{key_count} configured[/green]")

    # Per-model request dispatcher
    model_dispatcher = ModelDispatcher(
        max_workers=config.inference_workers,
        max_concurrent=config.max_concurrent_requests,
        concurrency_per_model=config.concurrency_per_model,
        model_queue_size=config.model_queue_size,
    )

    # Create the app
    from inferall.api.server import create_app

    app = create_app(
        orchestrator=orchestrator,
        registry=registry,
        api_key=resolved_api_key,
        compat_mode=compat_mode,
        inference_workers=config.inference_workers,
        file_store=file_store,
        files_dir=config.files_dir,
        assistants_store=assistants_store,
        fine_tuning_store=fine_tuning_store,
        batch_store=batch_store,
        dispatcher=model_dispatcher,
        key_store=key_store,
    )

    # Print startup info
    base_url = f"http://{config.default_host}:{config.default_port}"
    console.print(f"\n[bold cyan]InferAll API Server[/bold cyan]")
    console.print(f"  URL:         [bold]{base_url}[/bold]  [dim](OpenAI base_url: {base_url}/v1)[/dim]")
    console.print(
        f"  Auth:        "
        + ("[green]enabled[/green]" if resolved_api_key else "[yellow]disabled[/yellow] (local-only)")
    )
    console.print(f"  Workers:     {config.inference_workers}")
    console.print(f"  Compat mode: {compat_mode}")

    # List available models
    records = registry.list_all()
    if records:
        console.print(f"\n  [bold]Available models ({len(records)}):[/bold]")
        for r in records:
            engine_marker = ""
            if getattr(r, "preferred_engine", None) == "vllm":
                engine_marker = "  [green](vllm)[/green]"
            console.print(f"    • {r.model_id} [dim]({r.format.value})[/dim]{engine_marker}")
    else:
        console.print(
            "\n  [yellow]No models pulled yet.[/yellow] "
            "Pull one to get started:\n"
            "    [cyan]inferall pull Qwen/Qwen2.5-1.5B-Instruct[/cyan]   [dim]# small chat model[/dim]\n"
            "    [cyan]inferall pull llama3.1[/cyan]                      [dim]# from Ollama registry[/dim]"
        )

    # Quick-test hint — the single most useful thing a first-time user
    # can do to confirm the server is reachable
    console.print(
        f"\n  [dim]Test it:[/dim] "
        f"[cyan]curl {base_url}/v1/models[/cyan]"
    )
    console.print(f"  [dim]Stop:[/dim]    [cyan]Ctrl+C[/cyan]")
    console.print()

    # Start uvicorn
    import uvicorn
    uvicorn.run(
        app,
        host=config.default_host,
        port=config.default_port,
        log_level="info",
    )

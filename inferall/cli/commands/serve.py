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
    port: int = typer.Option(None, "--port", "-p", help="Port to listen on"),
    host: str = typer.Option(None, "--host", help="Host to bind to"),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="API key for authentication (prefer INFERALL_API_KEY env var)",
    ),
    compat_mode: str = typer.Option(
        "strict", "--compat-mode",
        help="Compatibility mode: 'strict' (reject unsupported) or 'lenient' (strip with warning)",
    ),
    workers: Optional[int] = typer.Option(
        None, "--workers", help="Inference thread pool size",
    ),
) -> None:
    """Start the OpenAI-compatible API server."""
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
    )

    # Print startup info
    console.print(f"\n[bold]InferAll API Server[/bold]")
    console.print(f"  Listening: http://{config.default_host}:{config.default_port}")
    console.print(f"  Auth: {'[green]enabled[/green]' if resolved_api_key else '[yellow]disabled[/yellow]'}")
    console.print(f"  Compat mode: {compat_mode}")
    console.print(f"  Workers: {config.inference_workers}")

    # List available models
    records = registry.list_all()
    if records:
        console.print(f"\n  Available models ({len(records)}):")
        for r in records:
            console.print(f"    - {r.model_id} ({r.format.value})")
    else:
        console.print("\n  [yellow]No models pulled yet.[/yellow] Run: inferall pull <model>")

    console.print()

    # Start uvicorn
    import uvicorn
    uvicorn.run(
        app,
        host=config.default_host,
        port=config.default_port,
        log_level="info",
    )

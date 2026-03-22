"""
Pull Command
-------------
Download a model from HuggingFace Hub or Ollama registry.

Auto-detects source:
- Names with "/" that look like HF IDs → HuggingFace
- Short names without "/" or with "ollama://" → Ollama registry

Usage:
    inferall pull meta-llama/Llama-3-8B-Instruct          # HuggingFace
    inferall pull TheBloke/Mistral-7B-GGUF --variant Q4_K_M
    inferall pull llama3.1                                  # Ollama
    inferall pull llama3.1:70b                              # Ollama with tag
    inferall pull --source ollama codellama                 # Force Ollama
    inferall pull --source hf meta-llama/Llama-3-8B-Instruct
"""

import logging
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from inferall.config import EngineConfig
from inferall.registry.hf_resolver import HFResolver, UnsupportedModelError
from inferall.registry.ollama_resolver import OllamaResolver, OllamaResolverError
from inferall.registry.registry import ModelRegistry

logger = logging.getLogger(__name__)
console = Console()


def _detect_source(model_id: str, source_hint: Optional[str]) -> str:
    """
    Detect whether a model name refers to HuggingFace or Ollama.

    Returns "ollama" or "hf".
    """
    if source_hint:
        return source_hint

    # Explicit ollama:// prefix
    if model_id.startswith("ollama://"):
        return "ollama"

    # HuggingFace IDs always have org/model format with typically
    # longer names and no colons (except in rare cases)
    # Ollama names are short: "llama3.1", "llama3.1:70b", "codellama"
    if "/" in model_id:
        # Could be "user/model" (Ollama) or "org/model-name" (HF)
        # HF models typically have dashes/underscores in names
        parts = model_id.split("/")
        if len(parts) == 2:
            # If the name part contains typical HF patterns, assume HF
            name_part = parts[1]
            if "-" in name_part or "_" in name_part or len(name_part) > 20:
                return "hf"
            # Short names like "user/model" could be either — default to HF
            # since Ollama defaults namespace to "library"
            return "hf"
        return "hf"  # 3+ parts = definitely HF

    # No slash — short name = Ollama
    return "ollama"


def pull(
    model_id: str = typer.Argument(
        ..., help="Model name — HuggingFace ID or Ollama name (e.g., llama3.1)"
    ),
    variant: Optional[str] = typer.Option(
        None, "--variant", "-v",
        help="GGUF variant name (e.g., Q4_K_M) — HuggingFace only",
    ),
    trust_remote_code: bool = typer.Option(
        False, "--trust-remote-code",
        help="Allow model to execute custom code (security risk)",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Bypass validation / re-pull existing",
    ),
    source: Optional[str] = typer.Option(
        None, "--source", "-s",
        help="Force source: 'hf' (HuggingFace) or 'ollama'",
    ),
):
    """Download a model from HuggingFace Hub or Ollama registry."""
    config = EngineConfig.load()
    config.ensure_dirs()

    registry = ModelRegistry(db_path=config.registry_path)

    # Strip ollama:// prefix if present
    clean_id = model_id.removeprefix("ollama://")

    # Detect source
    detected_source = _detect_source(model_id, source)

    # For Ollama, the registry key is "ollama://namespace/model:tag"
    if detected_source == "ollama":
        _pull_ollama(clean_id, registry, config, force)
    else:
        _pull_hf(clean_id, registry, config, variant, trust_remote_code, force)

    registry.close()


def _pull_ollama(
    model_name: str,
    registry: ModelRegistry,
    config,
    force: bool,
) -> None:
    """Pull from Ollama registry."""
    resolver = OllamaResolver(models_dir=config.models_dir)

    # Build the canonical ID for registry lookup
    parts = model_name.split("/", 1)
    if len(parts) == 1:
        namespace, rest = "library", parts[0]
    else:
        namespace, rest = parts[0], parts[1]
    if ":" not in rest:
        rest += ":latest"
    lookup_id = f"ollama://{namespace}/{rest}"

    # Check if already pulled
    existing = registry.get(lookup_id)
    if existing and not force:
        console.print(
            f"[green]Model '{model_name}' is up to date "
            f"(rev {existing.revision[:12]}).[/green]\n"
            f"  Use as: [bold]{existing.model_id}[/bold]\n"
            f"  Or just: [bold]{model_name}[/bold]\n"
            f"Use --force to re-pull."
        )
        return

    try:
        console.print(f"[bold blue]Pulling from Ollama:[/bold blue] {model_name}")
        record = resolver.pull(model_name)

        # If cloud model, ensure API key is available
        if record.format.value == "ollama_cloud":
            _ensure_ollama_api_key(config)

        registry.register(record)

        console.print(
            f"\n[bold green]Pulled[/bold green] {record.model_id} "
            f"({record.format.value}, "
            f"{record.file_size_bytes / 1024**3:.2f} GB, "
            f"rev {record.revision})"
        )
        console.print(f"  Task: [magenta]{record.task.value}[/magenta]")
        if record.format.value == "ollama_cloud":
            console.print(f"  Source: [cyan]Ollama Cloud (remote)[/cyan]")
        else:
            console.print(f"  Source: [cyan]Ollama registry[/cyan]")
        if record.gguf_variant:
            console.print(f"  Quantization: {record.gguf_variant}")

    except OllamaResolverError as e:
        console.print(f"[bold red]Ollama pull failed:[/bold red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Pull failed:[/bold red] {e}")
        logger.error("Ollama pull failed for %s", model_name, exc_info=True)
        raise typer.Exit(1)


def _pull_hf(
    model_id: str,
    registry: ModelRegistry,
    config,
    variant: Optional[str],
    trust_remote_code: bool,
    force: bool,
) -> None:
    """Pull from HuggingFace Hub."""
    resolver = HFResolver(models_dir=config.models_dir)

    # Check if already pulled
    existing = registry.get(model_id)
    if existing and not force:
        try:
            from huggingface_hub import model_info as get_model_info
            remote_info = get_model_info(model_id)
            remote_rev = remote_info.sha or "unknown"
        except Exception as e:
            logger.debug("Could not check remote revision: %s", e)
            remote_rev = None

        if remote_rev and remote_rev != existing.revision:
            console.print(
                f"[yellow]Update available for '{model_id}':[/yellow]\n"
                f"  Local:  {existing.revision[:12]}\n"
                f"  Remote: {remote_rev[:12]}"
            )
            if typer.confirm("Download update?"):
                force = True
            else:
                return
        else:
            console.print(
                f"[green]Model '{model_id}' is up to date "
                f"(rev {existing.revision[:12]}).[/green]"
            )
            return

    try:
        with console.status(f"[bold blue]Pulling {model_id}...[/bold blue]"):
            record = resolver.pull(
                model_id=model_id,
                variant=variant,
                trust_remote_code=trust_remote_code,
                force=force,
            )

        registry.register(record)

        console.print(
            f"[bold green]Pulled[/bold green] {model_id} "
            f"({record.format.value}, "
            f"{record.file_size_bytes / 1024**3:.2f} GB, "
            f"rev {record.revision[:12]})"
        )
        console.print(f"  Task: [magenta]{record.task.value}[/magenta]")

        if record.gguf_variant:
            console.print(f"  GGUF variant: {record.gguf_variant}")
        if trust_remote_code:
            console.print("  [yellow]trust_remote_code enabled[/yellow]")

    except UnsupportedModelError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Pull failed:[/bold red] {e}")
        logger.error("Pull failed for %s", model_id, exc_info=True)
        raise typer.Exit(1)


def _ensure_ollama_api_key(config: EngineConfig) -> None:
    """
    Ensure an Ollama API key is available for cloud models.
    Checks env var and config, prompts if missing, saves for future use.
    """
    # Already have a key?
    if os.environ.get("OLLAMA_API_KEY") or config.ollama_api_key:
        return

    console.print(
        "\n[yellow]This is a cloud model — it requires an Ollama API key.[/yellow]"
    )
    console.print(
        "[dim]Get your key at: https://ollama.com/settings/keys[/dim]\n"
    )

    key = typer.prompt("Ollama API key").strip()
    if not key:
        console.print("[red]No API key provided. Cloud models won't work without one.[/red]")
        return

    # Set for current session
    os.environ["OLLAMA_API_KEY"] = key

    # Offer to save it
    if typer.confirm("Save this key for future sessions?", default=True):
        _save_ollama_api_key(config, key)
        console.print("[green]API key saved.[/green]\n")
    else:
        console.print(
            "[dim]Key set for this session only. "
            "Set OLLAMA_API_KEY env var to persist it.[/dim]\n"
        )


def _save_ollama_api_key(config: EngineConfig, key: str) -> None:
    """Save the Ollama API key to the config file."""
    import yaml

    config_file = config.base_dir / "config.yaml"
    config.base_dir.mkdir(parents=True, exist_ok=True)

    # Load existing config or start fresh
    existing = {}
    if config_file.exists():
        try:
            with open(config_file) as f:
                existing = yaml.safe_load(f) or {}
        except Exception:
            pass

    existing["ollama_api_key"] = key

    with open(config_file, "w") as f:
        yaml.dump(existing, f, default_flow_style=False)

    # Also restrict file permissions (key is sensitive)
    try:
        config_file.chmod(0o600)
    except OSError:
        pass

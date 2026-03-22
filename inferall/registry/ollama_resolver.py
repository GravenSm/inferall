"""
Ollama Registry Resolver
-------------------------
Downloads models from the Ollama registry (registry.ollama.ai).

Protocol: Docker Registry V2 with Ollama-specific media types.
Models are GGUF files wrapped in OCI manifests.

Name resolution:
  "llama3.1"           → registry.ollama.ai/library/llama3.1:latest
  "llama3.1:70b"       → registry.ollama.ai/library/llama3.1:70b
  "user/model:tag"     → registry.ollama.ai/user/model:tag
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from inferall.registry.metadata import (
    ModelFormat,
    ModelRecord,
    ModelTask,
)

logger = logging.getLogger(__name__)

_REGISTRY_BASE = "https://registry.ollama.ai"
_MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"

# Ollama layer media types
_MEDIA_MODEL = "application/vnd.ollama.image.model"
_MEDIA_TEMPLATE = "application/vnd.ollama.image.template"
_MEDIA_SYSTEM = "application/vnd.ollama.image.system"
_MEDIA_PARAMS = "application/vnd.ollama.image.params"
_MEDIA_LICENSE = "application/vnd.ollama.image.license"
_MEDIA_CONFIG = "application/vnd.docker.container.image.v1+json"


class OllamaResolverError(Exception):
    """Raised when an Ollama pull fails."""
    pass


class OllamaResolver:
    """Downloads models from the Ollama registry."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

    def pull(self, model_name: str) -> ModelRecord:
        """
        Pull a model from the Ollama registry.

        Args:
            model_name: Ollama model name (e.g., "llama3.1", "llama3.1:70b")

        Returns:
            ModelRecord for the downloaded model
        """
        namespace, model, tag = self._parse_name(model_name)
        canonical = f"{namespace}/{model}:{tag}"

        logger.info("Pulling from Ollama registry: %s", canonical)

        # Fetch manifest
        manifest = self._fetch_manifest(namespace, model, tag)
        manifest_digest = manifest.get("_digest", "unknown")

        # Parse layers (null means cloud-only model)
        raw_layers = manifest.get("layers") or []
        layers = {l["mediaType"]: l for l in raw_layers}

        # Fetch config for metadata
        config_blob = manifest.get("config", {})

        config_data = self._fetch_config(namespace, model, config_blob)

        # Check for cloud-only models (no local weights)
        model_layer = layers.get(_MEDIA_MODEL)
        if model_layer is None:
            remote_host = config_data.get("remote_host", "")
            if remote_host:
                return self._register_cloud_model(
                    canonical, namespace, model, tag,
                    manifest_digest, config_data,
                )
            raise OllamaResolverError(
                f"No GGUF model layer found in manifest for {canonical}"
            )

        # Build local path
        local_dir = self.models_dir / "ollama" / namespace / model / tag
        local_dir.mkdir(parents=True, exist_ok=True)

        # Download GGUF blob
        gguf_path = local_dir / f"{model}-{tag}.gguf"
        if gguf_path.exists() and gguf_path.stat().st_size == model_layer["size"]:
            logger.info("GGUF already downloaded: %s", gguf_path)
        else:
            self._download_blob(
                namespace, model,
                model_layer["digest"],
                gguf_path,
                model_layer["size"],
            )

        # Fetch and save auxiliary data
        metadata = self._fetch_auxiliary(namespace, model, layers, local_dir)

        # Extract model info from config
        model_family = config_data.get("model_family", "unknown")
        model_type = config_data.get("model_type", "")
        file_type = config_data.get("file_type", "")

        # Estimate param count from model_type (e.g., "8.0B" → 8_000_000_000)
        param_count = self._parse_param_count(model_type)

        # Determine task — most Ollama models are chat, some are embeddings
        task = ModelTask.CHAT
        if "embed" in model.lower():
            task = ModelTask.EMBEDDING

        # Build the display name (ollama://namespace/model:tag)
        model_id = f"ollama://{namespace}/{model}:{tag}"

        record = ModelRecord(
            model_id=model_id,
            revision=manifest_digest[:12],
            format=ModelFormat.GGUF,
            local_path=local_dir,
            file_size_bytes=model_layer["size"],
            param_count=param_count,
            gguf_variant=file_type or None,
            trust_remote_code=False,
            pipeline_tag="text-generation",
            pulled_at=datetime.now(),
            task=task,
        )

        # Save metadata JSON for reference
        meta_path = local_dir / "ollama_metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "source": "ollama",
                "canonical_name": canonical,
                "model_family": model_family,
                "model_type": model_type,
                "file_type": file_type,
                "manifest_digest": manifest_digest,
                "template": metadata.get("template"),
                "system": metadata.get("system"),
                "params": metadata.get("params"),
            }, f, indent=2)

        logger.info(
            "Pulled %s (%s, %s, %.2f GB)",
            model_id, model_family, file_type,
            model_layer["size"] / 1024**3,
        )
        return record

    # -------------------------------------------------------------------------
    # Name Resolution
    # -------------------------------------------------------------------------

    def _parse_name(self, name: str) -> Tuple[str, str, str]:
        """
        Parse an Ollama model name into (namespace, model, tag).

        "llama3.1"         → ("library", "llama3.1", "latest")
        "llama3.1:70b"     → ("library", "llama3.1", "70b")
        "user/model:tag"   → ("user", "model", "tag")
        """
        # Split tag
        if ":" in name:
            name_part, tag = name.rsplit(":", 1)
        else:
            name_part, tag = name, "latest"

        # Split namespace/model
        if "/" in name_part:
            namespace, model = name_part.split("/", 1)
        else:
            namespace, model = "library", name_part

        return namespace, model, tag

    # -------------------------------------------------------------------------
    # Registry API
    # -------------------------------------------------------------------------

    def _fetch_manifest(self, namespace: str, model: str, tag: str) -> dict:
        """Fetch the OCI manifest for a model."""
        url = f"{_REGISTRY_BASE}/v2/{namespace}/{model}/manifests/{tag}"
        logger.debug("Fetching manifest: %s", url)

        req = Request(url, headers={"Accept": _MANIFEST_ACCEPT})
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                # Store digest from header
                data["_digest"] = resp.headers.get(
                    "docker-content-digest",
                    resp.headers.get("ollama-content-digest", "unknown"),
                )
                return data
        except HTTPError as e:
            if e.code == 404:
                raise OllamaResolverError(
                    f"Model not found: {namespace}/{model}:{tag}"
                )
            raise OllamaResolverError(
                f"Failed to fetch manifest: HTTP {e.code} {e.reason}"
            )

    def _fetch_config(self, namespace: str, model: str, config_ref: dict) -> dict:
        """Fetch the config blob (model metadata)."""
        digest = config_ref.get("digest", "")
        if not digest:
            return {}

        url = f"{_REGISTRY_BASE}/v2/{namespace}/{model}/blobs/{digest}"
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as e:
            logger.debug("Could not fetch config blob: %s", e)
            return {}

    def _download_blob(
        self,
        namespace: str,
        model: str,
        digest: str,
        dest: Path,
        expected_size: int,
    ) -> None:
        """Download a blob with progress logging."""
        url = f"{_REGISTRY_BASE}/v2/{namespace}/{model}/blobs/{digest}"
        logger.info(
            "Downloading: %s (%.2f GB)",
            dest.name, expected_size / 1024**3,
        )

        req = Request(url)

        # Support resuming
        temp_path = dest.with_suffix(".partial")
        start_byte = 0
        if temp_path.exists():
            start_byte = temp_path.stat().st_size
            if start_byte < expected_size:
                req.add_header("Range", f"bytes={start_byte}-")
                logger.info("Resuming from byte %d", start_byte)
            else:
                start_byte = 0  # Re-download

        try:
            with urlopen(req, timeout=60) as resp:
                mode = "ab" if start_byte > 0 else "wb"
                downloaded = start_byte
                with open(temp_path, mode) as f:
                    while True:
                        chunk = resp.read(8 * 1024 * 1024)  # 8MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Progress every 500MB
                        if downloaded % (500 * 1024**2) < len(chunk):
                            pct = downloaded / expected_size * 100
                            logger.info(
                                "Download progress: %.1f%% (%.2f / %.2f GB)",
                                pct, downloaded / 1024**3, expected_size / 1024**3,
                            )
        except Exception as e:
            logger.error("Download failed at %d bytes: %s", downloaded, e)
            raise OllamaResolverError(f"Download failed: {e}")

        # Verify size
        actual_size = temp_path.stat().st_size
        if actual_size != expected_size:
            logger.warning(
                "Size mismatch: expected %d, got %d", expected_size, actual_size
            )

        # Rename to final path
        temp_path.rename(dest)
        logger.info("Downloaded: %s", dest)

    # -------------------------------------------------------------------------
    # Auxiliary Data
    # -------------------------------------------------------------------------

    def _fetch_auxiliary(
        self,
        namespace: str,
        model: str,
        layers: dict,
        local_dir: Path,
    ) -> dict:
        """Fetch template, system prompt, params, and license."""
        result = {}

        for media_type, key, filename in [
            (_MEDIA_TEMPLATE, "template", "template.txt"),
            (_MEDIA_SYSTEM, "system", "system.txt"),
            (_MEDIA_PARAMS, "params", "params.json"),
            (_MEDIA_LICENSE, "license", "LICENSE"),
        ]:
            layer = layers.get(media_type)
            if layer is None:
                continue

            try:
                url = f"{_REGISTRY_BASE}/v2/{namespace}/{model}/blobs/{layer['digest']}"
                req = Request(url)
                with urlopen(req, timeout=30) as resp:
                    data = resp.read()

                # Save to file
                (local_dir / filename).write_bytes(data)

                # Parse for return
                if key == "params":
                    try:
                        result[key] = json.loads(data)
                    except json.JSONDecodeError:
                        result[key] = data.decode("utf-8", errors="replace")
                else:
                    result[key] = data.decode("utf-8", errors="replace")

                logger.debug("Fetched %s (%d bytes)", key, len(data))
            except Exception as e:
                logger.debug("Could not fetch %s: %s", key, e)

        return result

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _register_cloud_model(
        self,
        canonical: str,
        namespace: str,
        model: str,
        tag: str,
        manifest_digest: str,
        config_data: dict,
    ) -> ModelRecord:
        """Register a cloud-only model (no local weights)."""
        remote_host = config_data.get("remote_host", "https://ollama.com:443")
        remote_model = config_data.get("remote_model", model)
        capabilities = config_data.get("capabilities", [])
        context_length = config_data.get("context_length", 4096)

        model_id = f"ollama://{namespace}/{model}:{tag}"

        # Save cloud metadata locally
        local_dir = self.models_dir / "ollama" / namespace / model / tag
        local_dir.mkdir(parents=True, exist_ok=True)

        meta_path = local_dir / "ollama_metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "source": "ollama_cloud",
                "canonical_name": canonical,
                "remote_host": remote_host,
                "remote_model": remote_model,
                "capabilities": capabilities,
                "context_length": context_length,
                "manifest_digest": manifest_digest,
            }, f, indent=2)

        logger.info(
            "Registered cloud model %s (remote: %s, model: %s, capabilities: %s)",
            model_id, remote_host, remote_model, capabilities,
        )

        return ModelRecord(
            model_id=model_id,
            revision=manifest_digest[:12],
            format=ModelFormat.OLLAMA_CLOUD,
            local_path=local_dir,
            file_size_bytes=0,
            param_count=None,
            gguf_variant=None,
            trust_remote_code=False,
            pipeline_tag="text-generation",
            pulled_at=datetime.now(),
            task=ModelTask.CHAT,
        )

    def _parse_param_count(self, model_type: str) -> Optional[int]:
        """Parse parameter count from Ollama model_type string (e.g., '8.0B')."""
        if not model_type:
            return None

        model_type = model_type.strip().upper()
        try:
            if model_type.endswith("B"):
                return int(float(model_type[:-1]) * 1_000_000_000)
            if model_type.endswith("M"):
                return int(float(model_type[:-1]) * 1_000_000)
        except (ValueError, IndexError):
            pass
        return None

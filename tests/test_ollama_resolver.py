"""Tests for Ollama registry resolver — name parsing, metadata, integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inferall.registry.ollama_resolver import OllamaResolver, OllamaResolverError


class TestNameParsing:
    def test_simple_name(self):
        resolver = OllamaResolver(models_dir=Path("/tmp"))
        ns, model, tag = resolver._parse_name("llama3.1")
        assert ns == "library"
        assert model == "llama3.1"
        assert tag == "latest"

    def test_name_with_tag(self):
        resolver = OllamaResolver(models_dir=Path("/tmp"))
        ns, model, tag = resolver._parse_name("llama3.1:70b")
        assert ns == "library"
        assert model == "llama3.1"
        assert tag == "70b"

    def test_name_with_namespace(self):
        resolver = OllamaResolver(models_dir=Path("/tmp"))
        ns, model, tag = resolver._parse_name("myuser/mymodel:v2")
        assert ns == "myuser"
        assert model == "mymodel"
        assert tag == "v2"

    def test_name_with_namespace_no_tag(self):
        resolver = OllamaResolver(models_dir=Path("/tmp"))
        ns, model, tag = resolver._parse_name("myuser/mymodel")
        assert ns == "myuser"
        assert model == "mymodel"
        assert tag == "latest"


class TestParamCount:
    def test_billions(self):
        resolver = OllamaResolver(models_dir=Path("/tmp"))
        assert resolver._parse_param_count("8.0B") == 8_000_000_000

    def test_millions(self):
        resolver = OllamaResolver(models_dir=Path("/tmp"))
        assert resolver._parse_param_count("135M") == 135_000_000

    def test_empty(self):
        resolver = OllamaResolver(models_dir=Path("/tmp"))
        assert resolver._parse_param_count("") is None

    def test_invalid(self):
        resolver = OllamaResolver(models_dir=Path("/tmp"))
        assert resolver._parse_param_count("unknown") is None


class TestSourceDetection:
    def test_short_name_is_ollama(self):
        from inferall.cli.commands.pull import _detect_source
        assert _detect_source("llama3.1", None) == "ollama"
        assert _detect_source("codellama", None) == "ollama"
        assert _detect_source("llama3.1:70b", None) == "ollama"

    def test_hf_style_is_hf(self):
        from inferall.cli.commands.pull import _detect_source
        assert _detect_source("meta-llama/Llama-3-8B-Instruct", None) == "hf"
        assert _detect_source("sentence-transformers/all-MiniLM-L6-v2", None) == "hf"

    def test_explicit_ollama_prefix(self):
        from inferall.cli.commands.pull import _detect_source
        assert _detect_source("ollama://llama3.1", None) == "ollama"

    def test_forced_source(self):
        from inferall.cli.commands.pull import _detect_source
        assert _detect_source("llama3.1", "hf") == "hf"
        assert _detect_source("meta-llama/Llama-3-8B", "ollama") == "ollama"

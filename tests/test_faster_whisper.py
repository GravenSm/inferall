"""Tests for faster-whisper (CTranslate2) ASR path — detection, allocator, dispatch, backend."""

import io
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# =============================================================================
# Helpers (mirrors tests/test_hf_resolver.py's _make_model_info)
# =============================================================================

def _make_model_info(pipeline_tag=None, tags=None, library_name=None, filenames=None, sha="abc123"):
    info = MagicMock()
    info.pipeline_tag = pipeline_tag
    info.tags = tags or []
    info.library_name = library_name
    info.sha = sha
    info.siblings = []
    if filenames:
        for f in filenames:
            sib = MagicMock()
            sib.rfilename = f
            info.siblings.append(sib)
    return info


def _make_resolver():
    from inferall.registry.hf_resolver import HFResolver
    return HFResolver(models_dir=Path("/tmp/inferall-tests"))


# =============================================================================
# Metadata / Enum Tests
# =============================================================================

class TestFasterWhisperMetadata:
    def test_format_enum_exists(self):
        from inferall.registry.metadata import ModelFormat
        assert ModelFormat.FASTER_WHISPER.value == "faster_whisper"

    def test_format_maps_to_asr_task(self):
        from inferall.registry.metadata import FORMAT_TO_TASK, ModelFormat, ModelTask
        assert FORMAT_TO_TASK[ModelFormat.FASTER_WHISPER] == ModelTask.ASR

    def test_db_roundtrip(self):
        """Registry row serialisation survives through the FASTER_WHISPER value."""
        from inferall.registry.metadata import ModelFormat, ModelRecord, ModelTask
        rec = ModelRecord(
            model_id="Systran/faster-whisper-large-v3",
            revision="abc",
            format=ModelFormat.FASTER_WHISPER,
            local_path=Path("/tmp/fw"),
            file_size_bytes=0,
            param_count=None,
            gguf_variant=None,
            trust_remote_code=False,
            pipeline_tag="automatic-speech-recognition",
            pulled_at=datetime.now(),
            task=ModelTask.ASR,
        )
        row = rec.to_db_row()
        assert row["format"] == "faster_whisper"
        round_tripped = ModelRecord.from_db_row(row)
        assert round_tripped.format == ModelFormat.FASTER_WHISPER
        assert round_tripped.task == ModelTask.ASR


# =============================================================================
# Allocator
# =============================================================================

class TestAllocatorFasterWhisperFormat:
    def test_bytes_per_param_has_faster_whisper(self):
        from inferall.gpu.allocator import _BYTES_PER_PARAM
        from inferall.registry.metadata import ModelFormat
        assert ModelFormat.FASTER_WHISPER in _BYTES_PER_PARAM
        # CT2 float16 default — should match the regular ASR entry
        assert _BYTES_PER_PARAM[ModelFormat.FASTER_WHISPER] == 2.0


# =============================================================================
# HF Resolver Detection
# =============================================================================

class TestHFResolverFasterWhisperDetection:
    def test_ctranslate2_library_routes_to_faster_whisper(self):
        """Primary signal: library_name == 'ctranslate2' (what Systran publishes)."""
        from inferall.registry.metadata import ModelFormat
        resolver = _make_resolver()
        info = _make_model_info(
            pipeline_tag="automatic-speech-recognition",
            library_name="ctranslate2",
            tags=["audio", "automatic-speech-recognition"],
        )
        fmt, gguf = resolver._detect_format("Systran/faster-whisper-large-v3", info, variant=None)
        assert fmt == ModelFormat.FASTER_WHISPER
        assert gguf is None

    def test_ctranslate2_tag_routes_to_faster_whisper(self):
        """Fallback signal: 'ctranslate2' present only in tags."""
        from inferall.registry.metadata import ModelFormat
        resolver = _make_resolver()
        info = _make_model_info(
            pipeline_tag="automatic-speech-recognition",
            library_name=None,
            tags=["ctranslate2", "audio"],
        )
        fmt, _ = resolver._detect_format("some/ct2-whisper", info, variant=None)
        assert fmt == ModelFormat.FASTER_WHISPER

    def test_asr_without_ctranslate2_still_resolves_to_asr(self):
        """Regression guard: openai/whisper-* and similar stay on transformers."""
        from inferall.registry.metadata import ModelFormat
        resolver = _make_resolver()
        info = _make_model_info(
            pipeline_tag="automatic-speech-recognition",
            library_name="transformers",
            tags=["transformers", "whisper", "audio"],
        )
        fmt, _ = resolver._detect_format("openai/whisper-large-v3", info, variant=None)
        assert fmt == ModelFormat.ASR

    def test_asr_with_missing_library_field_still_resolves_to_asr(self):
        """HF metadata may omit library_name; default should remain ASR."""
        from inferall.registry.metadata import ModelFormat
        resolver = _make_resolver()
        info = _make_model_info(
            pipeline_tag="automatic-speech-recognition",
            library_name=None,
            tags=[],
        )
        fmt, _ = resolver._detect_format("test/whisper", info, variant=None)
        assert fmt == ModelFormat.ASR

    def test_ctranslate2_casing_insensitive(self):
        """library_name sometimes comes through capitalised/mixed case."""
        from inferall.registry.metadata import ModelFormat
        resolver = _make_resolver()
        info = _make_model_info(
            pipeline_tag="automatic-speech-recognition",
            library_name="CTranslate2",
        )
        fmt, _ = resolver._detect_format("test/ct2-whisper", info, variant=None)
        assert fmt == ModelFormat.FASTER_WHISPER

    def test_ctranslate2_only_applies_to_asr_pipeline(self):
        """A repo tagged ctranslate2 with a non-ASR pipeline tag should NOT hit
        the FASTER_WHISPER branch (that branch is scoped to ASR pipelines)."""
        from inferall.registry.metadata import ModelFormat
        resolver = _make_resolver()
        info = _make_model_info(
            pipeline_tag="text-generation",
            library_name="ctranslate2",
            tags=["ctranslate2"],
            filenames=["config.json", "model.bin"],
        )
        fmt, _ = resolver._detect_format("some/ct2-llm", info, variant=None)
        assert fmt != ModelFormat.FASTER_WHISPER


# =============================================================================
# Backend properties
# =============================================================================

class TestFasterWhisperBackendProperties:
    def test_name(self):
        from inferall.backends.faster_whisper_backend import FasterWhisperBackend
        assert FasterWhisperBackend().name == "faster_whisper"

    def test_select_device_and_dtype_gpu(self):
        from inferall.backends.faster_whisper_backend import FasterWhisperBackend
        from inferall.gpu.allocator import AllocationPlan
        plan = AllocationPlan(gpu_ids=[0])
        device, dtype = FasterWhisperBackend()._select_device_and_dtype(plan)
        assert device == "cuda"
        assert dtype == "float16"

    def test_select_device_and_dtype_cpu(self):
        from inferall.backends.faster_whisper_backend import FasterWhisperBackend
        from inferall.gpu.allocator import AllocationPlan
        plan = AllocationPlan(gpu_ids=[])
        device, dtype = FasterWhisperBackend()._select_device_and_dtype(plan)
        assert device == "cpu"
        assert dtype == "int8"


# =============================================================================
# Load path: surfaces a clear error when faster-whisper isn't installed
# =============================================================================

class TestFasterWhisperMissingDependency:
    def test_missing_faster_whisper_raises_clear_runtime_error(self, monkeypatch):
        """
        Simulate faster-whisper not being installed. load() should raise a
        RuntimeError that points the user at the [asr] extra — not the
        default ImportError traceback.
        """
        import builtins
        from inferall.backends.faster_whisper_backend import FasterWhisperBackend
        from inferall.gpu.allocator import AllocationPlan
        from inferall.registry.metadata import ModelFormat, ModelRecord, ModelTask

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "faster_whisper":
                raise ImportError("No module named 'faster_whisper'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        rec = ModelRecord(
            model_id="Systran/faster-whisper-large-v3",
            revision="abc",
            format=ModelFormat.FASTER_WHISPER,
            local_path=Path("/tmp/fw-test"),
            file_size_bytes=0,
            param_count=None,
            gguf_variant=None,
            trust_remote_code=False,
            pipeline_tag="automatic-speech-recognition",
            pulled_at=datetime.now(),
            task=ModelTask.ASR,
        )
        plan = AllocationPlan(gpu_ids=[])

        with pytest.raises(RuntimeError) as exc:
            FasterWhisperBackend().load(rec, plan)
        assert "faster-whisper" in str(exc.value).lower()
        assert "pip install" in str(exc.value).lower()


# =============================================================================
# Transcribe path (with a mocked faster-whisper model)
# =============================================================================

class TestFasterWhisperTranscribeMocked:
    def _build_loaded(self, segments, info_language="en", info_duration=3.21):
        from inferall.backends.base import LoadedModel

        fake_model = MagicMock()
        fake_info = MagicMock()
        fake_info.language = info_language
        fake_info.duration = info_duration
        fake_model.transcribe.return_value = (iter(segments), fake_info)
        return LoadedModel(
            model_id="Systran/faster-whisper-large-v3",
            backend_name="faster_whisper",
            model=fake_model,
            tokenizer=None,
        )

    def _segment(self, idx, start, end, text):
        seg = MagicMock()
        seg.id = idx
        seg.start = start
        seg.end = end
        seg.text = text
        return seg

    def test_joins_segment_text(self):
        from inferall.backends.base import TranscriptionParams
        from inferall.backends.faster_whisper_backend import FasterWhisperBackend

        loaded = self._build_loaded([
            self._segment(0, 0.0, 1.5, "Hello "),
            self._segment(1, 1.5, 3.0, "world."),
        ])
        result = FasterWhisperBackend().transcribe(loaded, b"\x00\x00", TranscriptionParams())
        assert result.text == "Hello world."

    def test_returns_detected_language_and_duration(self):
        from inferall.backends.base import TranscriptionParams
        from inferall.backends.faster_whisper_backend import FasterWhisperBackend

        loaded = self._build_loaded(
            [self._segment(0, 0.0, 2.0, "bonjour")],
            info_language="fr",
            info_duration=2.0,
        )
        result = FasterWhisperBackend().transcribe(loaded, b"\x00\x00", TranscriptionParams())
        assert result.language == "fr"
        assert result.duration == 2.0

    def test_verbose_json_includes_segments(self):
        from inferall.backends.base import TranscriptionParams
        from inferall.backends.faster_whisper_backend import FasterWhisperBackend

        loaded = self._build_loaded([
            self._segment(0, 0.0, 1.0, "one"),
            self._segment(1, 1.0, 2.0, "two"),
        ])
        params = TranscriptionParams(response_format="verbose_json")
        result = FasterWhisperBackend().transcribe(loaded, b"\x00\x00", params)
        assert result.segments is not None
        assert len(result.segments) == 2
        assert result.segments[0]["text"] == "one"

    def test_non_verbose_json_omits_segments(self):
        from inferall.backends.base import TranscriptionParams
        from inferall.backends.faster_whisper_backend import FasterWhisperBackend

        loaded = self._build_loaded([self._segment(0, 0.0, 1.0, "solo")])
        result = FasterWhisperBackend().transcribe(loaded, b"\x00\x00", TranscriptionParams())
        assert result.segments is None

    def test_passes_language_and_task_to_model(self):
        from inferall.backends.base import TranscriptionParams
        from inferall.backends.faster_whisper_backend import FasterWhisperBackend

        loaded = self._build_loaded([self._segment(0, 0.0, 1.0, "x")])
        params = TranscriptionParams(language="ja", task="translate")
        FasterWhisperBackend().transcribe(loaded, b"\x00\x00", params)
        call_kwargs = loaded.model.transcribe.call_args.kwargs
        assert call_kwargs["language"] == "ja"
        assert call_kwargs["task"] == "translate"


# =============================================================================
# Orchestrator dispatch
# =============================================================================

class TestOrchestratorFasterWhisperDispatch:
    def test_get_backend_routes_faster_whisper_format(self):
        from inferall.backends.faster_whisper_backend import FasterWhisperBackend
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.metadata import ModelFormat
        from inferall.registry.registry import ModelRegistry

        config = EngineConfig(idle_timeout=0)
        registry = MagicMock(spec=ModelRegistry)
        gpu_manager = MagicMock(spec=GPUManager)
        gpu_manager.gpu_assignments = {}
        allocator = MagicMock(spec=GPUAllocator)

        orch = Orchestrator(config, registry, gpu_manager, allocator)
        backend = orch._get_backend(ModelFormat.FASTER_WHISPER)
        assert isinstance(backend, FasterWhisperBackend)
        assert backend.name == "faster_whisper"

    def test_asr_format_still_routes_to_whisper_backend(self):
        """Regression guard — ASR-format models must NOT get the CT2 backend."""
        from inferall.backends.asr_backend import WhisperBackend
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.metadata import ModelFormat
        from inferall.registry.registry import ModelRegistry

        config = EngineConfig(idle_timeout=0)
        registry = MagicMock(spec=ModelRegistry)
        gpu_manager = MagicMock(spec=GPUManager)
        gpu_manager.gpu_assignments = {}
        allocator = MagicMock(spec=GPUAllocator)

        orch = Orchestrator(config, registry, gpu_manager, allocator)
        backend = orch._get_backend(ModelFormat.ASR)
        assert isinstance(backend, WhisperBackend)

    def test_format_from_backend_name_round_trips(self):
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.metadata import ModelFormat
        from inferall.registry.registry import ModelRegistry

        config = EngineConfig(idle_timeout=0)
        registry = MagicMock(spec=ModelRegistry)
        gpu_manager = MagicMock(spec=GPUManager)
        gpu_manager.gpu_assignments = {}
        allocator = MagicMock(spec=GPUAllocator)

        orch = Orchestrator(config, registry, gpu_manager, allocator)
        assert orch._format_from_backend_name("faster_whisper") == ModelFormat.FASTER_WHISPER


# =============================================================================
# Registry v7 migration — reclassify ASR → FASTER_WHISPER on disk signature
# =============================================================================

class TestLooksLikeCt2Helper:
    """Direct tests for the on-disk CTranslate2 signature helper."""

    def test_positive_ct2_layout(self, tmp_path):
        from inferall.registry.registry import _looks_like_ct2_on_disk
        (tmp_path / "model.bin").write_bytes(b"\x00")
        (tmp_path / "config.json").write_text(
            '{"multilingual": true, "suppress_ids": [1, 2, 3], "alignment_heads": []}'
        )
        assert _looks_like_ct2_on_disk(tmp_path) is True

    def test_transformers_layout_has_model_type(self, tmp_path):
        """Standard HF Whisper has model_type set, so shouldn't match."""
        from inferall.registry.registry import _looks_like_ct2_on_disk
        (tmp_path / "model.bin").write_bytes(b"\x00")
        (tmp_path / "config.json").write_text(
            '{"model_type": "whisper", "vocab_size": 51865}'
        )
        assert _looks_like_ct2_on_disk(tmp_path) is False

    def test_transformers_layout_uses_safetensors_no_model_bin(self, tmp_path):
        """Modern HF weights are model.safetensors; no model.bin → not CT2."""
        from inferall.registry.registry import _looks_like_ct2_on_disk
        (tmp_path / "model.safetensors").write_bytes(b"\x00")
        (tmp_path / "config.json").write_text('{"multilingual": true}')
        assert _looks_like_ct2_on_disk(tmp_path) is False

    def test_missing_dir(self, tmp_path):
        from inferall.registry.registry import _looks_like_ct2_on_disk
        assert _looks_like_ct2_on_disk(tmp_path / "does-not-exist") is False

    def test_missing_config_json(self, tmp_path):
        from inferall.registry.registry import _looks_like_ct2_on_disk
        (tmp_path / "model.bin").write_bytes(b"\x00")
        assert _looks_like_ct2_on_disk(tmp_path) is False

    def test_missing_model_bin(self, tmp_path):
        from inferall.registry.registry import _looks_like_ct2_on_disk
        (tmp_path / "config.json").write_text('{"multilingual": true}')
        assert _looks_like_ct2_on_disk(tmp_path) is False

    def test_malformed_config_json(self, tmp_path):
        from inferall.registry.registry import _looks_like_ct2_on_disk
        (tmp_path / "model.bin").write_bytes(b"\x00")
        (tmp_path / "config.json").write_text("{not valid json")
        assert _looks_like_ct2_on_disk(tmp_path) is False

    def test_config_is_array_not_object(self, tmp_path):
        """Defensive: config.json that isn't a dict shouldn't crash the check."""
        from inferall.registry.registry import _looks_like_ct2_on_disk
        (tmp_path / "model.bin").write_bytes(b"\x00")
        (tmp_path / "config.json").write_text("[1, 2, 3]")
        assert _looks_like_ct2_on_disk(tmp_path) is False


class TestV7ReclassifyMigration:
    """End-to-end migration tests: build a registry, run the sweep, verify state."""

    def _ct2_dir(self, base: Path, name: str) -> Path:
        """Create a plausible CT2 Whisper model directory on disk."""
        model_dir = base / name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.bin").write_bytes(b"\x00" * 16)
        (model_dir / "config.json").write_text(
            '{"multilingual": true, "suppress_ids": [1, 2, 3]}'
        )
        return model_dir

    def _hf_whisper_dir(self, base: Path, name: str) -> Path:
        """Create a transformers-format Whisper model directory on disk."""
        model_dir = base / name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 16)
        (model_dir / "config.json").write_text(
            '{"model_type": "whisper", "vocab_size": 51865}'
        )
        return model_dir

    def _make_asr_record(self, model_id, local_path):
        from inferall.registry.metadata import ModelFormat, ModelRecord, ModelTask
        return ModelRecord(
            model_id=model_id,
            revision="abc",
            format=ModelFormat.ASR,
            local_path=local_path,
            file_size_bytes=0,
            param_count=None,
            gguf_variant=None,
            trust_remote_code=False,
            pipeline_tag="automatic-speech-recognition",
            pulled_at=datetime.now(),
            task=ModelTask.ASR,
        )

    def test_reclassifies_ct2_model(self, tmp_path):
        """CT2 on disk + format='asr' → becomes 'faster_whisper'."""
        from inferall.registry.metadata import ModelFormat
        from inferall.registry.registry import ModelRegistry

        ct2_dir = self._ct2_dir(tmp_path, "faster-whisper-large-v3")

        db_path = tmp_path / "registry.db"
        reg = ModelRegistry(db_path)
        try:
            # Simulate a pre-v7 row: register with format=ASR but disk = CT2
            reg.register(self._make_asr_record(
                "Systran/faster-whisper-large-v3", ct2_dir
            ))
        finally:
            reg.close()

        # Force the migration to run again by resetting the schema_version
        # to 6, then reopen: v7 will re-run and inspect the disk.
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM schema_version WHERE version = 7")
        conn.commit()
        conn.close()

        reg2 = ModelRegistry(db_path)
        try:
            record = reg2.get("Systran/faster-whisper-large-v3")
            assert record is not None
            assert record.format == ModelFormat.FASTER_WHISPER
        finally:
            reg2.close()

    def test_leaves_transformers_whisper_alone(self, tmp_path):
        """Transformers-format Whisper stays ASR."""
        from inferall.registry.metadata import ModelFormat
        from inferall.registry.registry import ModelRegistry

        hf_dir = self._hf_whisper_dir(tmp_path, "whisper-large-v3")

        db_path = tmp_path / "registry.db"
        reg = ModelRegistry(db_path)
        try:
            reg.register(self._make_asr_record("openai/whisper-large-v3", hf_dir))
        finally:
            reg.close()

        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM schema_version WHERE version = 7")
        conn.commit()
        conn.close()

        reg2 = ModelRegistry(db_path)
        try:
            record = reg2.get("openai/whisper-large-v3")
            assert record is not None
            assert record.format == ModelFormat.ASR
        finally:
            reg2.close()

    def test_leaves_asr_with_missing_files_alone(self, tmp_path):
        """If local files are gone (user cleaned them up), don't guess."""
        from inferall.registry.metadata import ModelFormat
        from inferall.registry.registry import ModelRegistry

        ghost_dir = tmp_path / "ghost-model"
        # deliberately do not create it

        db_path = tmp_path / "registry.db"
        reg = ModelRegistry(db_path)
        try:
            reg.register(self._make_asr_record("some/ghost-model", ghost_dir))
        finally:
            reg.close()

        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM schema_version WHERE version = 7")
        conn.commit()
        conn.close()

        reg2 = ModelRegistry(db_path)
        try:
            record = reg2.get("some/ghost-model")
            assert record is not None
            assert record.format == ModelFormat.ASR
        finally:
            reg2.close()

    def test_does_not_affect_non_asr_rows(self, tmp_path):
        """Migration is scoped to format='asr'; other formats untouched."""
        from inferall.registry.metadata import ModelFormat, ModelRecord, ModelTask
        from inferall.registry.registry import ModelRegistry

        # Deliberately craft a weird case: a row with format=TRANSFORMERS but
        # pointed at a CT2-looking directory. Migration must NOT touch it,
        # because the WHERE clause only matches ASR rows.
        ct2_dir = self._ct2_dir(tmp_path, "mislabeled")

        db_path = tmp_path / "registry.db"
        reg = ModelRegistry(db_path)
        try:
            reg.register(ModelRecord(
                model_id="some/mislabeled",
                revision="abc",
                format=ModelFormat.TRANSFORMERS,
                local_path=ct2_dir,
                file_size_bytes=0,
                param_count=None,
                gguf_variant=None,
                trust_remote_code=False,
                pipeline_tag="text-generation",
                pulled_at=datetime.now(),
                task=ModelTask.CHAT,
            ))
        finally:
            reg.close()

        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM schema_version WHERE version = 7")
        conn.commit()
        conn.close()

        reg2 = ModelRegistry(db_path)
        try:
            record = reg2.get("some/mislabeled")
            assert record is not None
            assert record.format == ModelFormat.TRANSFORMERS
        finally:
            reg2.close()

    def test_mixed_registry_reclassifies_only_ct2_asr_rows(self, tmp_path):
        """End-to-end: several ASR rows, only CT2 ones get reclassified."""
        from inferall.registry.metadata import ModelFormat
        from inferall.registry.registry import ModelRegistry

        ct2_a = self._ct2_dir(tmp_path, "ct2_a")
        ct2_b = self._ct2_dir(tmp_path, "ct2_b")
        hf_c = self._hf_whisper_dir(tmp_path, "hf_c")

        db_path = tmp_path / "registry.db"
        reg = ModelRegistry(db_path)
        try:
            reg.register(self._make_asr_record("Systran/faster-whisper-large-v3", ct2_a))
            reg.register(self._make_asr_record("Systran/faster-whisper-medium", ct2_b))
            reg.register(self._make_asr_record("openai/whisper-large-v3", hf_c))
        finally:
            reg.close()

        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM schema_version WHERE version = 7")
        conn.commit()
        conn.close()

        reg2 = ModelRegistry(db_path)
        try:
            assert reg2.get("Systran/faster-whisper-large-v3").format == ModelFormat.FASTER_WHISPER
            assert reg2.get("Systran/faster-whisper-medium").format == ModelFormat.FASTER_WHISPER
            assert reg2.get("openai/whisper-large-v3").format == ModelFormat.ASR
        finally:
            reg2.close()

    def test_schema_version_advances_to_7(self, tmp_path):
        """Fresh registry should end up at schema_version = 7."""
        from inferall.registry.registry import ModelRegistry
        reg = ModelRegistry(tmp_path / "fresh.db")
        try:
            assert reg.get_schema_version() == 7
        finally:
            reg.close()

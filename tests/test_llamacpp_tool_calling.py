"""Tests for GGUF (llama.cpp) tool calling and the shared tool-parsing helpers."""

import json
from unittest.mock import MagicMock

import pytest

from inferall.backends.base import GenerationParams, ToolCall
from inferall.backends.tool_parsing import (
    ThinkingStreamFilter,
    ToolCallStreamExtractor,
    normalize_tool_call_messages,
    parse_tool_calls,
    strip_thinking,
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}

QWEN_CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "London"}}\n</tool_call>'


def _split(text: str, n: int = 3):
    """Split text into chunks of n characters, to simulate token streaming."""
    return [text[i:i + n] for i in range(0, len(text), n)]


# =============================================================================
# parse_tool_calls / strip_thinking / normalize
# =============================================================================

class TestParseToolCalls:
    def test_qwen_format(self):
        calls, remaining = parse_tool_calls(QWEN_CALL)
        assert len(calls) == 1
        assert calls[0].function_name == "get_weather"
        assert json.loads(calls[0].function_arguments) == {"city": "London"}
        assert remaining == ""

    def test_qwen_multiple_with_text(self):
        text = (
            "Let me check.\n"
            '<tool_call>\n{"name": "f1", "arguments": {"a": 1}}\n</tool_call>\n'
            '<tool_call>\n{"name": "f2", "arguments": {"b": 2}}\n</tool_call>'
        )
        calls, remaining = parse_tool_calls(text)
        assert [c.function_name for c in calls] == ["f1", "f2"]
        assert remaining == "Let me check."

    def test_mistral_format(self):
        calls, _ = parse_tool_calls('[TOOL_CALLS] [{"name": "search", "arguments": {"q": "x"}}]')
        assert calls[0].function_name == "search"

    def test_llama_format(self):
        calls, _ = parse_tool_calls('<function=get_weather>{"city": "Paris"}</function>')
        assert calls[0].function_name == "get_weather"

    def test_no_tool_calls(self):
        calls, remaining = parse_tool_calls("The weather in London is sunny.")
        assert calls is None
        assert remaining == "The weather in London is sunny."

    def test_invalid_json_inside_tags_is_left_alone(self):
        text = "<tool_call>not json</tool_call>"
        calls, remaining = parse_tool_calls(text)
        assert calls is None
        assert remaining == text

    def test_ids_are_unique(self):
        text = QWEN_CALL + QWEN_CALL
        calls, _ = parse_tool_calls(text)
        assert calls[0].id != calls[1].id
        assert calls[0].id.startswith("call_")


class TestStripThinking:
    def test_strips_closed_block(self):
        assert strip_thinking("<think>\nhmm\n</think>\n\nHello") == "Hello"

    def test_strips_unclosed_block(self):
        assert strip_thinking("<think>still going") == ""

    def test_keeps_plain_text(self):
        assert strip_thinking("Hello") == "Hello"

    def test_think_then_tool_call(self):
        text = "<think>I should call the tool</think>\n" + QWEN_CALL
        calls, remaining = parse_tool_calls(strip_thinking(text))
        assert calls[0].function_name == "get_weather"
        assert remaining == ""


class TestNormalizeToolCallMessages:
    def test_string_arguments_become_dicts(self):
        messages = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city": "London"}'}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]
        out = normalize_tool_call_messages(messages)
        assert out[1]["tool_calls"][0]["function"]["arguments"] == {"city": "London"}
        # Other messages are passed through untouched
        assert out[0] is messages[0]
        assert out[2] is messages[2]

    def test_original_is_not_mutated(self):
        messages = [{"role": "assistant", "tool_calls": [
            {"function": {"name": "f", "arguments": '{"a": 1}'}},
        ]}]
        normalize_tool_call_messages(messages)
        assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'

    def test_dict_arguments_untouched(self):
        messages = [{"role": "assistant", "tool_calls": [
            {"function": {"name": "f", "arguments": {"a": 1}}},
        ]}]
        out = normalize_tool_call_messages(messages)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}

    def test_invalid_json_left_as_string(self):
        messages = [{"role": "assistant", "tool_calls": [
            {"function": {"name": "f", "arguments": "{not json"}},
        ]}]
        out = normalize_tool_call_messages(messages)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == "{not json"

    def test_empty_string_arguments_become_empty_dict(self):
        messages = [{"role": "assistant", "tool_calls": [
            {"function": {"name": "f", "arguments": ""}},
        ]}]
        out = normalize_tool_call_messages(messages)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == {}


# =============================================================================
# Streaming filters
# =============================================================================

class TestThinkingStreamFilter:
    def _run(self, chunks):
        f = ThinkingStreamFilter()
        out = "".join(f.feed(c) for c in chunks)
        return out + f.flush()

    def test_passthrough(self):
        assert self._run(["Hel", "lo", " world"]) == "Hello world"

    def test_strips_block_split_across_tokens(self):
        text = "<think>\nreasoning here\n</think>\n\nThe answer is 42."
        assert self._run(_split(text, 2)) == "The answer is 42."

    def test_single_token(self):
        assert self._run(["<think>x</think>Hi"]) == "Hi"

    def test_partial_open_tag_is_held_then_released(self):
        # "<th" could be the start of <think>; must not be emitted early but
        # must be released once it turns out to be ordinary text.
        f = ThinkingStreamFilter()
        assert f.feed("a <th") == "a "
        assert f.feed("ing") == "<thing"
        assert f.flush() == ""

    def test_unclosed_block_is_dropped(self):
        assert self._run(["<think>", "never ", "closed"]) == ""

    def test_text_before_think_is_kept(self):
        assert self._run(_split("Sure. <think>x</think> Done.", 4)) == "Sure.  Done."


class TestToolCallStreamExtractor:
    def _run(self, chunks):
        ex = ToolCallStreamExtractor()
        events = []
        for c in chunks:
            events.extend(ex.feed(c))
        events.extend(ex.flush())
        return events

    def test_plain_content_passthrough(self):
        events = self._run(["Hello", " world"])
        assert events == [("content", "Hello"), ("content", " world")]

    def test_single_call_split_across_tokens(self):
        events = self._run(_split(QWEN_CALL, 3))
        kinds = [k for k, _ in events]
        assert kinds == ["tool_call"]
        call = events[0][1]
        assert isinstance(call, ToolCall)
        assert call.function_name == "get_weather"
        assert json.loads(call.function_arguments) == {"city": "London"}

    def test_content_before_and_after(self):
        text = "Checking. " + QWEN_CALL + " Done."
        events = self._run(_split(text, 5))
        content = "".join(p for k, p in events if k == "content")
        calls = [p for k, p in events if k == "tool_call"]
        assert content == "Checking.  Done."
        assert len(calls) == 1

    def test_multiple_calls(self):
        text = (
            '<tool_call>{"name": "f1", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "f2", "arguments": {"x": 1}}</tool_call>'
        )
        events = self._run(_split(text, 7))
        assert [p.function_name for k, p in events if k == "tool_call"] == ["f1", "f2"]
        assert not any(k == "content" and p.strip() for k, p in events)

    def test_partial_open_tag_not_leaked_early(self):
        ex = ToolCallStreamExtractor()
        assert list(ex.feed("Hi <tool")) == [("content", "Hi ")]
        assert list(ex.feed("_call>")) == []
        events = list(ex.feed('{"name": "f", "arguments": {}}</tool_call>'))
        assert len(events) == 1 and events[0][0] == "tool_call"
        assert events[0][1].function_name == "f"
        assert list(ex.flush()) == []

    def test_partial_open_tag_then_plain_text(self):
        events = self._run(["see <tool", "box"])
        assert "".join(p for k, p in events if k == "content") == "see <toolbox"

    def test_unterminated_call_is_salvaged(self):
        events = self._run(['<tool_call>{"name": "f", "arguments": {"a": 1}}'])
        assert events[0][0] == "tool_call"
        assert events[0][1].function_name == "f"

    def test_unterminated_garbage_returned_verbatim(self):
        events = self._run(["<tool_call>oops"])
        assert events == [("content", "<tool_call>oops")]

    def test_bad_json_inside_tags_returned_verbatim(self):
        events = self._run(["<tool_call>not json</tool_call>"])
        assert events == [("content", "<tool_call>not json</tool_call>")]


# =============================================================================
# LlamaCppBackend
# =============================================================================

def _loaded_with_response(content, tool_calls=None, finish_reason="stop"):
    loaded = MagicMock()
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    loaded.model.create_chat_completion.return_value = {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 15},
    }
    return loaded


def _loaded_with_stream(contents):
    loaded = MagicMock()
    loaded.model.create_chat_completion.return_value = iter(
        {"choices": [{"delta": {"content": c}}]} for c in contents
    )
    return loaded


class TestLlamaCppGenerate:
    def setup_method(self):
        from inferall.backends.llamacpp_backend import LlamaCppBackend
        self.backend = LlamaCppBackend()
        self.messages = [{"role": "user", "content": "Weather in London?"}]

    def test_text_tool_call_is_parsed(self):
        loaded = _loaded_with_response(QWEN_CALL)
        result = self.backend.generate(
            loaded, self.messages, GenerationParams(tools=[WEATHER_TOOL]),
        )
        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function_name == "get_weather"
        assert json.loads(result.tool_calls[0].function_arguments) == {"city": "London"}
        assert result.text == ""
        assert result.prompt_tokens == 20

    def test_tool_choice_auto_still_parses(self):
        loaded = _loaded_with_response(QWEN_CALL)
        result = self.backend.generate(
            loaded, self.messages,
            GenerationParams(tools=[WEATHER_TOOL], tool_choice="auto"),
        )
        assert result.finish_reason == "tool_calls"
        kwargs = loaded.model.create_chat_completion.call_args.kwargs
        assert kwargs["tools"] == [WEATHER_TOOL]
        assert kwargs["tool_choice"] == "auto"

    def test_thinking_is_stripped_before_parsing(self):
        loaded = _loaded_with_response("<think>\nuse the tool\n</think>\n" + QWEN_CALL)
        result = self.backend.generate(
            loaded, self.messages, GenerationParams(tools=[WEATHER_TOOL]),
        )
        assert result.finish_reason == "tool_calls"
        assert result.text == ""

    def test_thinking_is_stripped_without_tools(self):
        loaded = _loaded_with_response("<think>hmm</think>\n\nIt is sunny.")
        result = self.backend.generate(loaded, self.messages, GenerationParams())
        assert result.text == "It is sunny."
        assert result.finish_reason == "stop"
        assert result.tool_calls is None

    def test_plain_answer_with_tools_available(self):
        loaded = _loaded_with_response("I don't need a tool for that.")
        result = self.backend.generate(
            loaded, self.messages, GenerationParams(tools=[WEATHER_TOOL]),
        )
        assert result.finish_reason == "stop"
        assert result.tool_calls is None
        assert result.text == "I don't need a tool for that."

    def test_no_parsing_when_no_tools_requested(self):
        # Without tools, tag-like text is ordinary content.
        loaded = _loaded_with_response(QWEN_CALL)
        result = self.backend.generate(loaded, self.messages, GenerationParams())
        assert result.tool_calls is None
        assert "<tool_call>" in result.text

    def test_tool_choice_none_omits_tools(self):
        loaded = _loaded_with_response("Sunny.")
        self.backend.generate(
            loaded, self.messages,
            GenerationParams(tools=[WEATHER_TOOL], tool_choice="none"),
        )
        kwargs = loaded.model.create_chat_completion.call_args.kwargs
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    def test_structured_tool_calls_from_runtime(self):
        # llama-cpp-python returns these itself when tool_choice names a function.
        loaded = _loaded_with_response(
            None,
            tool_calls=[{"id": "call_x", "type": "function",
                         "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}}],
            finish_reason="tool_calls",
        )
        result = self.backend.generate(
            loaded, self.messages,
            GenerationParams(tools=[WEATHER_TOOL],
                             tool_choice={"type": "function", "function": {"name": "get_weather"}}),
        )
        assert result.finish_reason == "tool_calls"
        assert result.tool_calls[0].id == "call_x"
        assert result.tool_calls[0].function_arguments == '{"city": "Oslo"}'

    def test_history_arguments_normalized_for_template(self):
        history = [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city": "London"}'}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]
        loaded = _loaded_with_response("It is sunny in London.")
        self.backend.generate(loaded, history, GenerationParams(tools=[WEATHER_TOOL]))
        sent = loaded.model.create_chat_completion.call_args.kwargs["messages"]
        assert sent[1]["tool_calls"][0]["function"]["arguments"] == {"city": "London"}
        # Caller's list must not be mutated
        assert history[1]["tool_calls"][0]["function"]["arguments"] == '{"city": "London"}'

    def test_response_format_passed_through(self):
        loaded = _loaded_with_response('{"a": 1}')
        self.backend.generate(
            loaded, self.messages,
            GenerationParams(response_format={"type": "json_object"}),
        )
        kwargs = loaded.model.create_chat_completion.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}


class TestLlamaCppStream:
    def setup_method(self):
        from inferall.backends.llamacpp_backend import LlamaCppBackend
        self.backend = LlamaCppBackend()
        self.messages = [{"role": "user", "content": "Weather in London?"}]

    def test_tools_and_response_format_are_sent(self):
        loaded = _loaded_with_stream(["Hi"])
        list(self.backend.stream(
            loaded, self.messages,
            GenerationParams(tools=[WEATHER_TOOL], tool_choice="auto",
                             response_format={"type": "json_object"}),
        ))
        kwargs = loaded.model.create_chat_completion.call_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["tools"] == [WEATHER_TOOL]
        assert kwargs["tool_choice"] == "auto"
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_tool_call_text_streams_through_intact(self):
        # The backend streams the raw tool call; the API layer extracts it.
        loaded = _loaded_with_stream(_split(QWEN_CALL, 4))
        out = "".join(self.backend.stream(loaded, self.messages,
                                          GenerationParams(tools=[WEATHER_TOOL])))
        assert out == QWEN_CALL

    def test_thinking_stripped_across_chunks(self):
        loaded = _loaded_with_stream(_split("<think>\nplan\n</think>\n\nSunny.", 3))
        out = "".join(self.backend.stream(loaded, self.messages, GenerationParams()))
        assert out == "Sunny."

    def test_cancel_stops_stream(self):
        import threading
        cancel = threading.Event()
        loaded = _loaded_with_stream(["a", "b", "c"])
        gen = self.backend.stream(loaded, self.messages, GenerationParams(), cancel)
        assert next(gen) == "a"
        cancel.set()
        assert list(gen) == []


class TestLlamaCppContextWindow:
    def test_default_n_ctx(self):
        from inferall.backends.llamacpp_backend import DEFAULT_N_CTX, LlamaCppBackend
        assert LlamaCppBackend().n_ctx == DEFAULT_N_CTX
        assert DEFAULT_N_CTX >= 8192

    def test_config_default_and_env(self, monkeypatch):
        from inferall.config import EngineConfig
        assert EngineConfig().gguf_n_ctx == 8192
        monkeypatch.setenv("INFERALL_GGUF_N_CTX", "0")
        monkeypatch.setenv("INFERALL_BASE_DIR", "/nonexistent/inferall-test")
        assert EngineConfig.load().gguf_n_ctx == 0

    def test_load_passes_n_ctx(self, tmp_path, monkeypatch):
        import sys, types
        from inferall.backends.llamacpp_backend import LlamaCppBackend

        captured = {}

        class FakeLlama:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))
        (tmp_path / "m.gguf").touch()
        record = MagicMock(local_path=tmp_path, model_id="x/y")
        allocation = MagicMock(n_gpu_layers=-1, tensor_split=None, estimated_vram_bytes=0)
        LlamaCppBackend(n_ctx=32768).load(record, allocation)
        assert captured["n_ctx"] == 32768


# =============================================================================
# API streaming of tool calls (any backend)
# =============================================================================

def _sse_chunks(body: str):
    chunks = []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line[6:]))
    return chunks


class TestStreamingToolCallsAPI:
    @pytest.fixture
    def client(self):
        from inferall.api.server import create_app
        from inferall.orchestrator import Orchestrator
        from inferall.registry.registry import ModelRegistry
        from starlette.testclient import TestClient

        orch = MagicMock(spec=Orchestrator)
        orch.list_loaded.return_value = []
        registry = MagicMock(spec=ModelRegistry)
        registry.list_all.return_value = []
        app = create_app(orchestrator=orch, registry=registry, api_key=None)
        client = TestClient(app)
        client._orch = orch
        return client

    def _post(self, client, tokens, tools=True):
        client._orch.stream.return_value = iter(tokens)
        body = {
            "model": "test/model",
            "messages": [{"role": "user", "content": "Weather in London?"}],
            "stream": True,
        }
        if tools:
            body["tools"] = [WEATHER_TOOL]
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        return _sse_chunks(resp.text)

    def test_tool_call_streamed_as_delta(self, client):
        chunks = self._post(client, _split("Sure. " + QWEN_CALL, 4))
        deltas = [c["choices"][0]["delta"] for c in chunks]

        content = "".join(d.get("content", "") for d in deltas)
        assert "<tool_call>" not in content
        assert content.strip() == "Sure."

        tc_deltas = [d["tool_calls"] for d in deltas if "tool_calls" in d]
        assert len(tc_deltas) == 1
        tc = tc_deltas[0][0]
        assert tc["index"] == 0
        assert tc["type"] == "function"
        assert tc["id"].startswith("call_")
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "London"}

        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

    def test_two_tool_calls_get_sequential_indexes(self, client):
        text = (
            '<tool_call>{"name": "f1", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "f2", "arguments": {}}</tool_call>'
        )
        chunks = self._post(client, _split(text, 6))
        idx = [d["tool_calls"][0]["index"] for d in
               (c["choices"][0]["delta"] for c in chunks) if "tool_calls" in d]
        assert idx == [0, 1]

    def test_plain_reply_with_tools_finishes_stop(self, client):
        chunks = self._post(client, ["No ", "tool ", "needed."])
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert content == "No tool needed."
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    def test_without_tools_markup_passes_through_unchanged(self, client):
        chunks = self._post(client, _split(QWEN_CALL, 5), tools=False)
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert content == QWEN_CALL
        assert not any("tool_calls" in c["choices"][0]["delta"] for c in chunks)
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


# =============================================================================
# Review follow-ups
# =============================================================================

class TestMistralNestedArguments:
    def test_nested_array_in_arguments_and_id_preserved(self):
        text = ('[TOOL_CALLS] [{"name":"lookup","arguments":{"ids":[1,2]},'
                '"id":"abc123xyz"}]')
        calls, remaining = parse_tool_calls(text)
        assert calls is not None and len(calls) == 1
        assert calls[0].function_name == "lookup"
        assert calls[0].id == "abc123xyz"
        assert json.loads(calls[0].function_arguments) == {"ids": [1, 2]}
        assert remaining == ""

    def test_text_around_mistral_call(self):
        text = 'Sure. [TOOL_CALLS] [{"name":"f","arguments":{"a":[[1],[2]]}}] done'
        calls, remaining = parse_tool_calls(text)
        assert json.loads(calls[0].function_arguments) == {"a": [[1], [2]]}
        assert remaining.split() == ["Sure.", "done"]

    def test_truncated_mistral_array_is_left_as_text(self):
        text = '[TOOL_CALLS] [{"name":"f","arguments":{"a":[1,2'
        calls, remaining = parse_tool_calls(text)
        assert calls is None
        assert remaining == text


class TestLlamaCppNativeStreamingToolCalls:
    """tool_choice naming a function: llama-cpp-python streams native deltas."""

    def setup_method(self):
        from inferall.backends.llamacpp_backend import LlamaCppBackend
        self.backend = LlamaCppBackend()
        self.messages = [{"role": "user", "content": "Weather in London?"}]
        self.params = GenerationParams(
            tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )

    @staticmethod
    def _native_stream(fragments, name="get_weather"):
        chunks = [{"choices": [{"delta": {"role": "assistant", "content": None,
                                           "tool_calls": None}}]}]
        for i, frag in enumerate(fragments):
            chunks.append({"choices": [{"delta": {
                "content": None,
                "tool_calls": [{"index": 0, "id": "call_native",
                                "type": "function",
                                "function": {"name": name if i == 0 else None,
                                             "arguments": frag}}],
            }}]})
        loaded = MagicMock()
        loaded.model.create_chat_completion.return_value = iter(chunks)
        return loaded

    def test_fragmented_native_arguments_are_assembled(self):
        loaded = self._native_stream(['{"ci', 'ty": "Lo', 'ndon"}'])
        out = "".join(self.backend.stream(loaded, self.messages, self.params))
        ex = ToolCallStreamExtractor()
        events = list(ex.feed(out)) + list(ex.flush())
        assert [k for k, _ in events] == ["tool_call"]
        call = events[0][1]
        assert call.function_name == "get_weather"
        assert json.loads(call.function_arguments) == {"city": "London"}

    def test_truncated_native_arguments_not_dropped(self):
        loaded = self._native_stream(['{"city": "Lon'])
        out = "".join(self.backend.stream(loaded, self.messages, self.params))
        ex = ToolCallStreamExtractor()
        events = list(ex.feed(out)) + list(ex.flush())
        assert [k for k, _ in events] == ["tool_call"]
        call = events[0][1]
        assert call.function_name == "get_weather"
        # Unparseable fragment is carried through verbatim as the arguments string.
        assert call.function_arguments == '{"city": "Lon'

    def test_native_stream_end_to_end_through_api(self):
        # The assembled text must come out of the API as a tool_calls delta.
        loaded = self._native_stream(['{"city": ', '"Oslo"}'])
        tokens = list(self.backend.stream(loaded, self.messages, self.params))

        from inferall.api.server import create_app
        from inferall.orchestrator import Orchestrator
        from inferall.registry.registry import ModelRegistry
        from starlette.testclient import TestClient
        orch = MagicMock(spec=Orchestrator)
        orch.list_loaded.return_value = []
        orch.stream.return_value = iter(tokens)
        registry = MagicMock(spec=ModelRegistry)
        registry.list_all.return_value = []
        client = TestClient(create_app(orchestrator=orch, registry=registry, api_key=None))
        resp = client.post("/v1/chat/completions", json={
            "model": "m", "stream": True,
            "messages": [{"role": "user", "content": "x"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        })
        chunks = _sse_chunks(resp.text)
        tc = [d["tool_calls"][0] for d in (c["choices"][0]["delta"] for c in chunks)
              if "tool_calls" in d]
        assert len(tc) == 1 and tc[0]["function"]["name"] == "get_weather"
        assert json.loads(tc[0]["function"]["arguments"]) == {"city": "Oslo"}
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


class TestStreamingToolChoiceNone(TestStreamingToolCallsAPI):
    def test_tool_choice_none_disables_extraction(self, client):
        client._orch.stream.return_value = iter(_split(QWEN_CALL, 5))
        resp = client.post("/v1/chat/completions", json={
            "model": "test/model",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
            "tools": [WEATHER_TOOL],
            "tool_choice": "none",
        })
        chunks = _sse_chunks(resp.text)
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert content == QWEN_CALL
        assert not any("tool_calls" in c["choices"][0]["delta"] for c in chunks)
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


class TestAllocationContextLength:
    def test_heuristic_kv_estimate_scales_with_context(self):
        from inferall.gpu.allocator import GPUAllocator
        from inferall.registry.metadata import ModelFormat
        allocator = GPUAllocator(MagicMock(), vram_buffer_mb=0)
        record = MagicMock(model_id="x/y", param_count=1_000_000_000,
                           format=ModelFormat.GGUF, file_size_bytes=0)
        record.local_path = MagicMock()
        record.local_path.__truediv__ = lambda self, other: MagicMock(exists=lambda: False)
        base = allocator._estimate_vram(record, 4096)
        bigger = allocator._estimate_vram(record, 32768)
        assert bigger > base

    def test_orchestrator_plans_gguf_allocation_with_configured_ctx(self):
        from datetime import datetime
        from pathlib import Path
        from unittest.mock import patch
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import AllocationPlan, GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.metadata import ModelFormat, ModelRecord, ModelTask
        from inferall.registry.registry import ModelRegistry

        record = ModelRecord(
            model_id="org/model-GGUF", revision="abc123", format=ModelFormat.GGUF,
            local_path=Path("/tmp/x"), file_size_bytes=1, param_count=1_000_000_000,
            gguf_variant="Q4_K_M", trust_remote_code=False,
            pipeline_tag="text-generation", pulled_at=datetime.now(), task=ModelTask.CHAT,
        )
        registry = MagicMock(spec=ModelRegistry)
        registry.get.return_value = record
        allocator = MagicMock(spec=GPUAllocator)
        allocator.compute_allocation.return_value = AllocationPlan(
            gpu_ids=[0], estimated_vram_bytes=1,
        )
        gpu_manager = MagicMock(spec=GPUManager)
        gpu_manager.gpu_assignments = {}

        orch = Orchestrator(EngineConfig(idle_timeout=0, gguf_n_ctx=16384),
                            registry, gpu_manager, allocator)
        with patch.object(orch, "_get_backend") as backend:
            backend.return_value.load.return_value = MagicMock(model_id=record.model_id)
            orch.get_or_load(record.model_id)
        allocator.compute_allocation.assert_called_once_with(record, max_seq_len=16384)

    def test_orchestrator_keeps_default_when_ctx_is_model_length(self):
        from datetime import datetime
        from pathlib import Path
        from unittest.mock import patch
        from inferall.config import EngineConfig
        from inferall.gpu.allocator import AllocationPlan, GPUAllocator
        from inferall.gpu.manager import GPUManager
        from inferall.orchestrator import Orchestrator
        from inferall.registry.metadata import ModelFormat, ModelRecord, ModelTask
        from inferall.registry.registry import ModelRegistry

        record = ModelRecord(
            model_id="org/model-GGUF", revision="abc123", format=ModelFormat.GGUF,
            local_path=Path("/tmp/x"), file_size_bytes=1, param_count=1_000_000_000,
            gguf_variant="Q4_K_M", trust_remote_code=False,
            pipeline_tag="text-generation", pulled_at=datetime.now(), task=ModelTask.CHAT,
        )
        registry = MagicMock(spec=ModelRegistry)
        registry.get.return_value = record
        allocator = MagicMock(spec=GPUAllocator)
        allocator.compute_allocation.return_value = AllocationPlan(gpu_ids=[0], estimated_vram_bytes=1)
        gpu_manager = MagicMock(spec=GPUManager)
        gpu_manager.gpu_assignments = {}

        orch = Orchestrator(EngineConfig(idle_timeout=0, gguf_n_ctx=0),
                            registry, gpu_manager, allocator)
        with patch.object(orch, "_get_backend") as backend:
            backend.return_value.load.return_value = MagicMock(model_id=record.model_id)
            orch.get_or_load(record.model_id)
        allocator.compute_allocation.assert_called_once_with(record)

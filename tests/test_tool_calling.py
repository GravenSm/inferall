"""Tests for tool/function calling and structured output (JSON mode)."""

from unittest.mock import MagicMock

import pytest

from inferall.backends.base import (
    GenerationParams,
    GenerationResult,
    ToolCall,
)


# =============================================================================
# Data Structure Tests
# =============================================================================

class TestToolCall:
    def test_fields(self):
        tc = ToolCall(
            id="call_abc123",
            function_name="get_weather",
            function_arguments='{"city": "London"}',
        )
        assert tc.id == "call_abc123"
        assert tc.type == "function"
        assert tc.function_name == "get_weather"

class TestGenerationParamsTools:
    def test_defaults(self):
        p = GenerationParams()
        assert p.tools is None
        assert p.tool_choice is None
        assert p.response_format is None

    def test_with_tools(self):
        tools = [{"type": "function", "function": {"name": "test"}}]
        p = GenerationParams(tools=tools, tool_choice="auto")
        assert len(p.tools) == 1
        assert p.tool_choice == "auto"

class TestGenerationResultToolCalls:
    def test_with_tool_calls(self):
        r = GenerationResult(
            text="",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(id="call_1", function_name="f1", function_arguments="{}"),
                ToolCall(id="call_2", function_name="f2", function_arguments='{"x":1}'),
            ],
        )
        assert r.finish_reason == "tool_calls"
        assert len(r.tool_calls) == 2


# =============================================================================
# Tool Call Parsing Tests (Transformers backend)
# =============================================================================

class TestToolCallParsing:
    def test_qwen_format(self):
        from inferall.backends.transformers_backend import TransformersBackend
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "London"}}\n</tool_call>'
        tool_calls, remaining = TransformersBackend._parse_tool_calls(text)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function_name == "get_weather"
        assert '"city"' in tool_calls[0].function_arguments

    def test_mistral_format(self):
        from inferall.backends.transformers_backend import TransformersBackend
        text = '[TOOL_CALLS] [{"name": "search", "arguments": {"query": "test"}}]'
        tool_calls, remaining = TransformersBackend._parse_tool_calls(text)
        assert tool_calls is not None
        assert tool_calls[0].function_name == "search"

    def test_llama_format(self):
        from inferall.backends.transformers_backend import TransformersBackend
        text = '<function=get_weather>{"city": "Paris"}</function>'
        tool_calls, remaining = TransformersBackend._parse_tool_calls(text)
        assert tool_calls is not None
        assert tool_calls[0].function_name == "get_weather"

    def test_no_tool_calls(self):
        from inferall.backends.transformers_backend import TransformersBackend
        text = "The weather in London is sunny."
        tool_calls, remaining = TransformersBackend._parse_tool_calls(text)
        assert tool_calls is None
        assert remaining == text

    def test_multiple_qwen_calls(self):
        from inferall.backends.transformers_backend import TransformersBackend
        text = (
            '<tool_call>\n{"name": "f1", "arguments": {"a": 1}}\n</tool_call>'
            '<tool_call>\n{"name": "f2", "arguments": {"b": 2}}\n</tool_call>'
        )
        tool_calls, remaining = TransformersBackend._parse_tool_calls(text)
        assert len(tool_calls) == 2
        assert tool_calls[0].function_name == "f1"
        assert tool_calls[1].function_name == "f2"


# =============================================================================
# Response Format Injection Tests
# =============================================================================

class TestResponseFormatInjection:
    def test_json_object_mode(self):
        from inferall.backends.transformers_backend import TransformersBackend
        messages = [{"role": "user", "content": "List 3 colors"}]
        result = TransformersBackend._inject_response_format(
            messages, {"type": "json_object"}
        )
        assert result[0]["role"] == "system"
        assert "JSON" in result[0]["content"]

    def test_text_mode_no_change(self):
        from inferall.backends.transformers_backend import TransformersBackend
        messages = [{"role": "user", "content": "Hi"}]
        result = TransformersBackend._inject_response_format(
            messages, {"type": "text"}
        )
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_json_schema_mode(self):
        from inferall.backends.transformers_backend import TransformersBackend
        messages = [{"role": "user", "content": "Get info"}]
        result = TransformersBackend._inject_response_format(
            messages,
            {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
        )
        assert "schema" in result[0]["content"].lower()

    def test_existing_system_message(self):
        from inferall.backends.transformers_backend import TransformersBackend
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = TransformersBackend._inject_response_format(
            messages, {"type": "json_object"}
        )
        assert len(result) == 2
        assert "JSON" in result[0]["content"]
        assert "helpful" in result[0]["content"]


# =============================================================================
# API Endpoint Tests
# =============================================================================

class TestToolCallingAPI:
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

    def test_tool_call_response(self, client):
        client._orch.generate.return_value = GenerationResult(
            text="",
            prompt_tokens=20,
            completion_tokens=15,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(
                    id="call_abc123",
                    function_name="get_weather",
                    function_arguments='{"city": "London"}',
                ),
            ],
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "test/model",
            "messages": [{"role": "user", "content": "What's the weather in London?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }],
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        tc = data["choices"][0]["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "get_weather"
        assert data["choices"][0]["message"]["content"] is None

    def test_tool_result_message(self, client):
        """Test that tool result messages are passed through correctly."""
        client._orch.generate.return_value = GenerationResult(
            text="The weather in London is sunny, 22°C.",
            prompt_tokens=30,
            completion_tokens=10,
            finish_reason="stop",
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "test/model",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 22, "condition": "sunny"}'},
            ],
        })

        assert resp.status_code == 200
        assert "sunny" in resp.json()["choices"][0]["message"]["content"]

    def test_json_mode_request(self, client):
        client._orch.generate.return_value = GenerationResult(
            text='{"colors": ["red", "blue", "green"]}',
            prompt_tokens=10, completion_tokens=8, finish_reason="stop",
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "test/model",
            "messages": [{"role": "user", "content": "List 3 colors"}],
            "response_format": {"type": "json_object"},
        })

        assert resp.status_code == 200

    def test_health_tools_enabled(self, client):
        resp = client.get("/health")
        assert resp.json()["capabilities"]["tools"] is True

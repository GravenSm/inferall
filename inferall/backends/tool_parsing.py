"""
Tool-call parsing helpers shared by text-generation backends
------------------------------------------------------------
Open-weight chat models signal tool calls as plain text in a model-specific
format (Qwen's ``<tool_call>{...}</tool_call>``, Mistral's ``[TOOL_CALLS]``,
Llama 3.1's ``<function=...>``). Backends that do not get structured
``tool_calls`` back from their runtime use these helpers to:

- turn that text into :class:`ToolCall` objects (``parse_tool_calls``),
- strip ``<think>...</think>`` reasoning blocks (``strip_thinking``,
  ``ThinkingStreamFilter``),
- normalise incoming OpenAI-style messages so chat templates render prior
  tool calls correctly (``normalize_tool_call_messages``),
- extract tool calls from a token stream (``ToolCallStreamExtractor``).
"""

import copy
import json
import re
import uuid
from typing import Iterator, List, Optional, Tuple

from inferall.backends.base import ToolCall

# Qwen / Hermes format
_QWEN_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Mistral format
_MISTRAL_PATTERN = re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\])", re.DOTALL)
# Llama 3.1 format
_LLAMA_PATTERN = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)
# Bare JSON object with name + arguments/parameters
_GENERIC_PATTERN = re.compile(
    r'\{"name"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|parameters)"\s*:\s*\{.*?\}\s*\}',
    re.DOTALL,
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_UNCLOSED = re.compile(r"^\s*<think>.*", re.DOTALL)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _make_call(name: str, arguments, call_id: Optional[str] = None) -> ToolCall:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments if arguments is not None else {})
    return ToolCall(
        id=call_id or _new_call_id(),
        function_name=name,
        function_arguments=arguments,
    )


def parse_tool_calls(text: str) -> Tuple[Optional[List[ToolCall]], str]:
    """
    Parse tool calls from model output text.

    Handles formats from Qwen, Llama 3.1+, Mistral, and generic JSON.
    Returns ``(list_of_ToolCall_or_None, remaining_text)``.
    """
    tool_calls: List[ToolCall] = []

    # Pattern 1: Qwen format — <tool_call>{"name": ..., "arguments": ...}</tool_call>
    for match in _QWEN_PATTERN.finditer(text):
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        tool_calls.append(_make_call(data.get("name", ""), data.get("arguments", {})))
    if tool_calls:
        return tool_calls, _QWEN_PATTERN.sub("", text).strip()

    # Pattern 2: Mistral format — [TOOL_CALLS] [{"name": ..., "arguments": ...}]
    match = _MISTRAL_PATTERN.search(text)
    if match:
        try:
            calls = json.loads(match.group(1))
            for call in calls:
                tool_calls.append(_make_call(
                    call.get("name", ""), call.get("arguments", {}), call.get("id"),
                ))
            return tool_calls, text[:match.start()].strip()
        except (json.JSONDecodeError, AttributeError):
            pass

    # Pattern 3: Llama 3.1 format — <function=name>{...}</function>
    for match in _LLAMA_PATTERN.finditer(text):
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(_make_call(match.group(1), args))
    if tool_calls:
        return tool_calls, _LLAMA_PATTERN.sub("", text).strip()

    # Pattern 4: Generic JSON — {"name": ..., "arguments": ...}
    for match in _GENERIC_PATTERN.finditer(text):
        try:
            data = json.loads(match.group(0))
            args = data.get("arguments", data.get("parameters", {}))
            tool_calls.append(_make_call(data["name"], args))
        except (json.JSONDecodeError, KeyError):
            continue
    if tool_calls:
        return tool_calls, _GENERIC_PATTERN.sub("", text).strip()

    return None, text


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks (and an unclosed trailing one)."""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_UNCLOSED.sub("", text)
    return text.strip()


def normalize_tool_call_messages(messages: List[dict]) -> List[dict]:
    """
    Return a copy of ``messages`` where every assistant ``tool_calls[*]``
    has ``function.arguments`` as a dict.

    OpenAI clients send ``arguments`` as a JSON *string*. Jinja chat
    templates (Qwen 2.5, Llama 3.x, ...) render it with ``| tojson`` and
    expect an object; given a string they emit ``"arguments": "{\\"x\\": 1}"``
    which misrepresents the conversation history to the model.
    Arguments that are not valid JSON are left untouched.
    """
    out: List[dict] = []
    for msg in messages:
        calls = msg.get("tool_calls") if isinstance(msg, dict) else None
        if not calls:
            out.append(msg)
            continue
        msg = copy.deepcopy(msg)
        for call in msg["tool_calls"]:
            fn = call.get("function") if isinstance(call, dict) else None
            if not isinstance(fn, dict):
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    parsed = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    fn["arguments"] = parsed
        out.append(msg)
    return out


class ThinkingStreamFilter:
    """
    Incrementally strip ``<think>...</think>`` blocks from a token stream.

    Text that could be the start of a ``<think>`` tag is held back until it
    is known to be ordinary content, so tags split across tokens are caught.
    Whitespace directly after a leading think block is dropped, matching
    ``strip_thinking``.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False
        self._emitted_any = False
        self._strip_leading = False

    def feed(self, token: str) -> str:
        self._buf += token
        out = ""
        while self._buf:
            if self._in_think:
                idx = self._buf.find(self._CLOSE)
                if idx == -1:
                    return out
                self._buf = self._buf[idx + len(self._CLOSE):]
                self._in_think = False
                if not self._emitted_any:
                    self._strip_leading = True
                continue

            if self._strip_leading:
                self._buf = self._buf.lstrip()
                if not self._buf:
                    break

            idx = self._buf.find(self._OPEN)
            if idx != -1:
                out += self._buf[:idx]
                self._buf = self._buf[idx + len(self._OPEN):]
                self._in_think = True
                continue

            keep = _partial_suffix_len(self._buf, self._OPEN)
            out += self._buf[:len(self._buf) - keep]
            self._buf = self._buf[len(self._buf) - keep:]
            break
        if out:
            self._emitted_any = True
            self._strip_leading = False
        return out

    def flush(self) -> str:
        """Return any held-back content (an unclosed think block is dropped)."""
        out = "" if self._in_think else self._buf
        if self._strip_leading:
            out = out.lstrip()
        self._buf = ""
        self._in_think = False
        return out


class ToolCallStreamExtractor:
    """
    Split a streamed assistant reply into content and tool calls.

    ``feed(token)`` yields ``("content", str)`` and ``("tool_call", ToolCall)``
    events; ``flush()`` yields whatever is still buffered at end of stream.
    Only the Qwen/Hermes ``<tool_call>`` format can be detected incrementally;
    other formats pass through as content.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_call = False

    def feed(self, token: str) -> Iterator[tuple]:
        self._buf += token
        while self._buf:
            if self._in_call:
                idx = self._buf.find(TOOL_CALL_CLOSE)
                if idx == -1:
                    return
                body = self._buf[:idx]
                self._buf = self._buf[idx + len(TOOL_CALL_CLOSE):]
                self._in_call = False
                call = self._parse_body(body)
                if call is not None:
                    yield ("tool_call", call)
                else:
                    yield ("content", TOOL_CALL_OPEN + body + TOOL_CALL_CLOSE)
                continue

            idx = self._buf.find(TOOL_CALL_OPEN)
            if idx != -1:
                if idx > 0:
                    yield ("content", self._buf[:idx])
                self._buf = self._buf[idx + len(TOOL_CALL_OPEN):]
                self._in_call = True
                continue

            keep = _partial_suffix_len(self._buf, TOOL_CALL_OPEN)
            emit = self._buf[:len(self._buf) - keep]
            self._buf = self._buf[len(self._buf) - keep:]
            if emit:
                yield ("content", emit)
            return

    def flush(self) -> Iterator[tuple]:
        if self._in_call:
            # Stream ended inside a tool call (hit max_tokens or a stop
            # string). Try to salvage it; otherwise return it verbatim.
            call = self._parse_body(self._buf)
            if call is not None:
                yield ("tool_call", call)
            elif self._buf:
                yield ("content", TOOL_CALL_OPEN + self._buf)
        elif self._buf:
            yield ("content", self._buf)
        self._buf = ""
        self._in_call = False

    @staticmethod
    def _parse_body(body: str) -> Optional[ToolCall]:
        try:
            data = json.loads(body.strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or "name" not in data:
            return None
        return _make_call(data["name"], data.get("arguments", {}))


def _partial_suffix_len(buf: str, tag: str) -> int:
    """Length of the longest suffix of ``buf`` that is a proper prefix of ``tag``."""
    max_len = min(len(buf), len(tag) - 1)
    for n in range(max_len, 0, -1):
        if buf.endswith(tag[:n]):
            return n
    return 0

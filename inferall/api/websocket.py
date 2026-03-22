"""
WebSocket Streaming
--------------------
Persistent WebSocket connection for real-time chat streaming.
Eliminates SSE overhead for high-frequency token delivery.

Protocol:
  Client sends JSON: {"model": "...", "messages": [...], ...}
  Server streams JSON: {"type": "token", "content": "..."} per token
  Server sends: {"type": "done", "usage": {...}, "performance": {...}} at end
  Server sends: {"type": "error", "message": "..."} on error
"""

import asyncio
import json
import logging
import threading
import time

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


async def websocket_chat(
    websocket: WebSocket,
    orchestrator,
    dispatcher,
):
    """
    Handle a WebSocket chat connection.

    Supports multiple request/response cycles on the same connection.
    """
    await websocket.accept()
    logger.info("WebSocket connection opened")

    try:
        while True:
            # Receive a request
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            try:
                request = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            model_id = request.get("model", "")
            messages = request.get("messages", [])
            max_tokens = request.get("max_tokens", 2048)
            temperature = request.get("temperature", 0.7)
            stream = request.get("stream", True)

            if not model_id or not messages:
                await websocket.send_json({"type": "error", "message": "model and messages required"})
                continue

            from inferall.backends.base import GenerationParams
            params = GenerationParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=request.get("top_p", 0.9),
                stop=request.get("stop"),
                tools=request.get("tools"),
                tool_choice=request.get("tool_choice"),
                response_format=request.get("response_format"),
            )

            if stream:
                await _stream_over_ws(websocket, orchestrator, model_id, messages, params)
            else:
                await _generate_over_ws(websocket, orchestrator, dispatcher, model_id, messages, params)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        logger.info("WebSocket connection closed")


async def _stream_over_ws(websocket, orchestrator, model_id, messages, params):
    """Stream tokens over WebSocket."""
    import queue

    token_queue = queue.Queue()
    cancel = threading.Event()
    error_holder = []
    done_event = threading.Event()

    def _run():
        try:
            for token in orchestrator.stream(model_id, messages, params, cancel):
                token_queue.put(token)
        except Exception as e:
            error_holder.append(e)
        finally:
            done_event.set()

    t0 = time.perf_counter()
    token_count = 0
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run)

    try:
        while True:
            try:
                token = token_queue.get_nowait()
                token_count += 1
                await websocket.send_json({"type": "token", "content": token})
            except queue.Empty:
                if done_event.is_set() and token_queue.empty():
                    break
                await asyncio.sleep(0.01)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        tps = token_count / (elapsed_ms / 1000.0) if elapsed_ms > 0 and token_count > 0 else 0

        if error_holder:
            await websocket.send_json({"type": "error", "message": str(error_holder[0])})
        else:
            await websocket.send_json({
                "type": "done",
                "usage": {"completion_tokens": token_count},
                "performance": {
                    "total_time_ms": round(elapsed_ms, 1),
                    "tokens_per_second": round(tps, 1),
                },
            })
    except WebSocketDisconnect:
        cancel.set()


async def _generate_over_ws(websocket, orchestrator, dispatcher, model_id, messages, params):
    """Non-streaming generation over WebSocket."""
    try:
        result = await dispatcher.submit(
            model_id, orchestrator.generate,
            model_id, messages, params,
        )
        await websocket.send_json({
            "type": "response",
            "message": {"role": "assistant", "content": result.text},
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
            "performance": {
                "total_time_ms": result.total_time_ms,
                "tokens_per_second": result.tokens_per_second,
            },
        })
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})

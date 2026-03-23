"""
API Server
----------
FastAPI server implementing OpenAI-compatible API endpoints.

Endpoints (OpenAI-compatible):
  POST /v1/chat/completions      — Chat completion (streaming + non-streaming)
  POST /v1/completions           — Legacy text completions
  POST /v1/embeddings            — Text embeddings
  POST /v1/moderations           — Content moderation
  POST /v1/audio/transcriptions  — Audio transcription (Whisper)
  POST /v1/audio/translations    — Audio translation to English (Whisper)
  POST /v1/audio/speech          — Text-to-speech
  POST /v1/images/generations    — Image generation (diffusion)
  POST /v1/images/edits          — Image-to-image editing (img2img)
  POST /v1/images/variations     — Image variations
  GET  /v1/models                — List models
  GET  /v1/models/{model}        — Get model details
  DELETE /v1/models/{model}      — Delete a model

Extended endpoints:
  POST /v1/videos/generations    — Text-to-video generation
  POST /v1/text/generate         — Seq2seq (translation, summarization)
  POST /v1/classify              — Classification (image, audio, zero-shot)
  POST /v1/detect                — Object detection
  POST /v1/segment               — Image segmentation
  POST /v1/depth                 — Depth estimation
  POST /v1/document-qa           — Document question answering
  POST /v1/audio/process         — Audio-to-audio processing
  POST /v1/rerank                — Document reranking
  GET  /health                   — Health check

Threading model:
  All blocking inference is offloaded to a dedicated ThreadPoolExecutor.
  An asyncio.Semaphore admission gate prevents unbounded queueing.
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from inferall.backends.base import (
    EmbeddingParams,
    GenerationParams,
    GenerationResult,
    ImageGenerationParams,
    Img2ImgParams,
    RerankParams,
    AudioProcessingParams,
    ClassificationParams,
    DepthEstimationParams,
    DocumentQAParams,
    ImageSegmentationParams,
    ObjectDetectionParams,
    Seq2SeqParams,
    VideoGenerationParams,
    TranscriptionParams,
    TTSParams,
)
from inferall.orchestrator import ModelNotFoundError, Orchestrator
from inferall.registry.registry import ModelRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# Request / Response Models (OpenAI-compatible)
# =============================================================================

class ChatMessage(BaseModel):
    role: str
    content: Any = None       # str for text-only, list of dicts for VLM, None for tool_calls
    tool_calls: Optional[list] = None       # For assistant messages with tool calls
    tool_call_id: Optional[str] = None      # For role="tool" messages
    name: Optional[str] = None              # Function name for tool results


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    n: Optional[int] = 1
    echo: Optional[bool] = False
    seed: Optional[int] = None


class ModerationRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    n: Optional[int] = 1
    # Accepted but ignored
    seed: Optional[int] = None
    response_format: Optional[dict] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    # Unsupported — rejected in strict mode
    tools: Optional[list] = None
    tool_choice: Optional[Any] = None
    functions: Optional[list] = None
    function_call: Optional[Any] = None


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"


class ImageGenerationRequest(BaseModel):
    model: str
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None


class ImageEditRequest(BaseModel):
    model: str
    prompt: str
    image: str                                      # Base64-encoded input image
    strength: Optional[float] = 0.75
    n: Optional[int] = 1
    size: Optional[str] = None
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = False


class VideoGenerationRequest(BaseModel):
    model: str
    prompt: str
    num_frames: Optional[int] = 16
    fps: Optional[int] = 8
    size: Optional[str] = "512x512"
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    output_format: Optional[str] = "frames+mp4"


class Seq2SeqRequest(BaseModel):
    model: str
    input: str                                      # Raw text input
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 1.0
    num_beams: Optional[int] = 4
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None


class ClassificationRequest(BaseModel):
    model: str
    text: Optional[str] = None                      # For zero-shot text classification
    image: Optional[str] = None                     # Base64-encoded image
    audio: Optional[str] = None                     # Base64-encoded audio
    candidate_labels: Optional[List[str]] = None    # Required for zero-shot
    top_k: Optional[int] = 5


class ObjectDetectionRequest(BaseModel):
    model: str
    image: str                                      # Base64-encoded image
    threshold: Optional[float] = 0.5
    candidate_labels: Optional[List[str]] = None    # For zero-shot


class ImageSegmentationRequest(BaseModel):
    model: str
    image: str                                      # Base64-encoded image
    threshold: Optional[float] = 0.5


class DepthEstimationRequest(BaseModel):
    model: str
    image: str                                      # Base64-encoded image


class DocumentQARequest(BaseModel):
    model: str
    image: str                                      # Base64-encoded document image
    question: str


class AudioProcessingRequest(BaseModel):
    model: str
    audio: str                                      # Base64-encoded audio


class CreateAssistantRequest(BaseModel):
    model: str
    name: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[list] = Field(default_factory=list)
    file_ids: Optional[list] = Field(default_factory=list)
    metadata: Optional[dict] = Field(default_factory=dict)


class ModifyAssistantRequest(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[list] = None
    file_ids: Optional[list] = None
    metadata: Optional[dict] = None


class CreateThreadRequest(BaseModel):
    metadata: Optional[dict] = Field(default_factory=dict)


class CreateMessageRequest(BaseModel):
    role: str
    content: str
    file_ids: Optional[list] = Field(default_factory=list)
    metadata: Optional[dict] = Field(default_factory=dict)


class CreateRunRequest(BaseModel):
    assistant_id: str
    model: Optional[str] = None
    instructions: Optional[str] = None
    metadata: Optional[dict] = Field(default_factory=dict)


class CreateFineTuningJobRequest(BaseModel):
    model: str
    training_file: str
    validation_file: Optional[str] = None
    hyperparameters: Optional[dict] = None
    suffix: Optional[str] = None


class CreateBatchRequest(BaseModel):
    input_file_id: str
    endpoint: str
    completion_window: str = "24h"
    metadata: Optional[dict] = None


class TTSRequest(BaseModel):
    model: str
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0


# =============================================================================
# Error Helpers
# =============================================================================

def _error_response(
    status: int,
    message: str,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
    headers: Optional[dict] = None,
) -> JSONResponse:
    """Build an OpenAI-compatible error response."""
    content = {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }
    return JSONResponse(status_code=status, content=content, headers=headers)


# =============================================================================
# App Factory
# =============================================================================

def create_app(
    orchestrator: Orchestrator,
    registry: ModelRegistry,
    api_key: Optional[str] = None,
    compat_mode: str = "strict",
    inference_workers: int = 2,
    admission_timeout: float = 30.0,
    file_store=None,
    files_dir=None,
    assistants_store=None,
    fine_tuning_store=None,
    batch_store=None,
    dispatcher=None,
    key_store=None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        orchestrator: The model lifecycle orchestrator
        registry: Model registry for listing models
        api_key: Optional API key for auth (None = no auth)
        compat_mode: "strict" (reject unsupported) or "lenient" (strip with warning)
        inference_workers: Thread pool size for inference
        admission_timeout: Seconds to wait for a free worker before 503
    """
    # Per-model dispatcher (or fallback to simple pool+semaphore)
    from inferall.scheduling.dispatcher import ModelDispatcher
    if dispatcher is None:
        dispatcher = ModelDispatcher(
            max_workers=inference_workers,
            max_concurrent=inference_workers * 4,
            concurrency_per_model=1,
            model_queue_size=64,
            admission_timeout=admission_timeout,
        )

    # Keep references for backward compatibility with endpoints that use the pool directly
    inference_pool = dispatcher.pool
    inference_semaphore = asyncio.Semaphore(inference_workers)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        dispatcher.shutdown()

    app = FastAPI(title="InferAll", version="0.1.0", lifespan=lifespan)

    # ------------------------------------------------------------------
    # Middleware: API Key Auth (supports multi-key + single-key)
    # ------------------------------------------------------------------

    from inferall.auth.middleware import create_auth_middleware
    _auth_middleware = create_auth_middleware(
        key_store=key_store,
        single_api_key=api_key,
    )

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        return await _auth_middleware(request, call_next)

    # ------------------------------------------------------------------
    # POST /v1/chat/completions
    # ------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        # Validate n
        if request.n is not None and request.n > 1:
            return _error_response(
                400,
                "Only n=1 is supported.",
                param="n",
                code="invalid_request_error",
            )

        # Convert legacy functions/function_call → tools/tool_choice
        tools = request.tools
        tool_choice = request.tool_choice
        if request.functions and not tools:
            tools = [{"type": "function", "function": f} for f in request.functions]
            if request.function_call:
                if isinstance(request.function_call, str):
                    tool_choice = request.function_call  # "auto" or "none"
                elif isinstance(request.function_call, dict):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": request.function_call.get("name", "")},
                    }

        # Build generation params
        params = GenerationParams(
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature if request.temperature is not None else 0.7,
            top_p=request.top_p if request.top_p is not None else 0.9,
            stop=request.stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=request.response_format,
        )

        # Build messages — preserve tool-related fields
        messages = []
        for m in request.messages:
            msg = {"role": m.role, "content": m.content}
            if m.tool_calls:
                msg["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            if m.name:
                msg["name"] = m.name
            messages.append(msg)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if request.stream:
            # Streaming still uses the pool+semaphore for the generator bridge
            try:
                await asyncio.wait_for(
                    inference_semaphore.acquire(), timeout=admission_timeout
                )
            except asyncio.TimeoutError:
                return _error_response(503, "Server busy.", error_type="server_error",
                                       code="capacity", headers={"Retry-After": "5"})

            return EventSourceResponse(
                _stream_generator(
                    orchestrator, request.model, messages, params,
                    completion_id, created, inference_semaphore, inference_pool,
                ),
                media_type="text/event-stream",
            )

        # Non-streaming: use per-model dispatcher
        try:
            result = await dispatcher.submit(
                request.model, orchestrator.generate,
                request.model, messages, params,
            )
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except asyncio.TimeoutError as e:
            return _error_response(503, str(e), error_type="server_error",
                                   code="capacity", headers={"Retry-After": "5"})
        except RuntimeError as e:
            if "queue is full" in str(e):
                return _error_response(503, str(e), error_type="server_error",
                                       code="capacity", headers={"Retry-After": "5"})
            logger.error("Generation error", exc_info=True)
            return _error_response(503, f"Generation failed: {e}",
                                   error_type="server_error", code="generation_error")
        except Exception as e:
            logger.error("Generation error", exc_info=True)
            return _error_response(503, f"Generation failed: {e}",
                                   error_type="server_error", code="generation_error")

        return _format_completion_response(
            result, request.model, completion_id, created
        )

    # ------------------------------------------------------------------
    # POST /v1/completions (legacy)
    # ------------------------------------------------------------------

    @app.post("/v1/completions")
    async def create_completion(request: CompletionRequest):
        """Legacy completions endpoint — wraps text as a user message."""
        if request.n is not None and request.n > 1:
            return _error_response(400, "Only n=1 is supported.", param="n")

        prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]

        params = GenerationParams(
            max_tokens=request.max_tokens or 2048,
            temperature=request.temperature if request.temperature is not None else 0.7,
            top_p=request.top_p if request.top_p is not None else 0.9,
            stop=request.stop,
        )

        messages = [{"role": "user", "content": prompt}]
        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        try:
            await asyncio.wait_for(
                inference_semaphore.acquire(), timeout=admission_timeout
            )
        except asyncio.TimeoutError:
            return _error_response(
                503, "Server busy.",
                error_type="server_error",
                code="capacity",
                headers={"Retry-After": "5"},
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                inference_pool, orchestrator.generate,
                request.model, messages, params,
            )
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except Exception as e:
            logger.error("Completion error", exc_info=True)
            return _error_response(
                503, f"Completion failed: {e}",
                error_type="server_error", code="completion_error",
            )
        finally:
            inference_semaphore.release()

        text = result.text
        if request.echo:
            text = prompt + text

        return JSONResponse(content={
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": request.model,
            "choices": [{
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": result.finish_reason,
            }],
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

    # ------------------------------------------------------------------
    # POST /v1/moderations
    # ------------------------------------------------------------------

    @app.post("/v1/moderations")
    async def create_moderation(request: ModerationRequest):
        """Content moderation — flags harmful/toxic content."""
        texts = request.input if isinstance(request.input, list) else [request.input]

        # Use a default moderation model if none specified
        model_id = request.model or "default-moderation"

        # Simple keyword-based fallback if no moderation model is loaded
        # (when a proper moderation model is pulled, it goes through classify)
        results = []
        for text in texts:
            # Check if we have a model to use
            record = registry.get(model_id) if request.model else None

            if record:
                # Use the classification pipeline
                from inferall.backends.base import ClassificationParams
                params = ClassificationParams(
                    candidate_labels=["safe", "unsafe", "hate", "violence",
                                      "sexual", "self-harm", "harassment"],
                    top_k=7,
                )
                try:
                    await asyncio.wait_for(
                        inference_semaphore.acquire(), timeout=admission_timeout
                    )
                except asyncio.TimeoutError:
                    return _error_response(503, "Server busy.",
                                           error_type="server_error", code="capacity")
                try:
                    loop = asyncio.get_event_loop()
                    cls_result = await loop.run_in_executor(
                        inference_pool, orchestrator.classify,
                        model_id, text, params,
                    )
                finally:
                    inference_semaphore.release()

                categories = {}
                category_scores = {}
                flagged = False
                for label_info in cls_result.labels:
                    label = label_info["label"].lower().replace(" ", "-")
                    score = label_info["score"]
                    is_flagged = score > 0.5 and label != "safe"
                    categories[label] = is_flagged
                    category_scores[label] = score
                    if is_flagged:
                        flagged = True

                results.append({
                    "flagged": flagged,
                    "categories": categories,
                    "category_scores": category_scores,
                })
            else:
                # No model — return unflagged (passthrough)
                results.append({
                    "flagged": False,
                    "categories": {},
                    "category_scores": {},
                })

        return JSONResponse(content={
            "id": f"modr-{uuid.uuid4().hex[:24]}",
            "model": model_id,
            "results": results,
        })

    # ------------------------------------------------------------------
    # POST /v1/images/variations
    # ------------------------------------------------------------------

    @app.post("/v1/images/variations")
    async def create_image_variation(
        image: UploadFile = File(...),
        model: str = Form(...),
        n: int = Form(1),
        size: str = Form("1024x1024"),
    ):
        """Generate variations of an image (img2img with high strength)."""
        from inferall.backends.base import Img2ImgParams
        import base64 as b64_mod

        image_bytes = await image.read()
        image_b64 = b64_mod.b64encode(image_bytes).decode()

        params = Img2ImgParams(
            image_b64=image_b64,
            strength=0.85,  # High strength = more variation
            n=n,
            size=size,
            num_inference_steps=30,
            guidance_scale=7.5,
        )

        try:
            await asyncio.wait_for(
                inference_semaphore.acquire(), timeout=admission_timeout
            )
        except asyncio.TimeoutError:
            return _error_response(503, "Server busy.",
                                   error_type="server_error", code="capacity",
                                   headers={"Retry-After": "5"})
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                inference_pool, orchestrator.edit_image,
                model, "variation", params,
            )
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except Exception as e:
            logger.error("Image variation error", exc_info=True)
            return _error_response(503, f"Image variation failed: {e}",
                                   error_type="server_error", code="variation_error")
        finally:
            inference_semaphore.release()

        created = int(time.time())
        data = [{"b64_json": img} for img in result.images]

        return JSONResponse(content={
            "created": created,
            "data": data,
            "performance": {
                "total_time_ms": result.total_time_ms,
            },
        })

    # ------------------------------------------------------------------
    # POST /v1/embeddings
    # ------------------------------------------------------------------

    @app.post("/v1/embeddings")
    async def create_embeddings(request: EmbeddingRequest):
        texts = request.input if isinstance(request.input, list) else [request.input]

        params = EmbeddingParams(normalize=True)

        try:
            result = await dispatcher.submit(
                request.model, orchestrator.embed,
                request.model, texts, params,
            )
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except (asyncio.TimeoutError, RuntimeError) as e:
            return _error_response(503, str(e), error_type="server_error",
                                   code="capacity", headers={"Retry-After": "5"})
        except Exception as e:
            logger.error("Embedding error", exc_info=True)
            return _error_response(503, f"Embedding failed: {e}",
                                   error_type="server_error", code="embedding_error")

        data = []
        for i, emb in enumerate(result.embeddings):
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": emb,
            })

        return JSONResponse(content={
            "object": "list",
            "data": data,
            "model": result.model,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "total_tokens": result.prompt_tokens,
            },
            "performance": {
                "total_time_ms": result.total_time_ms,
            },
        })

    # ------------------------------------------------------------------
    # POST /v1/audio/transcriptions
    # ------------------------------------------------------------------

    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        file: UploadFile = File(...),
        model: str = Form(...),
        language: Optional[str] = Form(None),
    ):
        audio_bytes = await file.read()

        params = TranscriptionParams(language=language)

        try:
            await asyncio.wait_for(
                inference_semaphore.acquire(), timeout=admission_timeout
            )
        except asyncio.TimeoutError:
            return _error_response(
                503, "Server busy.",
                error_type="server_error",
                code="capacity",
                headers={"Retry-After": "5"},
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                inference_pool, orchestrator.transcribe,
                model, audio_bytes, params,
            )
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except Exception as e:
            logger.error("Transcription error", exc_info=True)
            return _error_response(
                503, f"Transcription failed: {e}",
                error_type="server_error", code="transcription_error",
            )
        finally:
            inference_semaphore.release()

        return JSONResponse(content={
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
            "performance": {
                "total_time_ms": result.total_time_ms,
            },
        })

    # ------------------------------------------------------------------
    # POST /v1/audio/translations
    # ------------------------------------------------------------------

    @app.post("/v1/audio/translations")
    async def create_translation(
        file: UploadFile = File(...),
        model: str = Form(...),
    ):
        audio_bytes = await file.read()

        params = TranscriptionParams(task="translate")

        try:
            await asyncio.wait_for(
                inference_semaphore.acquire(), timeout=admission_timeout
            )
        except asyncio.TimeoutError:
            return _error_response(
                503, "Server busy.",
                error_type="server_error",
                code="capacity",
                headers={"Retry-After": "5"},
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                inference_pool, orchestrator.transcribe,
                model, audio_bytes, params,
            )
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except Exception as e:
            logger.error("Translation error", exc_info=True)
            return _error_response(
                503, f"Audio translation failed: {e}",
                error_type="server_error", code="translation_error",
            )
        finally:
            inference_semaphore.release()

        return JSONResponse(content={
            "text": result.text,
            "language": "en",
            "duration": result.duration,
            "performance": {
                "total_time_ms": result.total_time_ms,
            },
        })

    # ------------------------------------------------------------------
    # POST /v1/images/generations
    # ------------------------------------------------------------------

    @app.post("/v1/images/generations")
    async def create_image(request: ImageGenerationRequest):
        params = ImageGenerationParams(
            n=request.n or 1,
            size=request.size or "1024x1024",
            num_inference_steps=request.num_inference_steps or 30,
            guidance_scale=request.guidance_scale if request.guidance_scale is not None else 7.5,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
        )

        try:
            await asyncio.wait_for(
                inference_semaphore.acquire(), timeout=admission_timeout
            )
        except asyncio.TimeoutError:
            return _error_response(
                503, "Server busy.",
                error_type="server_error",
                code="capacity",
                headers={"Retry-After": "5"},
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                inference_pool, orchestrator.generate_image,
                request.model, request.prompt, params,
            )
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except Exception as e:
            logger.error("Image generation error", exc_info=True)
            return _error_response(
                503, f"Image generation failed: {e}",
                error_type="server_error", code="image_generation_error",
            )
        finally:
            inference_semaphore.release()

        created = int(time.time())
        data = []
        for img_b64 in result.images:
            data.append({"b64_json": img_b64})

        return JSONResponse(content={
            "created": created,
            "data": data,
            "performance": {
                "total_time_ms": result.total_time_ms,
            },
        })

    # ------------------------------------------------------------------
    # POST /v1/images/edits
    # ------------------------------------------------------------------

    @app.post("/v1/images/edits")
    async def edit_image(request: ImageEditRequest):
        # Validate strength
        if request.strength is not None and not (0.0 <= request.strength <= 1.0):
            return _error_response(
                400,
                "strength must be between 0.0 and 1.0",
                param="strength",
            )

        params = Img2ImgParams(
            image_b64=request.image,
            strength=request.strength if request.strength is not None else 0.75,
            n=request.n or 1,
            size=request.size,
            num_inference_steps=request.num_inference_steps or 30,
            guidance_scale=request.guidance_scale if request.guidance_scale is not None else 7.5,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
        )

        try:
            await asyncio.wait_for(
                inference_semaphore.acquire(), timeout=admission_timeout
            )
        except asyncio.TimeoutError:
            return _error_response(
                503, "Server busy.",
                error_type="server_error",
                code="capacity",
                headers={"Retry-After": "5"},
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                inference_pool, orchestrator.edit_image,
                request.model, request.prompt, params,
            )
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except Exception as e:
            logger.error("Image edit error", exc_info=True)
            return _error_response(
                503, f"Image editing failed: {e}",
                error_type="server_error", code="image_edit_error",
            )
        finally:
            inference_semaphore.release()

        created = int(time.time())
        data = []
        for img_b64 in result.images:
            data.append({"b64_json": img_b64})

        return JSONResponse(content={
            "created": created,
            "data": data,
            "performance": {
                "total_time_ms": result.total_time_ms,
            },
        })

    # ------------------------------------------------------------------
    # POST /v1/videos/generations
    # ------------------------------------------------------------------

    @app.post("/v1/videos/generations")
    async def create_video(request: VideoGenerationRequest):
        params = VideoGenerationParams(
            num_frames=request.num_frames or 16,
            fps=request.fps or 8,
            size=request.size or "512x512",
            num_inference_steps=request.num_inference_steps or 50,
            guidance_scale=request.guidance_scale if request.guidance_scale is not None else 7.5,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            output_format=request.output_format or "frames+mp4",
        )

        try:
            await asyncio.wait_for(
                inference_semaphore.acquire(), timeout=admission_timeout
            )
        except asyncio.TimeoutError:
            return _error_response(
                503, "Server busy.",
                error_type="server_error",
                code="capacity",
                headers={"Retry-After": "5"},
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                inference_pool, orchestrator.generate_video,
                request.model, request.prompt, params,
            )
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except Exception as e:
            logger.error("Video generation error", exc_info=True)
            return _error_response(
                503, f"Video generation failed: {e}",
                error_type="server_error", code="video_generation_error",
            )
        finally:
            inference_semaphore.release()

        created = int(time.time())
        frame_data = [{"b64_json": f} for f in result.frames]

        return JSONResponse(content={
            "created": created,
            "model": request.model,
            "data": {
                "frames": frame_data,
                "video_b64": result.video_b64,
                "num_frames": result.num_frames,
                "fps": result.fps,
            },
            "performance": {
                "total_time_ms": result.total_time_ms,
            },
        })

    # ------------------------------------------------------------------
    # POST /v1/audio/speech
    # ------------------------------------------------------------------

    @app.post("/v1/audio/speech")
    async def create_speech(request: TTSRequest):
        params = TTSParams(
            voice=request.voice or "default",
        )

        try:
            await asyncio.wait_for(
                inference_semaphore.acquire(), timeout=admission_timeout
            )
        except asyncio.TimeoutError:
            return _error_response(
                503, "Server busy.",
                error_type="server_error",
                code="capacity",
                headers={"Retry-After": "5"},
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                inference_pool, orchestrator.synthesize,
                request.model, request.input, params,
            )
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except Exception as e:
            logger.error("TTS error", exc_info=True)
            return _error_response(
                503, f"Speech synthesis failed: {e}",
                error_type="server_error", code="tts_error",
            )
        finally:
            inference_semaphore.release()

        # Extract format from content_type (e.g. "audio/wav" → "wav")
        audio_format = result.content_type.split("/")[-1] if "/" in result.content_type else "wav"
        return Response(
            content=result.audio_bytes,
            media_type=result.content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{audio_format}",
                "X-Total-Time-Ms": f"{result.total_time_ms:.0f}" if result.total_time_ms else "0",
            },
        )

    # ------------------------------------------------------------------
    # POST /v1/rerank
    # ------------------------------------------------------------------

    @app.post("/v1/rerank")
    async def rerank_documents(request: RerankRequest):
        params = RerankParams(
            top_n=request.top_n,
            return_documents=request.return_documents or False,
        )

        try:
            await asyncio.wait_for(
                inference_semaphore.acquire(), timeout=admission_timeout
            )
        except asyncio.TimeoutError:
            return _error_response(
                503, "Server busy.",
                error_type="server_error",
                code="capacity",
                headers={"Retry-After": "5"},
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                inference_pool, orchestrator.rerank,
                request.model, request.query, request.documents, params,
            )
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except Exception as e:
            logger.error("Rerank error", exc_info=True)
            return _error_response(
                503, f"Reranking failed: {e}",
                error_type="server_error", code="rerank_error",
            )
        finally:
            inference_semaphore.release()

        return JSONResponse(content={
            "id": f"rerank-{uuid.uuid4().hex[:24]}",
            "results": result.results,
            "meta": {
                "model": result.model,
                "usage": result.usage,
            },
            "performance": {
                "total_time_ms": result.total_time_ms,
            },
        })

    # ------------------------------------------------------------------
    # POST /v1/text/generate
    # ------------------------------------------------------------------

    @app.post("/v1/text/generate")
    async def text_generate(request: Seq2SeqRequest):
        params = Seq2SeqParams(
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature if request.temperature is not None else 1.0,
            num_beams=request.num_beams or 4,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
        )

        try:
            await asyncio.wait_for(
                inference_semaphore.acquire(), timeout=admission_timeout
            )
        except asyncio.TimeoutError:
            return _error_response(
                503, "Server busy.",
                error_type="server_error",
                code="capacity",
                headers={"Retry-After": "5"},
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                inference_pool, orchestrator.seq2seq_generate,
                request.model, request.input, params,
            )
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except Exception as e:
            logger.error("Seq2seq error", exc_info=True)
            return _error_response(
                503, f"Text generation failed: {e}",
                error_type="server_error", code="seq2seq_error",
            )
        finally:
            inference_semaphore.release()

        return JSONResponse(content={
            "id": f"seq2seq-{uuid.uuid4().hex[:24]}",
            "model": request.model,
            "text": result.text,
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

    # ------------------------------------------------------------------
    # POST /v1/classify
    # ------------------------------------------------------------------

    @app.post("/v1/classify")
    async def classify(request: ClassificationRequest):
        if not request.text and not request.image and not request.audio:
            return _error_response(
                400,
                "At least one of 'text', 'image', or 'audio' must be provided.",
                param="text",
            )

        params = ClassificationParams(
            candidate_labels=request.candidate_labels,
            top_k=request.top_k or 5,
            image_b64=request.image,
            audio_b64=request.audio,
        )

        try:
            await asyncio.wait_for(
                inference_semaphore.acquire(), timeout=admission_timeout
            )
        except asyncio.TimeoutError:
            return _error_response(
                503, "Server busy.",
                error_type="server_error",
                code="capacity",
                headers={"Retry-After": "5"},
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                inference_pool, orchestrator.classify,
                request.model, request.text or "", params,
            )
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except Exception as e:
            logger.error("Classification error", exc_info=True)
            return _error_response(
                503, f"Classification failed: {e}",
                error_type="server_error", code="classification_error",
            )
        finally:
            inference_semaphore.release()

        return JSONResponse(content={
            "id": f"classify-{uuid.uuid4().hex[:24]}",
            "model": request.model,
            "labels": result.labels,
            "pipeline_tag": result.pipeline_tag,
            "performance": {
                "total_time_ms": result.total_time_ms,
            },
        })

    # ------------------------------------------------------------------
    # POST /v1/detect
    # ------------------------------------------------------------------

    @app.post("/v1/detect")
    async def detect_objects(request: ObjectDetectionRequest):
        params = ObjectDetectionParams(
            image_b64=request.image,
            threshold=request.threshold or 0.5,
            candidate_labels=request.candidate_labels,
        )
        try:
            await asyncio.wait_for(inference_semaphore.acquire(), timeout=admission_timeout)
        except asyncio.TimeoutError:
            return _error_response(503, "Server busy.", error_type="server_error", code="capacity", headers={"Retry-After": "5"})
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(inference_pool, orchestrator.detect_objects, request.model, params)
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except Exception as e:
            logger.error("Object detection error", exc_info=True)
            return _error_response(503, f"Object detection failed: {e}", error_type="server_error", code="detection_error")
        finally:
            inference_semaphore.release()
        return JSONResponse(content={
            "model": request.model,
            "detections": result.detections,
            "performance": {"total_time_ms": result.total_time_ms},
        })

    # ------------------------------------------------------------------
    # POST /v1/segment
    # ------------------------------------------------------------------

    @app.post("/v1/segment")
    async def segment_image(request: ImageSegmentationRequest):
        params = ImageSegmentationParams(
            image_b64=request.image,
            threshold=request.threshold or 0.5,
        )
        try:
            await asyncio.wait_for(inference_semaphore.acquire(), timeout=admission_timeout)
        except asyncio.TimeoutError:
            return _error_response(503, "Server busy.", error_type="server_error", code="capacity", headers={"Retry-After": "5"})
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(inference_pool, orchestrator.segment_image, request.model, params)
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except Exception as e:
            logger.error("Segmentation error", exc_info=True)
            return _error_response(503, f"Segmentation failed: {e}", error_type="server_error", code="segmentation_error")
        finally:
            inference_semaphore.release()
        return JSONResponse(content={
            "model": request.model,
            "segments": result.segments,
            "performance": {"total_time_ms": result.total_time_ms},
        })

    # ------------------------------------------------------------------
    # POST /v1/depth
    # ------------------------------------------------------------------

    @app.post("/v1/depth")
    async def estimate_depth(request: DepthEstimationRequest):
        params = DepthEstimationParams(image_b64=request.image)
        try:
            await asyncio.wait_for(inference_semaphore.acquire(), timeout=admission_timeout)
        except asyncio.TimeoutError:
            return _error_response(503, "Server busy.", error_type="server_error", code="capacity", headers={"Retry-After": "5"})
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(inference_pool, orchestrator.estimate_depth, request.model, params)
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except Exception as e:
            logger.error("Depth estimation error", exc_info=True)
            return _error_response(503, f"Depth estimation failed: {e}", error_type="server_error", code="depth_error")
        finally:
            inference_semaphore.release()
        return JSONResponse(content={
            "model": request.model,
            "depth_map_b64": result.depth_map_b64,
            "width": result.width,
            "height": result.height,
            "performance": {"total_time_ms": result.total_time_ms},
        })

    # ------------------------------------------------------------------
    # POST /v1/document-qa
    # ------------------------------------------------------------------

    @app.post("/v1/document-qa")
    async def document_qa(request: DocumentQARequest):
        params = DocumentQAParams(image_b64=request.image, question=request.question)
        try:
            await asyncio.wait_for(inference_semaphore.acquire(), timeout=admission_timeout)
        except asyncio.TimeoutError:
            return _error_response(503, "Server busy.", error_type="server_error", code="capacity", headers={"Retry-After": "5"})
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(inference_pool, orchestrator.answer_document, request.model, params)
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except Exception as e:
            logger.error("Document QA error", exc_info=True)
            return _error_response(503, f"Document QA failed: {e}", error_type="server_error", code="document_qa_error")
        finally:
            inference_semaphore.release()
        return JSONResponse(content={
            "model": request.model,
            "answer": result.answer,
            "score": result.score,
            "performance": {"total_time_ms": result.total_time_ms},
        })

    # ------------------------------------------------------------------
    # POST /v1/audio/process
    # ------------------------------------------------------------------

    @app.post("/v1/audio/process")
    async def process_audio(request: AudioProcessingRequest):
        params = AudioProcessingParams(audio_b64=request.audio)
        try:
            await asyncio.wait_for(inference_semaphore.acquire(), timeout=admission_timeout)
        except asyncio.TimeoutError:
            return _error_response(503, "Server busy.", error_type="server_error", code="capacity", headers={"Retry-After": "5"})
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(inference_pool, orchestrator.process_audio, request.model, params)
        except ModelNotFoundError as e:
            return _error_response(404, str(e), param="model", code="model_not_found")
        except Exception as e:
            logger.error("Audio processing error", exc_info=True)
            return _error_response(503, f"Audio processing failed: {e}", error_type="server_error", code="audio_processing_error")
        finally:
            inference_semaphore.release()
        audio_format = result.content_type.split("/")[-1] if "/" in result.content_type else "wav"
        return Response(
            content=result.audio_bytes,
            media_type=result.content_type,
            headers={
                "Content-Disposition": f"attachment; filename=processed.{audio_format}",
                "X-Total-Time-Ms": f"{result.total_time_ms:.0f}" if result.total_time_ms else "0",
            },
        )

    # ------------------------------------------------------------------
    # GET /v1/models
    # ------------------------------------------------------------------

    @app.get("/v1/models")
    async def list_models():
        records = registry.list_all()
        models = []
        for r in records:
            models.append({
                "id": r.model_id,
                "object": "model",
                "created": int(r.pulled_at.timestamp()) if r.pulled_at else 0,
                "owned_by": r.model_id.split("/")[0] if "/" in r.model_id else "local",
                "task": r.task.value,
            })
        return {"object": "list", "data": models}

    # ------------------------------------------------------------------
    # GET /v1/models/{model}
    # ------------------------------------------------------------------

    @app.get("/v1/models/{model:path}")
    async def get_model(model: str):
        record = registry.get(model)
        if record is None:
            return _error_response(404, f"Model '{model}' not found.", code="model_not_found")
        return {
            "id": record.model_id,
            "object": "model",
            "created": int(record.pulled_at.timestamp()) if record.pulled_at else 0,
            "owned_by": record.model_id.split("/")[0] if "/" in record.model_id else "local",
            "task": record.task.value,
            "format": record.format.value,
            "file_size_bytes": record.file_size_bytes,
            "param_count": record.param_count,
        }

    # ------------------------------------------------------------------
    # DELETE /v1/models/{model}
    # ------------------------------------------------------------------

    @app.delete("/v1/models/{model:path}")
    async def delete_model(model: str):
        record = registry.get(model)
        if record is None:
            return _error_response(404, f"Model '{model}' not found.", code="model_not_found")
        registry.remove(model)
        return {
            "id": record.model_id,
            "object": "model",
            "deleted": True,
        }

    # ------------------------------------------------------------------
    # Files API
    # ------------------------------------------------------------------

    @app.post("/v1/files")
    async def upload_file(
        file: UploadFile = File(...),
        purpose: str = Form(...),
    ):
        if file_store is None or files_dir is None:
            return _error_response(501, "Files API not configured.", code="not_configured")

        from inferall.registry.file_store import validate_file
        import shutil

        # Enforce upload size limit (100MB default)
        max_upload_bytes = 100 * 1024 * 1024
        content = await file.read()
        if len(content) > max_upload_bytes:
            return _error_response(
                413, f"File too large ({len(content)} bytes). Max: {max_upload_bytes} bytes.",
                code="file_too_large",
            )

        filename = file.filename or "unknown"

        # Validate
        error = validate_file(purpose, filename, content)
        if error:
            return _error_response(400, error, param="file")

        # Generate file ID and save to disk
        # Sanitize filename — strip path components to prevent traversal
        safe_filename = Path(filename).name  # strips all directory components
        safe_filename = safe_filename.replace("..", "").replace("/", "").replace("\\", "")
        if not safe_filename:
            safe_filename = "upload"

        file_id = f"file-{uuid.uuid4().hex[:24]}"
        file_dir = files_dir / file_id
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / safe_filename

        # Verify the resolved path is inside the expected directory
        if not str(file_path.resolve()).startswith(str(file_dir.resolve())):
            return _error_response(400, "Invalid filename.", param="file")

        file_path.write_bytes(content)

        # Create DB record
        result = file_store.create(
            filename=safe_filename,
            purpose=purpose,
            size_bytes=len(content),
            local_path=str(file_path),
        )
        # Override the auto-generated ID with our pre-generated one
        with file_store.conn:
            file_store.conn.execute(
                "UPDATE files SET file_id = ? WHERE file_id = ?",
                (file_id, result["id"]),
            )
        result["id"] = file_id

        return JSONResponse(content=result)

    @app.get("/v1/files")
    async def list_files(purpose: Optional[str] = None):
        if file_store is None:
            return _error_response(501, "Files API not configured.", code="not_configured")
        files = file_store.list_files(purpose=purpose)
        return {"object": "list", "data": files}

    @app.get("/v1/files/{file_id}")
    async def retrieve_file(file_id: str):
        if file_store is None:
            return _error_response(501, "Files API not configured.", code="not_configured")
        f = file_store.get(file_id)
        if f is None:
            return _error_response(404, f"File '{file_id}' not found.", code="file_not_found")
        return JSONResponse(content=f)

    @app.delete("/v1/files/{file_id}")
    async def delete_file(file_id: str):
        if file_store is None:
            return _error_response(501, "Files API not configured.", code="not_configured")

        import shutil

        local_path = file_store.get_local_path(file_id)
        if local_path is None:
            return _error_response(404, f"File '{file_id}' not found.", code="file_not_found")

        # Delete from disk — only if path is inside our files directory
        path = Path(local_path)
        if path.exists() and files_dir:
            # Verify the path is inside our files directory before deleting
            try:
                path.resolve().relative_to(files_dir.resolve())
                # Safe — delete the file's directory (file_id dir)
                shutil.rmtree(path.parent, ignore_errors=True)
            except ValueError:
                logger.warning("Refusing to delete file outside files_dir: %s", path)

        file_store.delete(file_id)
        return {"id": file_id, "object": "file", "deleted": True}

    @app.get("/v1/files/{file_id}/content")
    async def download_file_content(file_id: str):
        if file_store is None:
            return _error_response(501, "Files API not configured.", code="not_configured")

        local_path = file_store.get_local_path(file_id)
        if local_path is None:
            return _error_response(404, f"File '{file_id}' not found.", code="file_not_found")

        path = Path(local_path)
        if not path.exists():
            return _error_response(404, "File content not found on disk.", code="file_not_found")

        return Response(
            content=path.read_bytes(),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={path.name}"},
        )

    # ==================================================================
    # Assistants API
    # ==================================================================

    def _guard_assistants():
        if assistants_store is None:
            return _error_response(501, "Assistants API not configured.", code="not_configured")
        return None

    # -- Assistants CRUD --

    @app.post("/v1/assistants")
    async def create_assistant(request: CreateAssistantRequest):
        if (err := _guard_assistants()): return err
        result = assistants_store.create_assistant(
            model=request.model, name=request.name,
            instructions=request.instructions, tools=request.tools,
            file_ids=request.file_ids, metadata=request.metadata,
        )
        return JSONResponse(content=result)

    @app.get("/v1/assistants")
    async def list_assistants(limit: int = 20, order: str = "desc"):
        if (err := _guard_assistants()): return err
        return {"object": "list", "data": assistants_store.list_assistants(limit=limit, order=order)}

    @app.get("/v1/assistants/{assistant_id}")
    async def retrieve_assistant(assistant_id: str):
        if (err := _guard_assistants()): return err
        a = assistants_store.get_assistant(assistant_id)
        if not a: return _error_response(404, "Assistant not found.", code="not_found")
        return JSONResponse(content=a)

    @app.post("/v1/assistants/{assistant_id}")
    async def modify_assistant(assistant_id: str, request: ModifyAssistantRequest):
        if (err := _guard_assistants()): return err
        a = assistants_store.update_assistant(assistant_id, **request.model_dump(exclude_none=True))
        if not a: return _error_response(404, "Assistant not found.", code="not_found")
        return JSONResponse(content=a)

    @app.delete("/v1/assistants/{assistant_id}")
    async def delete_assistant(assistant_id: str):
        if (err := _guard_assistants()): return err
        if not assistants_store.delete_assistant(assistant_id):
            return _error_response(404, "Assistant not found.", code="not_found")
        return {"id": assistant_id, "object": "assistant.deleted", "deleted": True}

    # -- Threads CRUD --

    @app.post("/v1/threads")
    async def create_thread(request: CreateThreadRequest = None):
        if (err := _guard_assistants()): return err
        meta = request.metadata if request else {}
        return JSONResponse(content=assistants_store.create_thread(metadata=meta))

    @app.get("/v1/threads/{thread_id}")
    async def retrieve_thread(thread_id: str):
        if (err := _guard_assistants()): return err
        t = assistants_store.get_thread(thread_id)
        if not t: return _error_response(404, "Thread not found.", code="not_found")
        return JSONResponse(content=t)

    @app.post("/v1/threads/{thread_id}")
    async def modify_thread(thread_id: str, request: CreateThreadRequest):
        if (err := _guard_assistants()): return err
        t = assistants_store.update_thread(thread_id, request.metadata or {})
        if not t: return _error_response(404, "Thread not found.", code="not_found")
        return JSONResponse(content=t)

    @app.delete("/v1/threads/{thread_id}")
    async def delete_thread(thread_id: str):
        if (err := _guard_assistants()): return err
        if not assistants_store.delete_thread(thread_id):
            return _error_response(404, "Thread not found.", code="not_found")
        return {"id": thread_id, "object": "thread.deleted", "deleted": True}

    # -- Messages --

    @app.post("/v1/threads/{thread_id}/messages")
    async def create_message(thread_id: str, request: CreateMessageRequest):
        if (err := _guard_assistants()): return err
        if not assistants_store.get_thread(thread_id):
            return _error_response(404, "Thread not found.", code="not_found")
        msg = assistants_store.create_message(
            thread_id=thread_id, role=request.role, content=request.content,
            file_ids=request.file_ids, metadata=request.metadata,
        )
        return JSONResponse(content=msg)

    @app.get("/v1/threads/{thread_id}/messages")
    async def list_messages(thread_id: str, limit: int = 20, order: str = "desc"):
        if (err := _guard_assistants()): return err
        return {"object": "list", "data": assistants_store.list_messages(thread_id, limit=limit, order=order)}

    @app.get("/v1/threads/{thread_id}/messages/{message_id}")
    async def retrieve_message(thread_id: str, message_id: str):
        if (err := _guard_assistants()): return err
        m = assistants_store.get_message(thread_id, message_id)
        if not m: return _error_response(404, "Message not found.", code="not_found")
        return JSONResponse(content=m)

    # -- Runs --

    @app.post("/v1/threads/{thread_id}/runs")
    async def create_run(thread_id: str, request: CreateRunRequest):
        if (err := _guard_assistants()): return err
        if not assistants_store.get_thread(thread_id):
            return _error_response(404, "Thread not found.", code="not_found")
        try:
            run = assistants_store.create_run(
                thread_id=thread_id, assistant_id=request.assistant_id,
                model=request.model, instructions=request.instructions,
                metadata=request.metadata,
            )
        except ValueError as e:
            return _error_response(404, str(e), code="not_found")

        # Execute in background
        from inferall.registry.assistants_store import execute_run
        inference_pool.submit(execute_run, run["id"], thread_id, assistants_store, orchestrator)

        return JSONResponse(content=run)

    @app.get("/v1/threads/{thread_id}/runs")
    async def list_runs(thread_id: str, limit: int = 20, order: str = "desc"):
        if (err := _guard_assistants()): return err
        return {"object": "list", "data": assistants_store.list_runs(thread_id, limit=limit, order=order)}

    @app.get("/v1/threads/{thread_id}/runs/{run_id}")
    async def retrieve_run(thread_id: str, run_id: str):
        if (err := _guard_assistants()): return err
        r = assistants_store.get_run(thread_id, run_id)
        if not r: return _error_response(404, "Run not found.", code="not_found")
        return JSONResponse(content=r)

    @app.post("/v1/threads/{thread_id}/runs/{run_id}/cancel")
    async def cancel_run(thread_id: str, run_id: str):
        if (err := _guard_assistants()): return err
        r = assistants_store.get_run(thread_id, run_id)
        if not r: return _error_response(404, "Run not found.", code="not_found")
        if r["status"] not in ("queued", "in_progress"):
            return _error_response(400, f"Cannot cancel run with status '{r['status']}'.")
        from inferall.registry.assistants_store import _now_iso
        assistants_store.update_run_status(run_id, "cancelled", cancelled_at=_now_iso())
        return JSONResponse(content=assistants_store.get_run(thread_id, run_id))

    # ==================================================================
    # Fine-tuning API
    # ==================================================================

    _SUPPORTED_BATCH_ENDPOINTS = {"/v1/chat/completions", "/v1/embeddings", "/v1/completions"}

    @app.post("/v1/fine_tuning/jobs")
    async def create_fine_tuning_job(request: CreateFineTuningJobRequest):
        if fine_tuning_store is None:
            return _error_response(501, "Fine-tuning API not configured.", code="not_configured")
        job = fine_tuning_store.create_job(
            model=request.model, training_file=request.training_file,
            validation_file=request.validation_file, hyperparameters=request.hyperparameters,
        )
        from inferall.registry.jobs_store import execute_fine_tuning_job
        inference_pool.submit(execute_fine_tuning_job, job["id"], fine_tuning_store)
        return JSONResponse(content=job)

    @app.get("/v1/fine_tuning/jobs")
    async def list_fine_tuning_jobs(limit: int = 20, after: Optional[str] = None):
        if fine_tuning_store is None:
            return _error_response(501, "Fine-tuning API not configured.", code="not_configured")
        return {"object": "list", "data": fine_tuning_store.list_jobs(limit=limit, after=after)}

    @app.get("/v1/fine_tuning/jobs/{job_id}")
    async def retrieve_fine_tuning_job(job_id: str):
        if fine_tuning_store is None:
            return _error_response(501, "Fine-tuning API not configured.", code="not_configured")
        job = fine_tuning_store.get_job(job_id)
        if not job: return _error_response(404, "Job not found.", code="not_found")
        return JSONResponse(content=job)

    @app.post("/v1/fine_tuning/jobs/{job_id}/cancel")
    async def cancel_fine_tuning_job(job_id: str):
        if fine_tuning_store is None:
            return _error_response(501, "Fine-tuning API not configured.", code="not_configured")
        if not fine_tuning_store.cancel_job(job_id):
            return _error_response(400, "Job cannot be cancelled.", code="invalid_status")
        return JSONResponse(content=fine_tuning_store.get_job(job_id))

    @app.get("/v1/fine_tuning/jobs/{job_id}/events")
    async def list_fine_tuning_events(job_id: str, limit: int = 20):
        if fine_tuning_store is None:
            return _error_response(501, "Fine-tuning API not configured.", code="not_configured")
        return {"object": "list", "data": fine_tuning_store.list_events(job_id, limit=limit)}

    @app.get("/v1/fine_tuning/jobs/{job_id}/checkpoints")
    async def list_fine_tuning_checkpoints(job_id: str, limit: int = 10):
        if fine_tuning_store is None:
            return _error_response(501, "Fine-tuning API not configured.", code="not_configured")
        return {"object": "list", "data": fine_tuning_store.list_checkpoints(job_id, limit=limit)}

    # ==================================================================
    # Batch API
    # ==================================================================

    @app.post("/v1/batches")
    async def create_batch(request: CreateBatchRequest):
        if batch_store is None:
            return _error_response(501, "Batch API not configured.", code="not_configured")
        if request.endpoint not in _SUPPORTED_BATCH_ENDPOINTS:
            return _error_response(400,
                f"Unsupported endpoint '{request.endpoint}'. Supported: {', '.join(sorted(_SUPPORTED_BATCH_ENDPOINTS))}",
                param="endpoint")
        batch = batch_store.create_batch(
            input_file_id=request.input_file_id, endpoint=request.endpoint,
            completion_window=request.completion_window, metadata=request.metadata,
        )
        from inferall.registry.jobs_store import execute_batch
        inference_pool.submit(
            execute_batch, batch["id"], batch_store, file_store, orchestrator,
            files_dir or Path.home() / ".inferall" / "files",
        )
        return JSONResponse(content=batch)

    @app.get("/v1/batches")
    async def list_batches(limit: int = 20, after: Optional[str] = None):
        if batch_store is None:
            return _error_response(501, "Batch API not configured.", code="not_configured")
        return {"object": "list", "data": batch_store.list_batches(limit=limit, after=after)}

    @app.get("/v1/batches/{batch_id}")
    async def retrieve_batch(batch_id: str):
        if batch_store is None:
            return _error_response(501, "Batch API not configured.", code="not_configured")
        batch = batch_store.get_batch(batch_id)
        if not batch: return _error_response(404, "Batch not found.", code="not_found")
        return JSONResponse(content=batch)

    @app.post("/v1/batches/{batch_id}/cancel")
    async def cancel_batch(batch_id: str):
        if batch_store is None:
            return _error_response(501, "Batch API not configured.", code="not_configured")
        if not batch_store.cancel_batch(batch_id):
            return _error_response(400, "Batch cannot be cancelled.", code="invalid_status")
        return JSONResponse(content=batch_store.get_batch(batch_id))

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health_check():
        loaded = orchestrator.list_loaded()
        return {
            "status": "ok",
            "loaded_models": len(loaded),
            "capabilities": {
                "chat_completions": True,
                "streaming": True,
                "tools": True,
                "embeddings": True,
                "audio_transcriptions": True,
                "image_generations": True,
                "image_edits": True,
                "video_generations": True,
                "seq2seq": True,
                "classification": True,
                "object_detection": True,
                "image_segmentation": True,
                "depth_estimation": True,
                "document_qa": True,
                "audio_processing": True,
                "audio_speech": True,
                "reranking": True,
                "completions": True,
                "moderations": True,
                "audio_translations": True,
                "image_variations": True,
                "files": file_store is not None,
                "assistants": assistants_store is not None,
                "fine_tuning": fine_tuning_store is not None,
                "batches": batch_store is not None,
            },
            "compat_mode": compat_mode,
        }

    # ------------------------------------------------------------------
    # WebSocket /v1/ws/chat — Persistent streaming connection
    # ------------------------------------------------------------------

    @app.websocket("/v1/ws/chat")
    async def ws_chat(websocket: WebSocket):
        # Authenticate WebSocket handshake (HTTP middleware doesn't cover WebSockets)
        if api_key is not None:
            # Check query param or first message for auth
            token = websocket.query_params.get("token", "")
            if token != api_key:
                await websocket.close(code=4001, reason="Authentication required. Pass ?token=<api_key>")
                return
        from inferall.api.websocket import websocket_chat
        await websocket_chat(websocket, orchestrator, dispatcher)

    # ------------------------------------------------------------------
    # GET /v1/queue/stats — Dispatcher queue monitoring
    # ------------------------------------------------------------------

    @app.get("/v1/queue/stats")
    async def queue_stats():
        stats = dispatcher.get_stats()
        gpu_summary = []
        try:
            from inferall.gpu.manager import GPUManager
            gm = GPUManager()
            gpu_summary = gm.get_gpu_utilization_summary()
        except Exception:
            pass

        return {
            "object": "queue.stats",
            "models": {
                mid: {
                    "pending": s.pending,
                    "active": s.active,
                    "total_served": s.total_served,
                    "total_errors": s.total_errors,
                    "avg_latency_ms": round(s.avg_latency_ms, 1),
                }
                for mid, s in stats.items()
            },
            "gpus": gpu_summary,
        }

    return app


# =============================================================================
# Response Formatting
# =============================================================================

def _format_completion_response(
    result: GenerationResult,
    model: str,
    completion_id: str,
    created: int,
) -> JSONResponse:
    """Format a non-streaming OpenAI-compatible response."""
    # Build message — may include tool_calls
    message: dict = {"role": "assistant", "content": result.text}

    if result.tool_calls:
        message["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function_name,
                    "arguments": tc.function_arguments,
                },
            }
            for tc in result.tool_calls
        ]
        # When tool_calls are present, content may be None
        if not result.text:
            message["content"] = None

    return JSONResponse(content={
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": result.finish_reason,
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
        "performance": {
            "total_time_ms": result.total_time_ms,
            "tokens_per_second": result.tokens_per_second,
        },
        "system_fingerprint": None,
    })


# =============================================================================
# Streaming
# =============================================================================

async def _stream_generator(
    orchestrator: Orchestrator,
    model_id: str,
    messages: List[dict],
    params: GenerationParams,
    completion_id: str,
    created: int,
    semaphore: asyncio.Semaphore,
    pool: ThreadPoolExecutor,
) -> AsyncIterator[dict]:
    """
    Async generator that yields SSE events for streaming.

    Uses a queue.Queue bridge: the sync orchestrator.stream() runs in the
    thread pool and puts tokens into the queue; this async generator polls
    with get_nowait() + asyncio.sleep().
    """
    import queue

    token_queue: queue.Queue = queue.Queue()
    cancel_event = threading.Event()
    error_holder: list = []
    done_event = threading.Event()
    t0 = time.perf_counter()
    token_count = 0

    def _run_stream():
        try:
            for token in orchestrator.stream(model_id, messages, params, cancel_event):
                token_queue.put(token)
        except Exception as e:
            error_holder.append(e)
        finally:
            done_event.set()

    # Start the sync stream in the thread pool
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(pool, _run_stream)

    try:
        # First chunk: role announcement
        yield {
            "data": json.dumps({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                    "logprobs": None,
                }],
                "system_fingerprint": None,
            })
        }

        # Content chunks
        while True:
            try:
                token = token_queue.get_nowait()
                token_count += 1
                yield {
                    "data": json.dumps({
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                            "logprobs": None,
                        }],
                        "system_fingerprint": None,
                    })
                }
            except queue.Empty:
                if done_event.is_set() and token_queue.empty():
                    break
                await asyncio.sleep(0.01)

        # Check for errors
        if error_holder:
            error = error_holder[0]
            if isinstance(error, ModelNotFoundError):
                yield {
                    "data": json.dumps({
                        "error": {
                            "message": str(error),
                            "type": "invalid_request_error",
                            "code": "model_not_found",
                        }
                    })
                }
            else:
                yield {
                    "data": json.dumps({
                        "error": {
                            "message": f"Generation error: {error}",
                            "type": "server_error",
                            "code": "generation_error",
                        }
                    })
                }
        else:
            # Final chunk: finish reason + performance metrics
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            tps = token_count / (elapsed_ms / 1000.0) if elapsed_ms > 0 and token_count > 0 else 0
            logger.info(
                "Stream %s: %d tokens in %.0fms (%.1f tok/s)",
                model_id, token_count, elapsed_ms, tps,
            )
            yield {
                "data": json.dumps({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                        "logprobs": None,
                    }],
                    "usage": {
                        "completion_tokens": token_count,
                    },
                    "performance": {
                        "total_time_ms": round(elapsed_ms, 1),
                        "tokens_per_second": round(tps, 1),
                    },
                    "system_fingerprint": None,
                })
            }

        # [DONE] sentinel
        yield {"data": "[DONE]"}

    except asyncio.CancelledError:
        # Client disconnected
        cancel_event.set()
        raise
    finally:
        cancel_event.set()
        semaphore.release()
        # Wait for the thread to finish
        try:
            await asyncio.wait_for(future, timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            pass

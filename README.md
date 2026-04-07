# InferAll

**Run any AI model locally — one unified API for chat, vision, speech, images, video, and more. Built for multi-user serving.**

InferAll is a self-hosted inference server that exposes an **OpenAI-compatible REST API** for every type of AI model. Point any OpenAI SDK client, LangChain, LlamaIndex, or custom application at InferAll and it just works — no code changes needed.

### What it does

- **One API for everything** — 17 model types through standard OpenAI endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/images/generations`, `/v1/audio/transcriptions`, and 50+ more)
- **Runs as a server** — start it with `inferall serve` and any client on your network can connect
- **Multi-user ready** — per-API-key rate limiting, priority levels, and per-model request queuing so one user's request never blocks another's
- **Pull from anywhere** — models from HuggingFace Hub, Ollama registry, or Ollama cloud, all through one CLI
- **GPU optimized** — multi-GPU scheduling with load balancing, VRAM-aware allocation, GGUF at full speed (113 tok/s on RTX 4090), plus fp16/GPTQ/AWQ/BNB quantization
- **Optional vLLM acceleration** — opt any chat or VLM model into a high-throughput vLLM backend for ~50% faster single-stream inference and continuous batching under load (chandra-ocr-2: 31.6 → 48.2 tok/s on RTX 4090)
- **Production features** — Assistants API with threads and runs, Files API, Batch processing, Fine-tuning API, tool/function calling, structured JSON output
- **Built-in dashboard** — terminal UI for real-time GPU monitoring, request queues, performance metrics, and model management

### Supported model types

Chat/LLM · Embeddings · Reranking · Vision-Language · Speech Recognition · Text-to-Speech · Image Generation · Image-to-Image · Video Generation · Translation · Summarization · Classification · Object Detection · Segmentation · Depth Estimation · Document QA · Audio Processing

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (CPU fallback available)
- ~2GB disk for base install (models downloaded separately)

## Installation

### 1. Clone and create virtual environment

```bash
git clone https://github.com/GravenSm/inferall.git
cd inferall
python3 -m venv .venv
source .venv/bin/activate
```

> **Note:** If your filesystem doesn't support symlinks (NTFS, exFAT), use `python3 -m venv --copies .venv` or create the venv on a native Linux filesystem.

### 2. Install PyTorch first (with CUDA)

```bash
pip install torch>=2.0
```

Verify CUDA works:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 3. Install InferAll

**Minimal** (chat + embeddings):
```bash
pip install -e .
```

**Full install** (all model types):
```bash
pip install -e ".[all]"
```

**Custom install** (pick what you need):
```bash
# GGUF support (llama.cpp)
pip install -e ".[gguf]"

# Quantized models
pip install -e ".[bnb]"     # bitsandbytes 4/8-bit
pip install -e ".[gptq]"    # GPTQ models
pip install -e ".[awq]"     # AWQ models

# Multi-modal
pip install -e ".[multimodal]"  # embeddings + diffusion + ASR + TTS

# Development
pip install -e ".[dev]"     # pytest + httpx
```

### 4. Extra dependencies for specific tasks

```bash
# SSE streaming (required for streaming chat)
pip install sse-starlette

# Object detection (DETR, YOLO)
pip install timm

# Document QA (LayoutLM — needs Tesseract OCR)
pip install pytesseract
# Also install system tesseract: sudo apt install tesseract-ocr

# Video generation (optional MP4 encoding)
pip install imageio[ffmpeg]

# VLM models (Qwen-VL, etc.)
pip install torchvision
```

### 5. GGUF with CUDA (for GPU-accelerated llama.cpp)

The default `llama-cpp-python` pip install is CPU-only. For GPU acceleration:

```bash
# Install pre-built CUDA wheel
pip install llama-cpp-python --force-reinstall \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Set library path (add to your shell profile)
export LD_LIBRARY_PATH="$(python -c 'import nvidia.cuda_runtime; print(nvidia.cuda_runtime.__path__[0])')/lib:$LD_LIBRARY_PATH"
```

## Quick Start

### Pull a model

Models can be pulled from **HuggingFace Hub** or **Ollama's registry**. The source is auto-detected:

```bash
# From HuggingFace (org/model format)
inferall pull Qwen/Qwen2.5-1.5B-Instruct
inferall pull sentence-transformers/all-MiniLM-L6-v2

# From Ollama (short name = Ollama registry)
inferall pull llama3.1
inferall pull llama3.1:70b
inferall pull codellama

# Force a specific source
inferall pull --source ollama gemma2
inferall pull --source hf google/gemma-2-2b-it
```

Ollama models are GGUF files served from `registry.ollama.ai` — they work with the llama.cpp backend just like HuggingFace GGUF models.

### Chat interactively

```bash
inferall run Qwen/Qwen2.5-1.5B-Instruct
```

Commands inside the REPL:
- Type your message and press Enter
- `/system <prompt>` — set system prompt
- `/clear` — reset conversation
- `/params` — show generation parameters
- `/exit` or Ctrl+D — quit
- End a line with `\` for multi-line input

### Start the API server

```bash
inferall serve
```

With options:
```bash
inferall serve --port 8080 --host 0.0.0.0 --api-key mykey --workers 4
```

Or via environment variables:
```bash
INFERALL_PORT=8080 INFERALL_API_KEY=mykey inferall serve
```

### List pulled models

```bash
inferall list
```

### Check GPU status

```bash
inferall status
```

### Remove a model

```bash
inferall remove Qwen/Qwen2.5-1.5B-Instruct
```

### vLLM acceleration (optional)

```bash
inferall vllm install                              # bootstrap isolated vllm venv
inferall vllm enable datalab-to/chandra-ocr-2     # opt a model in
inferall vllm disable datalab-to/chandra-ocr-2    # revert to default backend
inferall vllm status                              # show runtime location
```

See the [vLLM Backend](#vllm-backend-optional-high-throughput) section below for details.

## API Reference

All endpoints are OpenAI-compatible where applicable. The server runs at `http://127.0.0.1:8000` by default.

### Chat Completion

```bash
# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Embeddings

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": ["Hello world", "How are you?"]
  }'
```

### Reranking

```bash
curl http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "query": "What is Python?",
    "documents": ["Python is a snake", "Python is a programming language"],
    "top_n": 2,
    "return_documents": true
  }'
```

### Translation (Seq2seq)

```bash
curl http://localhost:8000/v1/text/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Helsinki-NLP/opus-mt-en-fr",
    "input": "Hello, how are you today?",
    "num_beams": 4
  }'
```

### Image Generation

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/sdxl-turbo",
    "prompt": "a cat sitting on a chair",
    "size": "512x512",
    "num_inference_steps": 1,
    "guidance_scale": 0.0
  }'
```

### Image-to-Image

```bash
curl http://localhost:8000/v1/images/edits \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-img2img-model",
    "prompt": "make it a watercolor painting",
    "image": "<base64-encoded-image>",
    "strength": 0.7
  }'
```

### Video Generation

```bash
curl http://localhost:8000/v1/videos/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-video-model",
    "prompt": "a cat running in a field",
    "num_frames": 16,
    "fps": 8,
    "size": "512x512"
  }'
```

### Speech Recognition (ASR)

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=openai/whisper-tiny"
```

### Text-to-Speech

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "suno/bark-small",
    "input": "Hello world"
  }' -o speech.wav
```

### Classification

```bash
# Image classification
curl http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/vit-base-patch16-224",
    "image": "<base64-image>",
    "top_k": 5
  }'

# Zero-shot text classification
curl http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/bart-large-mnli",
    "text": "The stock market crashed today",
    "candidate_labels": ["politics", "finance", "sports"]
  }'
```

### Object Detection

```bash
curl http://localhost:8000/v1/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/detr-resnet-50",
    "image": "<base64-image>",
    "threshold": 0.5
  }'
```

### Image Segmentation

```bash
curl http://localhost:8000/v1/segment \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mattmdjaga/segformer_b2_clothes",
    "image": "<base64-image>"
  }'
```

### Depth Estimation

```bash
curl http://localhost:8000/v1/depth \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LiheYoung/depth-anything-small-hf",
    "image": "<base64-image>"
  }'
```

### Document QA

```bash
curl http://localhost:8000/v1/document-qa \
  -H "Content-Type: application/json" \
  -d '{
    "model": "impira/layoutlm-document-qa",
    "image": "<base64-document-image>",
    "question": "What is the invoice number?"
  }'
```

### Audio Processing

```bash
curl http://localhost:8000/v1/audio/process \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-audio-model",
    "audio": "<base64-audio>"
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

### List Models

```bash
curl http://localhost:8000/v1/models
```

## Supported Model Types

| Task | Endpoint | Example Models | Quantization |
|------|----------|---------------|-------------|
| Chat / LLM | `/v1/chat/completions` | Llama, Qwen, Mistral | fp16, GPTQ, AWQ, BNB 4/8bit, GGUF |
| Embeddings | `/v1/embeddings` | all-MiniLM, BGE, E5 | fp16 |
| Reranking | `/v1/rerank` | ms-marco, bge-reranker | fp16 |
| Vision-Language | `/v1/chat/completions` | Qwen-VL, LLaVA | fp16 |
| Translation | `/v1/text/generate` | OPUS-MT, NLLB, mBART | fp16 |
| Summarization | `/v1/text/generate` | T5, FLAN-T5, BART | fp16 |
| Image Generation | `/v1/images/generations` | SDXL, Stable Diffusion | fp16 |
| Image-to-Image | `/v1/images/edits` | SD img2img, ControlNet | fp16 |
| Video Generation | `/v1/videos/generations` | CogVideoX, AnimateDiff | fp16 |
| Speech Recognition | `/v1/audio/transcriptions` | Whisper | fp16 |
| Text-to-Speech | `/v1/audio/speech` | Bark, SpeechT5 | fp16 |
| Classification | `/v1/classify` | ViT, CLIP, BART-MNLI | fp16 |
| Object Detection | `/v1/detect` | DETR, OWL-ViT | fp16 |
| Segmentation | `/v1/segment` | SAM, Mask2Former | fp16 |
| Depth Estimation | `/v1/depth` | Depth Anything, DPT | fp16 |
| Document QA | `/v1/document-qa` | LayoutLM, Donut | fp16 |
| Audio Processing | `/v1/audio/process` | Voice conversion | fp16 |

## Configuration

Configuration is loaded in layers (highest priority first):

1. **CLI flags** (`--port`, `--host`, etc.)
2. **Environment variables** (`INFERALL_PORT`, `INFERALL_HOST`, etc.)
3. **Config file** (`~/.inferall/config.yaml`)
4. **Built-in defaults**

### Config file example

```yaml
# ~/.inferall/config.yaml
default_port: 8000
default_host: "127.0.0.1"
idle_timeout: 300          # seconds before idle models are unloaded
vram_buffer_mb: 512        # VRAM headroom to keep free
max_loaded_models: 3       # max models in GPU memory simultaneously
inference_workers: 2       # thread pool size for inference
trust_remote_code: false
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERALL_PORT` | 8000 | API server port |
| `INFERALL_HOST` | 127.0.0.1 | Bind address |
| `INFERALL_API_KEY` | None | API key for auth |
| `INFERALL_IDLE_TIMEOUT` | 300 | Idle model eviction (seconds) |
| `INFERALL_VRAM_BUFFER_MB` | 512 | VRAM headroom (MB) |
| `INFERALL_MAX_LOADED` | 3 | Max loaded models |
| `INFERALL_WORKERS` | 2 | Inference threads |
| `INFERALL_BASE_DIR` | ~/.inferall | Data directory |

## vLLM Backend (optional, high-throughput)

For chat and vision-language models, InferAll can route inference through [vLLM](https://github.com/vllm-project/vllm) instead of the default HuggingFace transformers backend. vLLM uses PagedAttention, custom CUDA kernels, and continuous batching to deliver substantially higher throughput — especially for models with linear-attention layers (Qwen3-Next, chandra-ocr-2, etc.) where HF transformers can't apply its standard cache optimizations.

### Why a separate venv?

vLLM pins `transformers<5` while InferAll uses transformers 5.x, so embedding it directly would force a downgrade and rewrite of every existing backend. Instead, the vLLM backend runs vLLM as a subprocess in its own isolated venv and proxies requests over its OpenAI-compatible HTTP server. This is also how chandra and most production deployments run vLLM.

### Setup

```bash
# One-time bootstrap — creates ~/.cache/inferall/vllm-venv and installs vllm
inferall vllm install

# Check that the runtime is detected
inferall vllm status

# Opt a model into the vLLM backend (persists in the registry)
inferall vllm enable datalab-to/chandra-ocr-2

# Revert to the default backend
inferall vllm disable datalab-to/chandra-ocr-2

# Or point at an existing vllm install instead of bootstrapping
export INFERALL_VLLM_PYTHON=/path/to/vllm-venv/bin/python
```

After enabling, the next request to that model will spawn a vLLM subprocess and serve through it. Unloading the model (idle eviction, manual unload, or `inferall serve` shutdown) cleans up the subprocess and frees the GPU.

### Tuning

vLLM's defaults (`gpu_memory_utilization=0.9`, `max_num_seqs=256`) are sized for shared serving infrastructure and OOM almost immediately when other processes already use any GPU memory. InferAll picks conservative defaults and exposes four env vars for tuning:

| Variable | Default | Purpose |
|----------|---------|---------|
| `INFERALL_VLLM_GPU_MEMORY_UTILIZATION` | auto (≤0.85) | Fraction of total GPU memory vLLM may claim |
| `INFERALL_VLLM_MAX_MODEL_LEN` | 4096 | Cap on context length (larger = bigger KV cache) |
| `INFERALL_VLLM_MAX_NUM_SEQS` | 8 | Concurrent in-flight sequences |
| `INFERALL_VLLM_PYTHON` | (auto-detect) | Path override for the vLLM interpreter |

The auto memory budget is computed from currently free VRAM minus a 1.5 GiB safety buffer, then clamped to `[0.30, 0.85]`. Override it explicitly if you know exactly how much vLLM should claim.

## Performance

All responses include a `performance` section with timing data:

```json
{
  "performance": {
    "total_time_ms": 647.0,
    "tokens_per_second": 18.5
  }
}
```

Streaming responses include performance in the final SSE chunk.

### Benchmarks (RTX 4090)

| Model | Backend | tok/s |
|-------|---------|-------|
| Llama 3.1 8B | GGUF Q4_K_M | ~113 |
| Qwen 2.5 1.5B | Transformers fp16 | ~18.5 |
| chandra-ocr-2 (5.3B VLM) | Transformers fp16 (default) | 31.6 |
| chandra-ocr-2 (5.3B VLM) | **vLLM** | **48.2** |

## Architecture

```
inferall/
├── api/server.py          # FastAPI server, OpenAI-compatible endpoints
├── backends/
│   ├── base.py            # ABCs and data structures
│   ├── transformers_backend.py   # HF transformers (fp16/GPTQ/AWQ/BNB)
│   ├── llamacpp_backend.py       # GGUF via llama.cpp
│   ├── vllm_backend.py           # vLLM via subprocess + HTTP (opt-in)
│   ├── vllm_runtime.py           # vLLM venv discovery + bootstrap
│   ├── embedding_backend.py      # Sentence embeddings
│   ├── rerank_backend.py         # Cross-encoder reranking
│   ├── vlm_backend.py            # Vision-language models
│   ├── asr_backend.py            # Whisper ASR
│   ├── tts_backend.py            # Bark/SpeechT5 TTS
│   ├── diffusion_backend.py      # Text-to-image (diffusers)
│   ├── img2img_backend.py        # Image-to-image
│   ├── video_backend.py          # Text-to-video
│   ├── seq2seq_backend.py        # Translation/summarization
│   └── classification_backend.py # Classification, detection, segmentation, etc.
├── cli/                   # Typer CLI (pull, run, serve, list, status, remove, login, vllm)
├── gpu/
│   ├── manager.py         # GPU enumeration, VRAM tracking (pynvml)
│   └── allocator.py       # VRAM estimation, multi-GPU allocation, load balancing
├── registry/
│   ├── registry.py        # SQLite model registry with migrations
│   ├── metadata.py        # ModelTask, ModelFormat enums, preferred_engine
│   └── hf_resolver.py     # HuggingFace download + format auto-detection
├── orchestrator.py        # Model lifecycle, LRU eviction, ref counting
└── config.py              # Layered configuration
```

## License

MIT

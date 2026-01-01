<div align="center">
  <img src="assets/index_icon.png" width="250" />
</div>

<div align="center">
  <a href="docs/README_zh.md" style="font-size: 20px">简体中文</a> |
  <a href="README.md" style="font-size: 20px">English</a>
</div>

# IndexTTS (Deployment Fork)

This repository is a deployment-focused fork of the official IndexTTS project:
- Upstream: `https://github.com/index-tts/index-tts`

It keeps the original model usage, and adds a production-oriented **OpenAI-compatible TTS API** plus
performance/caching improvements for real servers.

## What’s Included (Fork Changes)

- OpenAI-compatible REST API (`FastAPI`): `POST /v1/audio/speech`
- Voice registry endpoints: `GET/POST /v1/audio/voices`, `GET /voices/{voice_id}/preview`
- In-memory audio responses (no per-request temp WAV write/read)
- Stable base64 voice caching to improve model cache hit rate across requests (`voices/_tmp/`)
- Timing instrumentation:
  - `--timing_log` writes a per-request breakdown to server logs
  - `--timing_headers` adds `X-IndexTTS-Time-*` response headers
- Speed toggles for NVIDIA GPUs:
  - `--fp16`, `--torch_compile`
  - `--accel` (requires `flash-attn`)
  - `--s2mel_amp`, `--vocoder_amp` (with automatic float32 fallback if AMP produces silent output)
- API-side advanced controls via `x_advanced` (see “API Design”)

## Installation (Recommended)

### 1) Prerequisites

- `git` and `git-lfs`

```bash
git lfs install
```

### 2) Clone

```bash
git clone https://github.com/Charlie237/index-tts.git
cd index-tts
git lfs pull
```

### 3) Install `uv`

See: `https://docs.astral.sh/uv/getting-started/installation/`

Quick install:

```bash
pip install -U uv
```

### 4) Install dependencies

Full install (everything):

```bash
uv sync --all-extras
```

API-only:

```bash
uv sync --extra api
```

WebUI-only:

```bash
uv sync --extra webui
```

### 5) Download models

HuggingFace:

```bash
uv tool install "huggingface-hub[cli,hf_xet]"
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
```

ModelScope:

```bash
uv tool install "modelscope"
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

If your network is slow to HuggingFace:

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

## Quickstart (WebUI)

```bash
uv run webui.py
```

Then open `http://127.0.0.1:7860`.

## Quickstart (OpenAI-Compatible API Server)

### Start (recommended baseline)

```bash
uv run api_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model_version v2 \
  --model_dir checkpoints \
  --fp16 \
  --torch_compile \
  --max_concurrent_inferences 1 \
  --timing_log
```

### Optional: enable acceleration (`--accel`)

`--accel` requires `flash-attn` (not part of default dependencies because installs are highly environment-specific):

```bash
uv pip install flash-attn --no-build-isolation
```

Then:

```bash
uv run api_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model_version v2 \
  --model_dir checkpoints \
  --fp16 \
  --torch_compile \
  --accel \
  --max_concurrent_inferences 1 \
  --timing_log
```

### Optional: DeepSpeed (`--deepspeed`)

DeepSpeed is not enabled by default. If you want to try it:

```bash
uv sync --extra deepspeed
```

Then add `--deepspeed` to `api_server.py`. Performance impact varies by system; benchmark both ways.

### Optional: BigVGAN custom CUDA kernel (`--cuda_kernel`)

This requires a CUDA Toolkit toolchain (`nvcc`) and `ninja` that match your PyTorch CUDA build.
On managed compute platforms that only provide CUDA runtime (no Toolkit), keep this disabled.

### Notes

- Lowest latency: use `response_format="wav"` (other formats invoke `ffmpeg` conversion per request).
- `--max_concurrent_inferences` defaults to `1` for stability and predictable GPU usage.

## API Design

### Models

The server loads **one** model per process:
- `--model_version v2` → IndexTTS2 (`indextts/infer_v2.py`)
- `--model_version v1` → IndexTTS (`indextts/infer.py`)

`GET /v1/models` returns both IDs for compatibility, but the active model is chosen by `--model_version`.

### Endpoints

- `GET /health`
- `GET /v1/models`
- `GET /v1/audio/voices`
- `POST /v1/audio/voices`
- `GET /voices/{voice_id}/preview`
- `POST /v1/audio/speech`

### `POST /v1/audio/speech`

Request body (OpenAI-compatible + extensions):

```json
{
  "model": "indextts2",
  "input": "你好，欢迎使用 IndexTTS2。",
  "voice": "nahida",
  "response_format": "wav",
  "x_voice_audio": null,
  "x_emotion": {
    "type": "vector",
    "vector": [0, 0, 0, 0, 0, 0, 0.4, 0.0],
    "alpha": 0.65,
    "random": false
  },
  "x_advanced": {
    "max_tokens_per_segment": 120,
    "max_mel_tokens": 2500,
    "do_sample": true,
    "temperature": 0.8,
    "top_p": 0.8,
    "top_k": 30,
    "num_beams": 3,
    "length_penalty": 0.0,
    "repetition_penalty": 10.0,
    "interval_silence": 200,
    "diffusion_steps": 25,
    "inference_cfg_rate": 0.7
  }
}
```

Key fields:
- `voice`: registered voice ID (see `GET /v1/audio/voices`) or a resolvable file path.
- `x_voice_audio`: base64 voice reference audio. If present, it overrides `voice`.
- `x_emotion` (v2 only): emotion control (vector/reference/text).
- `x_advanced`: performance/quality knobs.

### Timing / Profiling

Enable log output:

```bash
--timing_log
```

Example log line:

```
Timings(ms): voice_resolve=0.33 infer=33988.48 convert=2.19 total=33991.56
```

Enable response headers:

```bash
--timing_headers
```

## Performance & Caching Notes

- Model caches are **in-process** (cleared on restart).
- v2 caches the *last* speaker/emotion reference features; frequent voice switching reduces cache hit rate.
- Base64 voices are stored by content hash under `voices/_tmp/` to stabilize paths and improve cache hits.
- Examples prefer `.wav` over `.mp3` for faster decoding when both exist.

### AMP (`--s2mel_amp` / `--vocoder_amp`)

AMP can significantly speed up v2 on modern NVIDIA GPUs, but may be numerically unstable on some setups.
This fork includes automatic float32 fallback if AMP produces silent/invalid output.

If fallback happens you’ll see:
- `AMP-FALLBACK: ... retry vocoder float32`
- `AMP-FALLBACK: retry s2mel+vocoder float32`

### About `max_mel_tokens` warnings

If you see warnings like:

```
generation stopped due to exceeding `max_mel_tokens`
```

it means the model hit a truncation limit. You can:
- increase `x_advanced.max_mel_tokens`, or
- reduce `x_advanced.max_tokens_per_segment` to split into smaller segments.

## Running IndexTTS2 in Python

Use `uv run` so your script runs in the correct environment:

```bash
PYTHONPATH="$PYTHONPATH:." uv run indextts/infer_v2.py
```

#!/usr/bin/env python3
"""
IndexTTS2 OpenAI-compatible REST API Server

Usage:
    python api_server.py --port 8000 --model_dir checkpoints
    python api_server.py --port 8000 --model_dir checkpoints --fp16 --torch_compile --accel --api_key sk-xxx
"""
import os
import sys
import argparse
import time
import logging
import traceback
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Response, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.concurrency import run_in_threadpool

# Import from api folder (project root level)
from api.schemas import (
    SpeechRequest,
    VoiceListResponse,
    VoiceInfo,
    VoiceCreateRequest,
    VoiceCreateResponse,
    ModelListResponse,
    ModelInfo,
    ErrorResponse,
    ErrorDetail
)
from api.voice_manager import VoiceManager
from api.audio_converter import convert_audio, get_content_type, check_ffmpeg_available


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Global instances
tts_model = None
voice_manager = None
model_version = "v2"
api_key_required: Optional[str] = None


# Security
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    authorization: Optional[str] = Header(None)
):
    """Verify API key from Authorization header."""
    if not api_key_required:
        return None
    
    token = None
    if credentials:
        token = credentials.credentials
    elif authorization:
        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
    
    if not token:
        raise HTTPException(status_code=401, detail="Missing API key")
    if token != api_key_required:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return token


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="IndexTTS2 API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_dir", type=str, default="checkpoints")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--voices_dir", type=str, default="voices")
    parser.add_argument("--model_version", type=str, default="v2", choices=["v1", "v2"])
    parser.add_argument("--torch_compile", action="store_true", help="Enable torch.compile (v2: s2mel, v1: n/a)")
    parser.add_argument("--accel", action="store_true", help="Enable GPT acceleration engine (v2 only)")
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed (v2 only, requires extra)")
    parser.add_argument("--s2mel_amp", action="store_true", help="Enable AMP for s2mel (v2 only)")
    parser.add_argument("--vocoder_amp", action="store_true", help="Enable AMP for vocoder (v2 only)")
    parser.add_argument("--empty_cache_on_cache_miss", action="store_true", help="Call torch.cuda.empty_cache() on voice/emotion cache misses (v2 only)")
    parser.add_argument("--max_concurrent_inferences", type=int, default=1, help="Max in-flight inferences (default: 1)")
    parser.add_argument("--timing_headers", action="store_true", help="Add X-IndexTTS-* timing headers to responses")
    parser.add_argument("--timing_log", action="store_true", help="Log per-request timings to server logs")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def load_model(args):
    """Load the TTS model."""
    global tts_model, model_version
    
    model_version = args.model_version
    config_path = args.config or os.path.join(args.model_dir, "config.yaml")
    
    logger.info(f"Loading IndexTTS{model_version.upper()} model...")
    logger.info(f"Config: {config_path}, Model dir: {args.model_dir}")
    
    if model_version == "v2":
        from indextts.infer_v2 import IndexTTS2
        tts_model = IndexTTS2(
            cfg_path=config_path,
            model_dir=args.model_dir,
            use_fp16=args.fp16,
            device=args.device,
            use_deepspeed=args.deepspeed,
            use_accel=args.accel,
            use_torch_compile=args.torch_compile,
            empty_cache_on_cache_miss=args.empty_cache_on_cache_miss,
            use_s2mel_amp=args.s2mel_amp,
            use_vocoder_amp=args.vocoder_amp,
        )
    else:
        from indextts.infer import IndexTTS
        tts_model = IndexTTS(
            cfg_path=config_path,
            model_dir=args.model_dir,
            use_fp16=args.fp16,
            device=args.device
        )
    
    logger.info("Model loaded successfully!")
    return tts_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global tts_model, voice_manager, api_key_required
    
    args = app.state.args
    
    # Set API key
    api_key_required = args.api_key
    if api_key_required:
        logger.info("API key authentication enabled")
    else:
        logger.warning("No API key set - server is open to all requests")
    
    # Initialize voice manager
    voice_manager = VoiceManager(voices_dir=args.voices_dir, examples_dir="examples")
    
    # Load model
    tts_model = load_model(args)
    app.state.infer_semaphore = asyncio.Semaphore(max(1, int(args.max_concurrent_inferences)))
    
    yield
    
    # Cleanup
    if tts_model is not None:
        del tts_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Create FastAPI app
app = FastAPI(
    title="IndexTTS2 API",
    description="OpenAI-compatible Text-to-Speech API powered by IndexTTS2",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": tts_model is not None,
        "model_version": model_version,
        "ffmpeg_available": check_ffmpeg_available()
    }


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models."""
    return ModelListResponse(data=[
        ModelInfo(id="indextts2", created=int(time.time())),
        ModelInfo(id="indextts", created=int(time.time())),
    ])


@app.get("/v1/audio/voices", response_model=VoiceListResponse)
async def list_voices(_: str = Depends(verify_api_key)):
    """List all available voices."""
    voices = voice_manager.list_voices()
    return VoiceListResponse(voices=[VoiceInfo(**v) for v in voices])


@app.post("/v1/audio/voices", response_model=VoiceCreateResponse)
async def create_voice(request: VoiceCreateRequest, _: str = Depends(verify_api_key)):
    """Register a new voice."""
    try:
        metadata = voice_manager.register_voice(
            name=request.name,
            audio_base64=request.audio,
            description=request.description
        )
        return VoiceCreateResponse(id=metadata["id"], name=metadata["name"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/voices/{voice_id}/preview")
async def get_voice_preview(voice_id: str):
    """Get voice preview audio."""
    voice_path = voice_manager.get_voice_path(voice_id)
    if not voice_path:
        raise HTTPException(status_code=404, detail="Voice not found")
    return FileResponse(voice_path, media_type="audio/wav")


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest, _: str = Depends(verify_api_key)):
    """Create speech from text (OpenAI TTS API compatible)."""
    global tts_model, voice_manager, model_version
    
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    t0 = time.perf_counter()

    # Resolve voice path
    voice_path = None
    emo_audio_path = None
    
    if request.x_voice_audio:
        try:
            voice_path = voice_manager.save_voice_from_base64(request.x_voice_audio)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        voice_path = voice_manager.get_voice_path(request.voice)
        if not voice_path:
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{request.voice}' not found. Use GET /v1/audio/voices to list available voices."
            )
    
    logger.info(f"Speech request: voice={request.voice}, voice_path={voice_path}, text_len={len(request.input)}")
    t_voice_resolve = time.perf_counter()
    
    # Prepare parameters
    generation_kwargs = {}
    interval_silence = 200
    
    if request.x_advanced:
        generation_kwargs["max_text_tokens_per_segment"] = request.x_advanced.max_tokens_per_segment
        generation_kwargs["temperature"] = request.x_advanced.temperature
        generation_kwargs["top_p"] = request.x_advanced.top_p
        generation_kwargs["top_k"] = request.x_advanced.top_k
        generation_kwargs["repetition_penalty"] = request.x_advanced.repetition_penalty
        interval_silence = request.x_advanced.interval_silence
    
    # Emotion settings (v2 only)
    emo_params = {}
    if request.x_emotion and model_version == "v2":
        emo_config = request.x_emotion
        
        if emo_config.type == "vector" and emo_config.vector:
            emo_params["emo_vector"] = emo_config.vector
        elif emo_config.type == "reference" and emo_config.reference_audio:
            try:
                emo_audio_path = voice_manager.save_voice_from_base64(emo_config.reference_audio)
                emo_params["emo_audio_prompt"] = emo_audio_path
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid emotion reference audio: {e}")
        elif emo_config.type == "text" and emo_config.text:
            emo_params["use_emo_text"] = True
            emo_params["emo_text"] = emo_config.text
        
        emo_params["emo_alpha"] = emo_config.alpha
        emo_params["use_random"] = emo_config.random
    
    # Create temp output file
    try:
        logger.info("Starting inference (in-memory)")

        async with app.state.infer_semaphore:
            t_infer_start = time.perf_counter()

            def _do_infer():
                if model_version == "v2":
                    return tts_model.infer(
                        spk_audio_prompt=voice_path,
                        text=request.input,
                        output_path=None,
                        interval_silence=interval_silence,
                        verbose=False,
                        log_timings=False,
                        return_tensor=True,
                        **emo_params,
                        **generation_kwargs,
                    )
                return tts_model.infer(
                    audio_prompt=voice_path,
                    text=request.input,
                    output_path=None,
                    verbose=False,
                    log_timings=False,
                    return_tensor=True,
                    **generation_kwargs,
                )

            infer_result = await run_in_threadpool(_do_infer)
            t_infer_end = time.perf_counter()

        if not infer_result or not isinstance(infer_result, (tuple, list)) or len(infer_result) != 2:
            raise RuntimeError("Inference returned an invalid result")
        sampling_rate, audio_tensor = infer_result
        if audio_tensor is None:
            raise RuntimeError("Inference returned empty audio")
        logger.info("Inference completed")
        
        # Convert format if needed
        t_convert_start = time.perf_counter()
        try:
            audio_data = convert_audio(audio_tensor, int(sampling_rate), request.response_format)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        t_convert_end = time.perf_counter()
        
        logger.info(f"Returning audio: {len(audio_data)} bytes, format={request.response_format}")

        voice_resolve_ms = (t_voice_resolve - t0) * 1000
        infer_ms = (t_infer_end - t_infer_start) * 1000
        convert_ms = (t_convert_end - t_convert_start) * 1000
        total_ms = (t_convert_end - t0) * 1000

        if app.state.args.timing_log:
            logger.info(
                "Timings(ms): voice_resolve=%.2f infer=%.2f convert=%.2f total=%.2f",
                voice_resolve_ms,
                infer_ms,
                convert_ms,
                total_ms,
            )

        headers = {"Content-Disposition": f'attachment; filename="speech.{request.response_format}"'}
        if app.state.args.timing_headers:
            headers.update({
                "X-IndexTTS-Time-VoiceResolve-MS": f"{voice_resolve_ms:.2f}",
                "X-IndexTTS-Time-Infer-MS": f"{infer_ms:.2f}",
                "X-IndexTTS-Time-Convert-MS": f"{convert_ms:.2f}",
                "X-IndexTTS-Time-Total-MS": f"{total_ms:.2f}",
            })
        
        return Response(
            content=audio_data,
            media_type=get_content_type(request.response_format),
            headers=headers,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"Speech synthesis failed: {e}\n{error_detail}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")
    
    finally:
        # Cleanup per-request temp files (cached base64 voices are managed by VoiceManager)
        pass


# ============================================================================
# Error Handler
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions in OpenAI-compatible format."""
    return Response(
        content=ErrorResponse(
            error=ErrorDetail(
                message=exc.detail,
                type="invalid_request_error" if exc.status_code == 400 else "server_error",
                code=str(exc.status_code)
            )
        ).model_dump_json(),
        status_code=exc.status_code,
        media_type="application/json"
    )


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point."""
    args = parse_args()
    app.state.args = args
    
    print(f">> Starting IndexTTS2 API Server")
    print(f">> Host: {args.host}, Port: {args.port}")
    print(f">> Model version: {args.model_version}")
    print(f">> API docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api_server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()

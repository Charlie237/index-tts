"""
Audio format converter for IndexTTS API.
"""
import io
import subprocess
from typing import Union
import torch
import torchaudio


def convert_audio(audio_data: Union[bytes, torch.Tensor], sample_rate: int, output_format: str) -> bytes:
    """Convert audio to the specified format."""
    if output_format == "wav":
        if isinstance(audio_data, bytes):
            return audio_data
        else:
            return tensor_to_wav(audio_data, sample_rate)
    
    if output_format == "pcm":
        return tensor_to_pcm(audio_data, sample_rate)
    
    return convert_with_ffmpeg(audio_data, sample_rate, output_format)


def tensor_to_wav(tensor: torch.Tensor, sample_rate: int) -> bytes:
    """Convert a torch tensor to WAV bytes."""
    buffer = io.BytesIO()
    
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        tensor = torch.clamp(tensor * 32767, -32768, 32767).to(torch.int16)
    
    torchaudio.save(buffer, tensor, sample_rate, format="wav")
    buffer.seek(0)
    return buffer.read()


def tensor_to_pcm(tensor: torch.Tensor, sample_rate: int) -> bytes:
    """Convert a torch tensor to raw PCM bytes."""
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        tensor = torch.clamp(tensor * 32767, -32768, 32767).to(torch.int16)
    
    return tensor.numpy().tobytes()


def convert_with_ffmpeg(audio_data: Union[bytes, torch.Tensor], sample_rate: int, output_format: str) -> bytes:
    """Convert audio using ffmpeg."""
    if isinstance(audio_data, torch.Tensor):
        wav_data = tensor_to_wav(audio_data, sample_rate)
    else:
        wav_data = audio_data
    
    quality_settings = {
        "mp3": ["-b:a", "192k"],
        "opus": ["-b:a", "128k"],
        "flac": ["-compression_level", "8"]
    }
    
    cmd = ["ffmpeg", "-y", "-f", "wav", "-i", "pipe:0"]
    
    if output_format in quality_settings:
        cmd.extend(quality_settings[output_format])
    
    cmd.extend(["-f", output_format, "pipe:1"])
    
    try:
        process = subprocess.run(cmd, input=wav_data, capture_output=True, check=True)
        return process.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode()}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Install with: brew install ffmpeg")


def get_content_type(output_format: str) -> str:
    """Get the MIME content type for an audio format."""
    content_types = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "flac": "audio/flac",
        "pcm": "audio/pcm"
    }
    return content_types.get(output_format, "audio/octet-stream")


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

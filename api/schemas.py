"""
Pydantic schemas for IndexTTS OpenAI-compatible API.
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict


class EmotionConfig(BaseModel):
    """Emotion control configuration for IndexTTS2."""
    model_config = ConfigDict(extra="ignore")
    
    type: Literal["vector", "reference", "text"] = Field(
        default="vector",
        description="Emotion control type"
    )
    vector: Optional[List[float]] = Field(
        default=None,
        description="8-dimensional emotion vector: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]"
    )
    reference_audio: Optional[str] = Field(
        default=None,
        description="Base64 encoded emotion reference audio"
    )
    text: Optional[str] = Field(
        default=None,
        description="Emotion text prompt"
    )
    alpha: float = Field(default=0.65, ge=0.0, le=1.0)
    random: bool = Field(default=False)
    
    @field_validator("vector")
    @classmethod
    def validate_vector(cls, v):
        if v is not None:
            if len(v) != 8:
                raise ValueError("Emotion vector must have exactly 8 dimensions")
            for val in v:
                if not 0.0 <= val <= 1.0:
                    raise ValueError("Each emotion value must be between 0.0 and 1.0")
        return v


class AdvancedConfig(BaseModel):
    """Advanced generation parameters."""
    model_config = ConfigDict(extra="ignore")
    
    max_tokens_per_segment: int = Field(default=120, ge=50, le=500)
    max_mel_tokens: int = Field(default=1500, ge=50, le=20000)
    do_sample: bool = Field(default=True)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=30, ge=1, le=100)
    num_beams: int = Field(default=3, ge=1, le=10)
    length_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=10.0, ge=1.0, le=20.0)
    interval_silence: int = Field(default=200, ge=0, le=2000)
    diffusion_steps: int = Field(default=25, ge=1, le=50)
    inference_cfg_rate: float = Field(default=0.7, ge=0.0, le=2.0)


class SpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request."""
    model_config = ConfigDict(extra="ignore")
    
    model: str = Field(default="indextts2")
    input: str = Field(..., min_length=1, max_length=10000)
    voice: str = Field(...)
    response_format: Literal["wav", "mp3", "opus", "flac", "pcm"] = Field(default="wav")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    
    # IndexTTS2 extensions
    x_voice_audio: Optional[str] = Field(default=None)
    x_emotion: Optional[EmotionConfig] = Field(default=None)
    x_advanced: Optional[AdvancedConfig] = Field(default=None)


class VoiceInfo(BaseModel):
    """Voice information."""
    id: str
    name: str
    description: Optional[str] = None
    preview_url: Optional[str] = None
    created_at: Optional[str] = None


class VoiceListResponse(BaseModel):
    """Response for GET /v1/audio/voices."""
    voices: List[VoiceInfo]


class VoiceCreateRequest(BaseModel):
    """Request for POST /v1/audio/voices."""
    name: str = Field(..., min_length=1, max_length=100)
    audio: str = Field(...)
    description: Optional[str] = Field(default=None, max_length=500)


class VoiceCreateResponse(BaseModel):
    """Response for POST /v1/audio/voices."""
    id: str
    name: str
    message: str = "Voice created successfully"


class ModelInfo(BaseModel):
    """Model information (OpenAI compatible)."""
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "indextts"


class ModelListResponse(BaseModel):
    """Response for GET /v1/models."""
    object: str = "list"
    data: List[ModelInfo]


class ErrorDetail(BaseModel):
    """Error detail."""
    message: str
    type: str = "invalid_request_error"
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response (OpenAI compatible)."""
    error: ErrorDetail

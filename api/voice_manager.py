"""
Voice manager for IndexTTS API.
"""
import os
import base64
import hashlib
import json
import tempfile
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path


class VoiceManager:
    """Manages voice references for TTS synthesis."""
    
    def __init__(self, voices_dir: str = "voices", examples_dir: str = "examples"):
        self.voices_dir = Path(voices_dir)
        self.examples_dir = Path(examples_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        
        self._registry: Dict[str, dict] = {}
        self._registry_file = self.voices_dir / "registry.json"
        
        self._load_registry()
        self._register_examples()
    
    def _load_registry(self):
        """Load voice registry from file."""
        if self._registry_file.exists():
            try:
                with open(self._registry_file, "r", encoding="utf-8") as f:
                    self._registry = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._registry = {}
    
    def _save_registry(self):
        """Save voice registry to file."""
        with open(self._registry_file, "w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2, ensure_ascii=False)
    
    def _register_examples(self):
        """Register example voices from examples directory."""
        if not self.examples_dir.exists():
            return
        
        # Support both wav and mp3 files
        for ext in ["*.wav", "*.mp3"]:
            for audio_file in self.examples_dir.glob(ext):
                voice_id = audio_file.stem
                if voice_id not in self._registry:
                    self._registry[voice_id] = {
                        "id": voice_id,
                        "name": voice_id.replace("_", " ").title(),
                        "path": str(audio_file.absolute()),  # Use absolute path!
                        "description": f"Example voice: {voice_id}",
                        "preview_url": f"/voices/{voice_id}/preview",
                        "created_at": datetime.now().isoformat(),
                        "is_example": True
                    }
        
        self._save_registry()
    
    def get_voice_path(self, voice_id: str) -> Optional[str]:
        """Get the audio file path for a voice ID."""
        # Check if it's already an absolute path that exists
        if os.path.isabs(voice_id) and os.path.isfile(voice_id):
            return voice_id
        
        # Check if it's a relative path that exists
        if os.path.isfile(voice_id):
            return os.path.abspath(voice_id)
        
        # Check registry
        if voice_id in self._registry:
            path = self._registry[voice_id].get("path")
            if path and os.path.isfile(path):
                return path
        
        # Check examples directory with various extensions
        for ext in [".wav", ".mp3", ""]:
            example_path = self.examples_dir / f"{voice_id}{ext}"
            if example_path.exists():
                return str(example_path.absolute())
        
        return None
    
    def save_voice_from_base64(self, audio_base64: str) -> str:
        """Save a base64-encoded audio to a temporary file."""
        try:
            audio_data = base64.b64decode(audio_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 audio data: {e}")
        
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(audio_data)
        except Exception:
            os.close(fd)
            raise
        
        return temp_path
    
    def register_voice(self, name: str, audio_base64: str, description: Optional[str] = None) -> dict:
        """Register a new voice from base64 audio."""
        audio_data = base64.b64decode(audio_base64)
        voice_hash = hashlib.md5(audio_data).hexdigest()[:12]
        voice_id = f"voice_{voice_hash}"
        
        audio_path = self.voices_dir / f"{voice_id}.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_data)
        
        metadata = {
            "id": voice_id,
            "name": name,
            "path": str(audio_path.absolute()),
            "description": description,
            "preview_url": f"/voices/{voice_id}/preview",
            "created_at": datetime.now().isoformat(),
            "is_example": False
        }
        
        self._registry[voice_id] = metadata
        self._save_registry()
        
        return metadata
    
    def list_voices(self) -> List[dict]:
        """List all registered voices."""
        return [
            {
                "id": v["id"],
                "name": v["name"],
                "description": v.get("description"),
                "preview_url": v.get("preview_url"),
                "created_at": v.get("created_at")
            }
            for v in self._registry.values()
        ]
    
    def get_voice(self, voice_id: str) -> Optional[dict]:
        """Get voice metadata by ID."""
        return self._registry.get(voice_id)

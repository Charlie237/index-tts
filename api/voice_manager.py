"""
Voice manager for IndexTTS API.
"""
import os
import base64
import hashlib
import json
import time
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path


class VoiceManager:
    """Manages voice references for TTS synthesis."""
    
    def __init__(self, voices_dir: str = "voices", examples_dir: str = "examples"):
        self.voices_dir = Path(voices_dir)
        self.examples_dir = Path(examples_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_voices_dir = self.voices_dir / "_tmp"
        self._tmp_voices_dir.mkdir(parents=True, exist_ok=True)
        
        self._registry: Dict[str, dict] = {}
        self._registry_file = self.voices_dir / "registry.json"
        
        self._load_registry()
        self._register_examples()
        self._cleanup_tmp_voices()
    
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

        # Prefer WAV over MP3 for the same stem (faster decode, fewer dependencies).
        by_stem: Dict[str, Path] = {}
        for audio_file in self.examples_dir.glob("*.mp3"):
            by_stem[audio_file.stem] = audio_file
        for audio_file in self.examples_dir.glob("*.wav"):
            by_stem[audio_file.stem] = audio_file

        changed = False
        for voice_id, audio_file in by_stem.items():
            new_path = str(audio_file.absolute())
            existing = self._registry.get(voice_id)

            if not existing:
                self._registry[voice_id] = {
                    "id": voice_id,
                    "name": voice_id.replace("_", " ").title(),
                    "path": new_path,
                    "description": f"Example voice: {voice_id}",
                    "preview_url": f"/voices/{voice_id}/preview",
                    "created_at": datetime.now().isoformat(),
                    "is_example": True,
                }
                changed = True
                continue

            # Upgrade existing example entries to prefer WAV if available.
            if existing.get("is_example") is True:
                old_path = existing.get("path")
                old_ext = (Path(old_path).suffix.lower() if old_path else "")
                new_ext = audio_file.suffix.lower()
                should_upgrade = (old_ext == ".mp3" and new_ext == ".wav") or (not old_path) or (not os.path.isfile(old_path))
                if should_upgrade and old_path != new_path:
                    existing["path"] = new_path
                    changed = True

        if changed:
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
        """Save a base64-encoded audio to a stable cached file.

        Returning a stable path allows the TTS model's internal caches to work
        across repeated requests that use the same reference audio.
        """
        try:
            audio_data = base64.b64decode(audio_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 audio data: {e}")

        voice_hash = hashlib.md5(audio_data).hexdigest()[:12]
        cached_path = self._tmp_voices_dir / f"b64_{voice_hash}.wav"
        if not cached_path.exists():
            cached_path.write_bytes(audio_data)
            # Best-effort cleanup to avoid unbounded growth.
            self._cleanup_tmp_voices()
        return str(cached_path.absolute())

    def _cleanup_tmp_voices(self, max_age_seconds: int = 7 * 24 * 60 * 60, max_files: int = 512) -> None:
        """Remove cached temp voices older than max_age_seconds and enforce max_files."""
        now = time.time()
        try:
            files = list(self._tmp_voices_dir.glob("b64_*.wav"))
            for p in files:
                try:
                    if now - p.stat().st_mtime > max_age_seconds:
                        p.unlink(missing_ok=True)
                except OSError:
                    continue
            if max_files > 0:
                files = [p for p in files if p.exists()]
                if len(files) > max_files:
                    files.sort(key=lambda p: p.stat().st_mtime)
                    for p in files[: max(0, len(files) - max_files)]:
                        try:
                            p.unlink(missing_ok=True)
                        except OSError:
                            continue
        except OSError:
            # best-effort cleanup
            return
    
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

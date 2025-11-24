"""
Voice Manager module for handling voice samples.
Manages storage and retrieval of voice samples for TTS voice cloning.
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Optional
from pydub import AudioSegment


class VoiceManager:
    """Manages voice samples for TTS voice cloning."""
    
    def __init__(self, voices_dir: str = "voices"):
        """
        Initialize Voice Manager.
        
        Args:
            voices_dir: Directory to store voice samples
        """
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.voices_dir / "voices.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Load voice metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_metadata(self):
        """Save voice metadata to JSON file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save metadata: {str(e)}")
    
    def add_voice(self, name: str, audio_file) -> bool:
        """
        Add a new voice sample.
        
        Args:
            name: Name for the voice
            audio_file: Audio file object (from Streamlit file uploader)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate name
            if not name or name in self.metadata:
                return False
            
            # Create directory for this voice
            voice_dir = self.voices_dir / name.replace(" ", "_")
            voice_dir.mkdir(exist_ok=True)
            
            # Determine file extension
            original_name = audio_file.name.lower()
            if original_name.endswith('.wav'):
                ext = 'wav'
            elif original_name.endswith('.mp3'):
                ext = 'mp3'
            else:
                return False
            
            # Save the audio file
            audio_path = voice_dir / f"sample.{ext}"
            
            # Write the uploaded file
            with open(audio_path, 'wb') as f:
                audio_file.seek(0)
                f.write(audio_file.read())
            
            # Convert to WAV if needed (TTS usually works best with WAV)
            wav_path = voice_dir / "sample.wav"
            if ext != 'wav':
                try:
                    audio = AudioSegment.from_file(audio_path, format=ext)
                    audio.export(wav_path, format="wav")
                    # Remove original if conversion successful
                    os.remove(audio_path)
                except Exception as e:
                    # If conversion fails, keep original
                    print(f"Warning: Could not convert to WAV: {e}")
                    if not wav_path.exists():
                        shutil.copy(audio_path, wav_path)
            
            # Update metadata
            self.metadata[name] = {
                "path": str(wav_path),
                "original_name": audio_file.name,
                "dir": str(voice_dir)
            }
            
            self._save_metadata()
            return True
            
        except Exception as e:
            print(f"Error adding voice: {e}")
            return False
    
    def remove_voice(self, name: str) -> bool:
        """
        Remove a voice sample.
        
        Args:
            name: Name of the voice to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if name not in self.metadata:
                return False
            
            # Remove voice directory
            voice_dir = Path(self.metadata[name]["dir"])
            if voice_dir.exists():
                shutil.rmtree(voice_dir)
            
            # Remove from metadata
            del self.metadata[name]
            self._save_metadata()
            
            return True
            
        except Exception as e:
            print(f"Error removing voice: {e}")
            return False
    
    def get_voice_list(self) -> List[str]:
        """
        Get list of available voice names.
        
        Returns:
            List of voice names including "Model default" option
        """
        voices = sorted(list(self.metadata.keys()))
        # Add "Model default" at the beginning of the list
        return ["Model default"] + voices
    
    def get_voice_path(self, name: str) -> Optional[str]:
        """
        Get path to voice sample file.
        
        Args:
            name: Name of the voice
            
        Returns:
            Path to voice sample, or None if not found or if "Model default" is selected
        """
        # Return None for "Model default" to indicate no voice sample should be used
        if name == "Model default":
            return None
            
        if name in self.metadata:
            path = self.metadata[name]["path"]
            if os.path.exists(path):
                return path
        return None
    
    def voice_exists(self, name: str) -> bool:
        """
        Check if a voice exists.
        
        Args:
            name: Name of the voice
            
        Returns:
            True if voice exists, False otherwise
        """
        return name in self.metadata

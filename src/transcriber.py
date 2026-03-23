"""
Audio transcription module using OpenAI Whisper.
"""

import os
from pathlib import Path
from typing import Optional
import openai
from dotenv import load_dotenv

load_dotenv()


class Transcriber:
    """Transcribe audio files using OpenAI Whisper."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize transcriber with OpenAI API key.
        
        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
        """
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
    
    def transcribe(
        self, 
        audio_path: str, 
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> dict:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (mp3, mp4, mpeg, mpga, m4a, wav, webm)
            model: Whisper model to use
            language: Optional language code (e.g., 'zh', 'en')
            prompt: Optional prompt to guide transcription
            
        Returns:
            dict: Transcription result with text, segments, etc.
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        with open(audio_path, "rb") as audio:
            params = {
                "model": model,
                "file": audio,
            }
            if language:
                params["language"] = language
            if prompt:
                params["prompt"] = prompt
                
            response = self.client.audio.transcriptions.create(**params)
        
        return {
            "text": response.text,
            "model": model,
            "language": language,
        }
    
    def transcribe_with_timestamps(
        self,
        audio_path: str,
        model: str = "whisper-1",
        language: Optional[str] = None
    ) -> dict:
        """
        Transcribe with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            model: Whisper model to use
            language: Optional language code
            
        Returns:
            dict: Transcription with timestamps
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        with open(audio_path, "rb") as audio:
            params = {
                "model": model,
                "file": audio,
                "response_format": "verbose_json",
                "timestamp_granularities": ["word"]
            }
            if language:
                params["language"] = language
                
            response = self.client.audio.transcriptions.create(**params)
        
        return {
            "text": response.text,
            "segments": response.segments if hasattr(response, 'segments') else [],
            "words": response.words if hasattr(response, 'words') else [],
            "model": model,
            "language": language,
        }

"""Whisper transcription module."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".wav",
    ".webm",
}


@dataclass(frozen=True)
class TranscriptionResult:
    """Transcription result returned by Whisper."""

    text: str
    language: Optional[str]
    model: str


class Transcriber:
    """Wraps OpenAI Whisper transcription API calls."""

    def __init__(self, client: Optional[OpenAI] = None, api_key: Optional[str] = None) -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = client or OpenAI(api_key=resolved_key)

    def transcribe(
        self,
        audio_file: Union[str, Path],
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file into text."""
        path = Path(audio_file)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise ValueError(f"Unsupported audio extension '{path.suffix}'. Allowed: {allowed}")

        with path.open("rb") as audio_handle:
            payload = {
                "model": model,
                "file": audio_handle,
            }
            if language:
                payload["language"] = language
            if prompt:
                payload["prompt"] = prompt

            response = self.client.audio.transcriptions.create(**payload)

        text = getattr(response, "text", "")
        if not text:
            raise ValueError("Whisper returned an empty transcription.")

        return TranscriptionResult(text=text.strip(), language=language, model=model)

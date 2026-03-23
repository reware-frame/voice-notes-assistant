from types import SimpleNamespace

import pytest

from src.transcriber import Transcriber


class FakeTranscriptions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text="this is a transcript")


class FakeClient:
    def __init__(self) -> None:
        self.audio = SimpleNamespace(transcriptions=FakeTranscriptions())


def test_transcribe_success(tmp_path):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"dummy audio")

    fake_client = FakeClient()
    transcriber = Transcriber(client=fake_client)
    result = transcriber.transcribe(audio_file, language="zh", prompt="meeting notes")

    assert result.text == "this is a transcript"
    assert result.language == "zh"
    assert result.model == "whisper-1"

    api_call = fake_client.audio.transcriptions.calls[0]
    assert api_call["model"] == "whisper-1"
    assert api_call["language"] == "zh"
    assert api_call["prompt"] == "meeting notes"


def test_transcribe_missing_file_raises(tmp_path):
    missing_file = tmp_path / "missing.wav"
    transcriber = Transcriber(client=FakeClient())

    with pytest.raises(FileNotFoundError):
        transcriber.transcribe(missing_file)


def test_transcribe_invalid_extension_raises(tmp_path):
    text_file = tmp_path / "note.txt"
    text_file.write_text("not audio", encoding="utf-8")

    transcriber = Transcriber(client=FakeClient())

    with pytest.raises(ValueError):
        transcriber.transcribe(text_file)

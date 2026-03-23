from pathlib import Path

from src import cli
from src.processor import StructuredNote
from src.transcriber import TranscriptionResult


class FakeTranscriber:
    def transcribe(self, **kwargs):
        return TranscriptionResult(
            text="Call vendor and finalize budget",
            language=kwargs.get("language"),
            model=kwargs.get("model", "whisper-1"),
        )


class FakeProcessor:
    def __init__(self, model: str):
        self.model = model

    def process(self, transcript: str, hint=None):
        return StructuredNote(
            title="Budget follow-up",
            category="todo",
            tags=["finance"],
            priority="high",
            summary=f"Structured: {transcript}",
            action_items=["Send quote", "Confirm approval"],
        )


class FakeFormatter:
    def export(self, note, output_path, source_text=None, generated_at=None):
        target = Path(output_path)
        target.write_text("# fake markdown\n", encoding="utf-8")
        return target


def test_process_text_command_writes_output(tmp_path, monkeypatch):
    output_file = tmp_path / "text-note.md"

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "Processor", FakeProcessor)
    monkeypatch.setattr(cli, "MarkdownFormatter", FakeFormatter)

    exit_code = cli.main([
        "process-text",
        "Need to follow up with vendor",
        "--output",
        str(output_file),
    ])

    assert exit_code == 0
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == "# fake markdown\n"


def test_transcribe_command_writes_output(tmp_path, monkeypatch):
    audio_file = tmp_path / "audio.wav"
    output_file = tmp_path / "voice-note.md"
    audio_file.write_bytes(b"fake wav data")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(cli, "Processor", FakeProcessor)
    monkeypatch.setattr(cli, "MarkdownFormatter", FakeFormatter)

    exit_code = cli.main([
        "transcribe",
        str(audio_file),
        "--output",
        str(output_file),
    ])

    assert exit_code == 0
    assert output_file.exists()


def test_main_returns_error_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = cli.main(["process-text", "hello"])

    assert exit_code == 1


def test_batch_command_processes_multiple_files(tmp_path, monkeypatch):
    """Test batch processing of multiple audio files."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create fake audio files
    (input_dir / "meeting1.wav").write_bytes(b"fake wav 1")
    (input_dir / "meeting2.wav").write_bytes(b"fake wav 2")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(cli, "Processor", FakeProcessor)
    monkeypatch.setattr(cli, "MarkdownFormatter", FakeFormatter)

    exit_code = cli.main([
        "batch",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == 0
    assert (output_dir / "meeting1.md").exists()
    assert (output_dir / "meeting2.md").exists()


def test_batch_command_generates_report(tmp_path, monkeypatch):
    """Test batch command with report generation."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    report_file = tmp_path / "report.json"
    input_dir.mkdir()
    output_dir.mkdir()

    (input_dir / "note.m4a").write_bytes(b"fake audio")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(cli, "Processor", FakeProcessor)
    monkeypatch.setattr(cli, "MarkdownFormatter", FakeFormatter)

    exit_code = cli.main([
        "batch",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--report",
        str(report_file),
    ])

    assert exit_code == 0
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "success" in content
    assert "total_files" in content


def test_batch_command_empty_directory(tmp_path, monkeypatch):
    """Test batch command with empty input directory."""
    input_dir = tmp_path / "empty_input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    exit_code = cli.main([
        "batch",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    ])

    assert exit_code == 0


def test_batch_command_missing_directory(monkeypatch):
    """Test batch command with non-existent input directory."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    exit_code = cli.main([
        "batch",
        "--input-dir",
        "/non/existent/path",
        "--output-dir",
        "/output/path",
    ])

    assert exit_code == 1

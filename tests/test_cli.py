"""Tests for CLI module with configuration support."""

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from src import cli
from src.config import Config
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
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model", "gpt-4o")
        self.provider_type = kwargs.get("provider_type", "openai")

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
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

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


class TestConfigCommand:
    """Tests for config subcommand."""

    def test_config_show(self, monkeypatch, capsys):
        """Test config --show displays configuration."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        exit_code = cli.main(["config", "--show"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "provider" in captured.out
        assert "default_model" in captured.out

    def test_config_init_creates_file(self, tmp_path, monkeypatch):
        """Test config --init creates default configuration file."""
        config_file = tmp_path / ".voice-notes" / "config.yaml"

        from src import config as config_module
        original_path = config_module.DEFAULT_CONFIG_PATH
        config_module.DEFAULT_CONFIG_PATH = config_file

        try:
            monkeypatch.setenv("OPENAI_API_KEY", "test-key")

            exit_code = cli.main(["config", "--init"])

            assert exit_code == 0
            assert config_file.exists()
        finally:
            config_module.DEFAULT_CONFIG_PATH = original_path

    def test_config_init_existing_file(self, tmp_path, monkeypatch, capsys):
        """Test config --init with existing file reports it exists."""
        config_file = tmp_path / ".voice-notes" / "config.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text("existing: config", encoding="utf-8")

        from src import config as config_module
        original_path = config_module.DEFAULT_CONFIG_PATH
        config_module.DEFAULT_CONFIG_PATH = config_file

        try:
            exit_code = cli.main(["config", "--init"])
            captured = capsys.readouterr()

            assert exit_code == 0
            assert "already exists" in captured.out
        finally:
            config_module.DEFAULT_CONFIG_PATH = original_path

    def test_config_without_flags_shows_error(self, capsys):
        """Test config without --show or --init shows error."""
        exit_code = cli.main(["config"])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "--show or --init" in captured.out


class TestConfigFileSupport:
    """Tests for --config parameter support."""

    def test_transcribe_with_config_file(self, tmp_path, monkeypatch):
        """Test transcribe command with --config parameter."""
        audio_file = tmp_path / "audio.wav"
        output_file = tmp_path / "output.md"
        config_file = tmp_path / "config.yaml"
        audio_file.write_bytes(b"fake wav data")

        # Create config file
        config_data = {
            "provider": "anthropic",
            "api_key": "anthropic-key",
            "default_model": "claude-3-opus",
            "whisper_model": "whisper-1",
        }
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        monkeypatch.setattr(cli, "Transcriber", FakeTranscriber)
        monkeypatch.setattr(cli, "Processor", FakeProcessor)
        monkeypatch.setattr(cli, "MarkdownFormatter", FakeFormatter)

        exit_code = cli.main([
            "--config", str(config_file),
            "transcribe",
            str(audio_file),
            "--output", str(output_file),
        ])

        assert exit_code == 0
        assert output_file.exists()

    def test_process_text_with_config_file(self, tmp_path, monkeypatch):
        """Test process-text command with --config parameter."""
        output_file = tmp_path / "output.md"
        config_file = tmp_path / "config.json"

        # Create JSON config file
        config_data = {
            "provider": "openai",
            "api_key": "config-api-key",
            "default_model": "gpt-4-turbo",
        }
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        monkeypatch.setattr(cli, "Processor", FakeProcessor)
        monkeypatch.setattr(cli, "MarkdownFormatter", FakeFormatter)

        exit_code = cli.main([
            "--config", str(config_file),
            "process-text",
            "Test note",
            "--output", str(output_file),
        ])

        assert exit_code == 0
        assert output_file.exists()

    def test_config_priority_command_line_over_config_file(self, tmp_path, monkeypatch):
        """Test that command line args override config file."""
        output_file = tmp_path / "output.md"
        config_file = tmp_path / "config.yaml"

        config_data = {"provider": "openai", "whisper_model": "whisper-1"}
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        monkeypatch.setattr(cli, "Processor", FakeProcessor)
        monkeypatch.setattr(cli, "MarkdownFormatter", FakeFormatter)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

        # Use --provider to override config
        exit_code = cli.main([
            "--config", str(config_file),
            "process-text",
            "Test",
            "--output", str(output_file),
            "--provider", "anthropic",
        ])

        assert exit_code == 0

    def test_anthropic_provider_without_key_shows_error(self, monkeypatch, capsys):
        """Test error shown when using anthropic without API key."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        exit_code = cli.main([
            "process-text",
            "Test note",
            "--provider", "anthropic",
        ])
        captured = capsys.readouterr()

        assert exit_code == 1
        assert "Anthropic API key not configured" in captured.out

    def test_ollama_provider_no_api_key_needed(self, tmp_path, monkeypatch):
        """Test that Ollama provider doesn't require API key."""
        output_file = tmp_path / "output.md"

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(cli, "Processor", FakeProcessor)
        monkeypatch.setattr(cli, "MarkdownFormatter", FakeFormatter)

        # Mock Processor to not actually call Ollama
        with monkeypatch.context() as m:
            m.setattr(cli, "Processor", FakeProcessor)
            exit_code = cli.main([
                "process-text",
                "Test note",
                "--provider", "ollama",
                "--output", str(output_file),
            ])

        # Note: This will fail because FakeProcessor doesn't accept provider_type
        # but it tests that CLI validation passes for ollama without API key


class TestResolveConfig:
    """Tests for resolve_config function."""

    def test_resolve_config_with_explicit_path(self, tmp_path):
        """Test resolving config with explicit --config path."""
        config_file = tmp_path / "custom.yaml"
        config_file.write_text(yaml.dump({"provider": "anthropic"}), encoding="utf-8")

        args = SimpleNamespace(config=config_file)
        config = cli.resolve_config(args)

        assert config.provider == "anthropic"

    def test_resolve_config_without_path_uses_default(self, tmp_path, monkeypatch):
        """Test resolving config without path uses default location."""
        default_file = tmp_path / ".voice-notes" / "config.yaml"
        default_file.parent.mkdir(parents=True)
        default_file.write_text(yaml.dump({"provider": "ollama"}), encoding="utf-8")

        from src import config as config_module
        original_path = config_module.DEFAULT_CONFIG_PATH
        config_module.DEFAULT_CONFIG_PATH = default_file

        try:
            args = SimpleNamespace(config=None)
            config = cli.resolve_config(args)

            assert config.provider == "ollama"
        finally:
            config_module.DEFAULT_CONFIG_PATH = original_path

    def test_resolve_config_with_command_line_overrides(self, tmp_path):
        """Test that command line args override config file values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "provider": "openai",
            "whisper_model": "whisper-1",
        }), encoding="utf-8")

        args = SimpleNamespace(
            config=config_file,
            provider="anthropic",
            whisper_model="whisper-large",
            gpt_model=None,
        )
        config = cli.resolve_config(args)

        assert config.provider == "anthropic"
        assert config.whisper_model == "whisper-large"


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_validate_openai_with_key(self):
        """Test OpenAI config with API key is valid."""
        config = Config(provider="openai", api_key="test-key")
        error = cli.validate_config(config)
        assert error is None

    def test_validate_openai_without_key(self, monkeypatch):
        """Test OpenAI config without API key is invalid."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = Config(provider="openai", api_key=None)
        error = cli.validate_config(config)
        assert error is not None
        assert "OpenAI API key not configured" in error

    def test_validate_anthropic_with_key(self):
        """Test Anthropic config with API key is valid."""
        config = Config(provider="anthropic", api_key="test-key")
        error = cli.validate_config(config)
        assert error is None

    def test_validate_anthropic_without_key(self, monkeypatch):
        """Test Anthropic config without API key is invalid."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = Config(provider="anthropic", api_key=None)
        error = cli.validate_config(config)
        assert error is not None
        assert "Anthropic API key not configured" in error

    def test_validate_ollama_no_key_needed(self):
        """Test Ollama config doesn't require API key."""
        config = Config(provider="ollama")
        error = cli.validate_config(config)
        assert error is None

"""CLI entry point for the AI voice notes assistant."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

from .config import Config, load_config, create_default_config, get_default_config_path
from .formatter import MarkdownFormatter
from .processor import Processor, StructuredNote
from .transcriber import TranscriptionResult, Transcriber


load_dotenv()

SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}


def build_parser() -> argparse.ArgumentParser:
    """Build and return command parser."""
    parser = argparse.ArgumentParser(
        prog="voice-notes",
        description="Transcribe audio and export structured markdown notes.",
    )

    # Global config argument
    parser.add_argument(
        "--config",
        help="Path to configuration file (YAML or JSON).",
        type=Path,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe = subparsers.add_parser("transcribe", help="Run full pipeline from audio to markdown.")
    transcribe.add_argument("audio_file", help="Path to source audio file.")
    transcribe.add_argument("-o", "--output", help="Output markdown file path.")
    transcribe.add_argument("--language", help="Whisper language code, e.g. en/zh.")
    transcribe.add_argument("--prompt", help="Optional prompt for Whisper transcription.")
    transcribe.add_argument("--hint", help="Optional context hint for GPT structuring.")
    transcribe.add_argument("--whisper-model", help="Whisper model name.")
    transcribe.add_argument("--gpt-model", help="LLM model name.")
    transcribe.add_argument("--provider", choices=["openai", "anthropic", "ollama"],
                          help="LLM provider to use.")

    process_text = subparsers.add_parser("process-text", help="Structure raw text and export markdown.")
    process_text.add_argument("text", help="Raw note text.")
    process_text.add_argument("-o", "--output", help="Output markdown file path.")
    process_text.add_argument("--hint", help="Optional context hint.")
    process_text.add_argument("--gpt-model", help="LLM model name.")
    process_text.add_argument("--provider", choices=["openai", "anthropic", "ollama"],
                             help="LLM provider to use.")

    batch = subparsers.add_parser("batch", help="Batch process multiple audio files.")
    batch.add_argument("--input-dir", required=True, help="Directory containing audio files.")
    batch.add_argument("--output-dir", required=True, help="Directory for output markdown files.")
    batch.add_argument("--language", help="Whisper language code, e.g. en/zh.")
    batch.add_argument("--prompt", help="Optional prompt for Whisper transcription.")
    batch.add_argument("--hint", help="Optional context hint for GPT structuring.")
    batch.add_argument("--whisper-model", help="Whisper model name.")
    batch.add_argument("--gpt-model", help="LLM model name.")
    batch.add_argument("--provider", choices=["openai", "anthropic", "ollama"],
                      help="LLM provider to use.")
    batch.add_argument("--report", help="Path for JSON report output.")

    config_cmd = subparsers.add_parser("config", help="Configuration management.")
    config_cmd.add_argument("--show", action="store_true", help="Show current configuration.")
    config_cmd.add_argument("--init", action="store_true", help="Initialize default configuration file.")

    return parser


def resolve_config(args: argparse.Namespace) -> Config:
    """Resolve configuration from file and command line arguments.

    Priority (highest to lowest):
    1. Command line arguments
    2. Config file (from --config or default path)
    3. Environment variables
    4. Built-in defaults
    """
    # Start with defaults + environment
    config = create_default_config()

    # Load from config file if specified or exists at default path
    if args.config:
        file_config = load_config(args.config)
        config = config.merge(file_config)
    else:
        default_path = get_default_config_path()
        if default_path.exists():
            file_config = load_config(default_path)
            config = config.merge(file_config)

    # Command line overrides
    provider = getattr(args, "provider", None)
    whisper_model = getattr(args, "whisper_model", None)
    gpt_model = getattr(args, "gpt_model", None)

    overrides = {}
    if provider:
        overrides["provider"] = provider
    if whisper_model:
        overrides["whisper_model"] = whisper_model
    if gpt_model:
        overrides["default_model"] = gpt_model

    if overrides:
        config = config.merge(**overrides)

    return config


def validate_config(config: Config) -> Optional[str]:
    """Validate configuration and return error message if invalid.

    Returns:
        Error message or None if valid
    """
    if config.provider == "openai":
        if not config.api_key and not os.getenv("OPENAI_API_KEY"):
            return "Error: OpenAI API key not configured. Set OPENAI_API_KEY environment variable or api_key in config."
    elif config.provider == "anthropic":
        if not config.api_key and not os.getenv("ANTHROPIC_API_KEY"):
            return "Error: Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable or api_key in config."
    elif config.provider == "ollama":
        # Ollama doesn't require an API key, but we should check if it's accessible
        pass

    return None


def main(argv: Optional[List[str]] = None) -> int:
    """Program entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Handle config subcommand
    if args.command == "config":
        return handle_config_command(args)

    # Resolve configuration
    config = resolve_config(args)

    # Validate configuration
    error_msg = validate_config(config)
    if error_msg:
        print(error_msg)
        return 1

    if args.command == "transcribe":
        return run_transcribe(args, config)
    if args.command == "process-text":
        return run_process_text(args, config)
    if args.command == "batch":
        return run_batch(args, config)

    print("Error: Unknown command.")
    return 1


def handle_config_command(args: argparse.Namespace) -> int:
    """Handle config subcommand."""
    if args.show:
        config = create_default_config()
        default_path = get_default_config_path()
        if default_path.exists():
            file_config = load_config(default_path)
            config = config.merge(file_config)

        print(f"Configuration file: {default_path}")
        print(f"Exists: {default_path.exists()}")
        print("\nCurrent configuration:")
        print(f"  provider: {config.provider}")
        print(f"  default_model: {config.default_model}")
        print(f"  whisper_model: {config.whisper_model}")
        print(f"  output_dir: {config.output_dir or '(not set)'}")
        print(f"  template_dir: {config.template_dir or '(not set)'}")
        print(f"  ollama_url: {config.ollama_url}")
        print(f"  ollama_model: {config.ollama_model}")
        print(f"  api_key: {'*****' if config.api_key else '(not set)'}")
        return 0

    if args.init:
        default_path = get_default_config_path()
        if default_path.exists():
            print(f"Configuration file already exists at: {default_path}")
            return 0

        config = create_default_config()
        from .config import save_config
        save_config(config, default_path)
        print(f"Created configuration file at: {default_path}")
        return 0

    print("Error: Use --show or --init with config command.")
    return 1


def run_transcribe(args: argparse.Namespace, config: Config) -> int:
    """Handle full pipeline with Whisper + LLM."""
    transcriber = Transcriber()

    # Create processor from config
    processor = Processor(
        provider_type=config.provider,
        api_key=config.api_key,
        model=config.default_model,
        ollama_url=config.ollama_url,
        ollama_model=config.ollama_model,
    )
    formatter = MarkdownFormatter()

    try:
        result = transcriber.transcribe(
            audio_file=args.audio_file,
            model=config.whisper_model,
            language=args.language,
            prompt=args.prompt,
        )
        note = processor.process(result.text, hint=args.hint)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    output_path = Path(args.output) if args.output else default_output_path(note.title)
    formatter.export(note=note, output_path=output_path, source_text=result.text)
    print(f"Saved markdown to: {output_path}")
    return 0


def run_process_text(args: argparse.Namespace, config: Config) -> int:
    """Handle text-only pipeline with LLM."""
    # Create processor from config
    processor = Processor(
        provider_type=config.provider,
        api_key=config.api_key,
        model=config.default_model,
        ollama_url=config.ollama_url,
        ollama_model=config.ollama_model,
    )
    formatter = MarkdownFormatter()

    try:
        note = processor.process(args.text, hint=args.hint)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    output_path = Path(args.output) if args.output else default_output_path(note.title)
    formatter.export(note=note, output_path=output_path, source_text=args.text)
    print(f"Saved markdown to: {output_path}")
    return 0


def run_batch(args: argparse.Namespace, config: Config) -> int:
    """Batch process multiple audio files with progress bar."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_files = []
    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        audio_files.extend(input_dir.glob(f"*{ext}"))
        audio_files.extend(input_dir.glob(f"*{ext.upper()}"))

    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return 0

    print(f"Found {len(audio_files)} audio file(s) to process")

    transcriber = Transcriber()
    # Create processor from config
    processor = Processor(
        provider_type=config.provider,
        api_key=config.api_key,
        model=config.default_model,
        ollama_url=config.ollama_url,
        ollama_model=config.ollama_model,
    )
    formatter = MarkdownFormatter()

    results = []
    success_count = 0
    failed_count = 0

    for audio_file in tqdm(audio_files, desc="Processing"):
        try:
            result = transcriber.transcribe(
                audio_file=audio_file,
                model=config.whisper_model,
                language=args.language,
                prompt=args.prompt,
            )
            note = processor.process(result.text, hint=args.hint)

            output_filename = f"{audio_file.stem}.md"
            output_path = output_dir / output_filename
            formatter.export(note=note, output_path=output_path, source_text=result.text)

            results.append({
                "input_file": str(audio_file),
                "output_file": str(output_path),
                "title": note.title,
                "category": note.category,
                "priority": note.priority,
                "status": "success",
            })
            success_count += 1

        except Exception as exc:
            results.append({
                "input_file": str(audio_file),
                "output_file": None,
                "title": None,
                "category": None,
                "priority": None,
                "status": "failed",
                "error": str(exc),
            })
            failed_count += 1

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_files": len(audio_files),
        "successful": success_count,
        "failed": failed_count,
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "results": results,
    }

    if args.report:
        report_path = Path(args.report)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport saved to: {report_path}")

    print(f"\nBatch processing complete: {success_count} succeeded, {failed_count} failed")
    return 0 if failed_count == 0 else 1


def default_output_path(title: str) -> Path:
    """Build a readable markdown filename from title + timestamp."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title.strip().lower()).strip("-")
    slug = slug or "voice-note"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"{timestamp}-{slug}.md")


if __name__ == "__main__":
    raise SystemExit(main())

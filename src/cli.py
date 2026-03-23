"""CLI entry point for the AI voice notes assistant."""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from .formatter import MarkdownFormatter
from .processor import Processor
from .transcriber import Transcriber


load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    """Build and return command parser."""
    parser = argparse.ArgumentParser(
        prog="voice-notes",
        description="Transcribe audio and export structured markdown notes.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe = subparsers.add_parser("transcribe", help="Run full pipeline from audio to markdown.")
    transcribe.add_argument("audio_file", help="Path to source audio file.")
    transcribe.add_argument("-o", "--output", help="Output markdown file path.")
    transcribe.add_argument("--language", help="Whisper language code, e.g. en/zh.")
    transcribe.add_argument("--prompt", help="Optional prompt for Whisper transcription.")
    transcribe.add_argument("--hint", help="Optional context hint for GPT structuring.")
    transcribe.add_argument("--whisper-model", default=os.getenv("WHISPER_MODEL", "whisper-1"))
    transcribe.add_argument("--gpt-model", default=os.getenv("GPT_MODEL", "gpt-4o"))

    process_text = subparsers.add_parser("process-text", help="Structure raw text and export markdown.")
    process_text.add_argument("text", help="Raw note text.")
    process_text.add_argument("-o", "--output", help="Output markdown file path.")
    process_text.add_argument("--hint", help="Optional context hint.")
    process_text.add_argument("--gpt-model", default=os.getenv("GPT_MODEL", "gpt-4o"))

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Program entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not configured.")
        return 1

    if args.command == "transcribe":
        return run_transcribe(args)
    if args.command == "process-text":
        return run_process_text(args)

    print("Error: Unknown command.")
    return 1


def run_transcribe(args: argparse.Namespace) -> int:
    """Handle full pipeline with Whisper + GPT-4."""
    transcriber = Transcriber()
    processor = Processor(model=args.gpt_model)
    formatter = MarkdownFormatter()

    try:
        result = transcriber.transcribe(
            audio_file=args.audio_file,
            model=args.whisper_model,
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


def run_process_text(args: argparse.Namespace) -> int:
    """Handle text-only pipeline with GPT-4."""
    processor = Processor(model=args.gpt_model)
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


def default_output_path(title: str) -> Path:
    """Build a readable markdown filename from title + timestamp."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title.strip().lower()).strip("-")
    slug = slug or "voice-note"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(f"{timestamp}-{slug}.md")


if __name__ == "__main__":
    raise SystemExit(main())

"""
Command-line interface for Voice Notes Assistant.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from .transcriber import Transcriber
from .processor import Processor
from .formatter import MarkdownFormatter

load_dotenv()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Voice Notes Assistant - Structure your voice notes automatically"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version="%(prog)s 0.1.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Transcribe command
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe audio file to structured note"
    )
    transcribe_parser.add_argument(
        "audio_file",
        help="Path to audio file (mp3, wav, m4a, etc.)"
    )
    transcribe_parser.add_argument(
        "-o", "--output",
        help="Output markdown file path"
    )
    transcribe_parser.add_argument(
        "-l", "--language",
        help="Language code (e.g., zh, en)"
    )
    transcribe_parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple format instead of full markdown"
    )
    
    # Process command (text input)
    process_parser = subparsers.add_parser(
        "process",
        help="Process text input to structured note"
    )
    process_parser.add_argument(
        "text",
        help="Text to process"
    )
    process_parser.add_argument(
        "-o", "--output",
        help="Output markdown file path"
    )
    
    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Show configuration"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    if args.command == "transcribe":
        return transcribe_command(args)
    elif args.command == "process":
        return process_command(args)
    elif args.command == "config":
        return config_command()
    
    return 0


def transcribe_command(args) -> int:
    """Handle transcribe command."""
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Please set it in .env file or environment.")
        return 1
    
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {args.audio_file}")
        return 1
    
    print(f"🎤 Transcribing: {audio_path.name}")
    
    # Transcribe
    transcriber = Transcriber()
    try:
        result = transcriber.transcribe(
            str(audio_path),
            language=args.language
        )
        print(f"✅ Transcription complete ({len(result['text'])} characters)")
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return 1
    
    # Process
    print("🧠 Processing with GPT-4...")
    processor = Processor()
    try:
        note = processor.process(result["text"])
        print(f"✅ Structured as: {note.category} | {note.priority} priority")
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return 1
    
    # Format
    formatter = MarkdownFormatter()
    if args.simple:
        markdown = formatter.format_simple(note)
    else:
        markdown = formatter.format(note, created_at=datetime.now())
    
    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"💾 Saved to: {output_path}")
    else:
        print("\n" + "="*60)
        print(markdown)
        print("="*60)
    
    return 0


def process_command(args) -> int:
    """Handle process command."""
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        return 1
    
    print("🧠 Processing text...")
    
    processor = Processor()
    try:
        note = processor.process(args.text)
        print(f"✅ Structured as: {note.category} | {note.priority} priority")
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return 1
    
    formatter = MarkdownFormatter()
    markdown = formatter.format(note, created_at=datetime.now())
    
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
        print(f"💾 Saved to: {args.output}")
    else:
        print("\n" + "="*60)
        print(markdown)
        print("="*60)
    
    return 0


def config_command() -> int:
    """Show configuration."""
    print("Voice Notes Assistant Configuration")
    print("="*40)
    print(f"OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}")
    print(f"PINECONE_API_KEY: {'✅ Set' if os.getenv('PINECONE_API_KEY') else '⚪ Optional'}")
    print(f"NOTION_TOKEN: {'✅ Set' if os.getenv('NOTION_TOKEN') else '⚪ Optional'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

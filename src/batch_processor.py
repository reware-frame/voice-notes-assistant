"""Batch processing module for voice notes assistant."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

from .formatter import MarkdownFormatter
from .processor import Processor
from .transcriber import SUPPORTED_AUDIO_EXTENSIONS, Transcriber


@dataclass(frozen=True)
class BatchItemResult:
    """Result of processing a single audio file."""

    input_file: str
    output_file: Optional[str]
    success: bool
    error: Optional[str]
    note_title: Optional[str]


@dataclass(frozen=True)
class BatchReport:
    """Summary report for a batch processing run."""

    total: int
    succeeded: int
    failed: int
    generated_at: str
    results: List[BatchItemResult]

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "generated_at": self.generated_at,
            "results": [
                {
                    "input_file": r.input_file,
                    "output_file": r.output_file,
                    "success": r.success,
                    "error": r.error,
                    "note_title": r.note_title,
                }
                for r in self.results
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class BatchProcessor:
    """Orchestrates batch transcription and structuring of audio files."""

    def __init__(
        self,
        transcriber: Optional[Transcriber] = None,
        processor: Optional[Processor] = None,
        formatter: Optional[MarkdownFormatter] = None,
        whisper_model: str = "whisper-1",
        gpt_model: str = "gpt-4o",
    ) -> None:
        self.transcriber = transcriber or Transcriber()
        self.processor = processor or Processor(model=gpt_model)
        self.formatter = formatter or MarkdownFormatter()
        self.whisper_model = whisper_model

    def collect_audio_files(self, input_dir: Path) -> List[Path]:
        """Return all supported audio files inside input_dir (sorted, non-recursive)."""
        return [
            f
            for f in sorted(input_dir.iterdir())
            if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        ]

    def process_file(
        self,
        audio_file: Path,
        output_dir: Path,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        hint: Optional[str] = None,
    ) -> BatchItemResult:
        """Process a single audio file and write the markdown output."""
        try:
            transcription = self.transcriber.transcribe(
                audio_file=audio_file,
                model=self.whisper_model,
                language=language,
                prompt=prompt,
            )
            note = self.processor.process(transcription.text, hint=hint)
            output_path = output_dir / f"{audio_file.stem}.md"
            self.formatter.export(note=note, output_path=output_path, source_text=transcription.text)
            return BatchItemResult(
                input_file=str(audio_file),
                output_file=str(output_path),
                success=True,
                error=None,
                note_title=note.title,
            )
        except Exception as exc:  # noqa: BLE001
            return BatchItemResult(
                input_file=str(audio_file),
                output_file=None,
                success=False,
                error=str(exc),
                note_title=None,
            )

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        hint: Optional[str] = None,
        show_progress: bool = True,
    ) -> BatchReport:
        """Process all supported audio files in input_dir and return a BatchReport."""
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_files = self.collect_audio_files(input_dir)

        iterator: Iterator[Path] = iter(audio_files)
        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore[import]

                iterator = tqdm(audio_files, desc="Processing audio files", unit="file")
            except ImportError:
                pass

        results: List[BatchItemResult] = []
        for audio_file in iterator:
            results.append(
                self.process_file(
                    audio_file=audio_file,
                    output_dir=output_dir,
                    language=language,
                    prompt=prompt,
                    hint=hint,
                )
            )

        succeeded = sum(1 for r in results if r.success)
        return BatchReport(
            total=len(results),
            succeeded=succeeded,
            failed=len(results) - succeeded,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            results=results,
        )

"""Markdown formatter for structured notes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from .processor import StructuredNote


class MarkdownFormatter:
    """Formats structured notes into markdown output."""

    def format(
        self,
        note: StructuredNote,
        source_text: Optional[str] = None,
        generated_at: Optional[datetime] = None,
    ) -> str:
        """Render note as markdown text."""
        created = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M")
        tags = ", ".join(f"`{tag}`" for tag in note.tags) if note.tags else "`untagged`"

        lines = [
            f"# {note.title}",
            "",
            "## Metadata",
            "",
            f"- Category: `{note.category}`",
            f"- Priority: `{note.priority}`",
            f"- Tags: {tags}",
            f"- Generated At: `{created}`",
            "",
            "## Summary",
            "",
            note.summary,
        ]

        if note.action_items:
            lines.extend(["", "## Action Items", ""])
            lines.extend(f"- [ ] {item}" for item in note.action_items)

        if source_text:
            lines.extend(["", "## Source Transcript", "", "```text", source_text.strip(), "```"])

        return "\n".join(lines).rstrip() + "\n"

    def export(
        self,
        note: StructuredNote,
        output_path: Union[str, Path],
        source_text: Optional[str] = None,
        generated_at: Optional[datetime] = None,
    ) -> Path:
        """Write markdown note to disk and return output path."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            self.format(note=note, source_text=source_text, generated_at=generated_at),
            encoding="utf-8",
        )
        return target

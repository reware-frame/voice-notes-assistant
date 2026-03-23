from datetime import datetime

from src.formatter import MarkdownFormatter
from src.processor import StructuredNote


def sample_note() -> StructuredNote:
    return StructuredNote(
        title="Weekly sync",
        category="meeting",
        tags=["team", "sync"],
        priority="medium",
        summary="Discussed progress and blockers.",
        action_items=["Share recap", "Update roadmap"],
    )


def test_format_contains_required_sections():
    formatter = MarkdownFormatter()
    markdown = formatter.format(
        note=sample_note(),
        source_text="we discussed milestones",
        generated_at=datetime(2026, 3, 23, 9, 30),
    )

    assert "# Weekly sync" in markdown
    assert "## Metadata" in markdown
    assert "- Category: `meeting`" in markdown
    assert "- Priority: `medium`" in markdown
    assert "## Summary" in markdown
    assert "## Action Items" in markdown
    assert "## Source Transcript" in markdown


def test_export_writes_file(tmp_path):
    formatter = MarkdownFormatter()
    output_file = tmp_path / "note.md"

    formatter.export(sample_note(), output_file)

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "Weekly sync" in content

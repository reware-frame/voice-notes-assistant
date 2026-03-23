"""
Tests for formatter module.
"""

import pytest
from datetime import datetime

from src.formatter import MarkdownFormatter
from src.processor import StructuredNote


class TestMarkdownFormatter:
    """Test cases for MarkdownFormatter."""
    
    @pytest.fixture
    def sample_note(self):
        """Create a sample note for testing."""
        return StructuredNote(
            title="Test Idea",
            category="idea",
            priority="high",
            summary="This is a test summary",
            key_points=["Point 1", "Point 2"],
            tags=["test", "idea"],
            sentiment="excited",
            action_items=["Action 1"]
        )
    
    def test_format(self, sample_note):
        """Test full markdown formatting."""
        formatter = MarkdownFormatter()
        result = formatter.format(sample_note, created_at=datetime(2024, 3, 23, 10, 0))
        
        assert "# 💡 Test Idea" in result
        assert "**Category**: idea" in result
        assert "**Priority**: 🔴 high" in result
        assert "**Sentiment**: 🤩 excited" in result
        assert "**Tags**: `test`, `idea`" in result
        assert "## Summary" in result
        assert "## Key Points" in result
        assert "- Point 1" in result
        assert "## Action Items" in result
        assert "1. [ ] Action 1" in result
    
    def test_format_simple(self, sample_note):
        """Test simple formatting."""
        formatter = MarkdownFormatter()
        result = formatter.format_simple(sample_note)
        
        assert "## 🔴 💡 Test Idea" in result
        assert "**IDEA**" in result
        assert "test, idea" in result
        assert "Actions:" in result
        assert "- [ ] Action 1" in result

"""
Tests for processor module.
"""

import pytest
import json
from unittest.mock import Mock, patch

from src.processor import Processor, StructuredNote


class TestProcessor:
    """Test cases for Processor class."""
    
    @patch('src.processor.openai.OpenAI')
    def test_init(self, mock_openai):
        """Test initialization."""
        processor = Processor(api_key="test-key")
        mock_openai.assert_called_once_with(api_key="test-key")
    
    @patch('src.processor.openai.OpenAI')
    def test_process_success(self, mock_openai):
        """Test successful processing."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "title": "Test Note",
            "category": "idea",
            "priority": "high",
            "summary": "Test summary",
            "key_points": ["Point 1"],
            "tags": ["test"],
            "sentiment": "excited",
            "action_items": ["Action 1"]
        })
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        processor = Processor(api_key="test-key")
        result = processor.process("This is a test note")
        
        assert isinstance(result, StructuredNote)
        assert result.title == "Test Note"
        assert result.category == "idea"
        assert result.priority == "high"
    
    @patch('src.processor.openai.OpenAI')
    def test_batch_process(self, mock_openai):
        """Test batch processing."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "title": "Test",
            "category": "memo",
            "priority": "medium",
            "summary": "Summary",
            "key_points": [],
            "tags": [],
            "sentiment": "neutral",
            "action_items": []
        })
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        processor = Processor(api_key="test-key")
        results = processor.batch_process(["text1", "text2"])
        
        assert len(results) == 2
        assert all(isinstance(r, StructuredNote) for r in results)

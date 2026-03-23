"""
Tests for transcriber module.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from src.transcriber import Transcriber


class TestTranscriber:
    """Test cases for Transcriber class."""
    
    @patch('src.transcriber.openai.OpenAI')
    def test_init_with_api_key(self, mock_openai):
        """Test initialization with explicit API key."""
        transcriber = Transcriber(api_key="test-key")
        mock_openai.assert_called_once_with(api_key="test-key")
    
    @patch('src.transcriber.openai.OpenAI')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key'})
    def test_init_with_env_var(self, mock_openai):
        """Test initialization with environment variable."""
        transcriber = Transcriber()
        mock_openai.assert_called_once_with(api_key="env-key")
    
    @patch('src.transcriber.openai.OpenAI')
    def test_transcribe_success(self, mock_openai):
        """Test successful transcription."""
        # Mock response
        mock_response = Mock()
        mock_response.text = "Hello world"
        mock_openai.return_value.audio.transcriptions.create.return_value = mock_response
        
        transcriber = Transcriber(api_key="test-key")
        
        with patch('builtins.open', mock_open()):
            result = transcriber.transcribe("test.mp3")
        
        assert result["text"] == "Hello world"
        assert result["model"] == "whisper-1"
    
    def test_transcribe_file_not_found(self):
        """Test transcription with missing file."""
        transcriber = Transcriber(api_key="test-key")
        
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe("nonexistent.mp3")

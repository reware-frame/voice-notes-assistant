"""Tests for processor module with multiple LLM providers."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.processor import (
    ALLOWED_CATEGORIES,
    ALLOWED_PRIORITIES,
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
    Processor,
    StructuredNote,
)


class FakeLLMProvider:
    """Fake LLM provider for testing."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.content


def test_process_successfully_parses_json():
    """Test successful JSON parsing from LLM response."""
    payload = {
        "title": "Sprint planning",
        "category": "todo",
        "tags": ["product", "team"],
        "priority": "high",
        "summary": "Need to prepare sprint scope and owners.",
        "action_items": ["Draft ticket list", "Book meeting room"],
    }
    fake_provider = FakeLLMProvider(content=json.dumps(payload))
    processor = Processor(provider=fake_provider)

    note = processor.process("Tomorrow we should plan sprint tasks")

    assert note.title == "Sprint planning"
    assert note.category == "todo"
    assert note.priority == "high"
    assert note.tags == ["product", "team"]
    assert len(note.action_items) == 2


def test_process_invalid_json_raises_error():
    """Test error raised for invalid JSON response."""
    fake_provider = FakeLLMProvider(content="not-json")
    processor = Processor(provider=fake_provider)

    with pytest.raises(ValueError, match="not valid JSON"):
        processor.process("A quick note")


def test_process_normalizes_invalid_values():
    """Test normalization of invalid values from LLM."""
    payload = {
        "title": "Random",
        "category": "anything",
        "tags": ["alpha", "alpha", ""],
        "priority": "urgent",
        "summary": "Summary",
        "action_items": "not-a-list",
    }
    fake_provider = FakeLLMProvider(content=json.dumps(payload))
    processor = Processor(provider=fake_provider)

    note = processor.process("test")

    assert note.category == "idea"  # Invalid category falls back to default
    assert note.priority == "medium"  # Invalid priority falls back to default
    assert note.tags == ["alpha"]  # Duplicates and empty strings removed
    assert note.action_items == []  # Non-list becomes empty


def test_process_empty_transcript_raises_error():
    """Test error raised for empty transcript."""
    fake_provider = FakeLLMProvider(content='{"title": "test"}')
    processor = Processor(provider=fake_provider)

    with pytest.raises(ValueError, match="cannot be empty"):
        processor.process("   ")


def test_batch_process():
    """Test batch processing multiple transcripts."""
    payload = {
        "title": "Note",
        "category": "idea",
        "tags": [],
        "priority": "medium",
        "summary": "Summary",
        "action_items": [],
    }
    fake_provider = FakeLLMProvider(content=json.dumps(payload))
    processor = Processor(provider=fake_provider)

    notes = processor.batch_process(["First note", "Second note", "Third note"])

    assert len(notes) == 3
    assert all(isinstance(note, StructuredNote) for note in notes)
    assert len(fake_provider.calls) == 3


def test_normalize_payload_handles_missing_fields():
    """Test normalization handles missing fields."""
    fake_provider = FakeLLMProvider(content='{"title": "Test"}')
    processor = Processor(provider=fake_provider)

    note = processor.process("test input")

    assert note.title == "Test"
    assert note.category == "idea"  # Default
    assert note.priority == "medium"  # Default
    assert note.tags == []
    assert note.action_items == []
    assert note.summary == "No summary provided."


def test_clean_list_removes_duplicates_and_empty():
    """Test _clean_list removes duplicates and empty strings."""
    processor = Processor.__new__(Processor)

    result = processor._clean_list(["a", "b", "a", "", "c", "  "])
    assert result == ["a", "b", "c"]


def test_clean_list_handles_non_list():
    """Test _clean_list handles non-list values."""
    processor = Processor.__new__(Processor)

    assert processor._clean_list("not a list") == []
    assert processor._clean_list(None) == []
    assert processor._clean_list(123) == []


def test_allowed_categories():
    """Test allowed categories constant."""
    assert ALLOWED_CATEGORIES == ("idea", "todo", "meeting", "inspiration")


def test_allowed_priorities():
    """Test allowed priorities constant."""
    assert ALLOWED_PRIORITIES == ("high", "medium", "low")


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(api_key="test-key", model="gpt-4")

            mock_openai.assert_called_once_with(api_key="test-key")
            assert provider.model == "gpt-4"

    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises error."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key not configured"):
                OpenAIProvider()

    def test_generate_calls_api(self):
        """Test generate method calls OpenAI API correctly."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content='{"title": "Test"}'))]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(api_key="test-key")
            result = provider.generate("system prompt", "user prompt")

            assert result == '{"title": "Test"}'
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "gpt-4o"
            assert call_kwargs["temperature"] == 0.2
            assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_generate_empty_response_raises_error(self):
        """Test error raised for empty response."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content=None))]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(api_key="test-key")

            with pytest.raises(ValueError, match="empty response"):
                provider.generate("system", "user")


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            provider = AnthropicProvider(api_key="anthropic-key", model="claude-3-opus")

            mock_anthropic.assert_called_once_with(api_key="anthropic-key")
            assert provider.model == "claude-3-opus"

    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises error."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=True):
            with pytest.raises(ValueError, match="Anthropic API key not configured"):
                AnthropicProvider()

    def test_generate_calls_api(self):
        """Test generate method calls Anthropic API correctly."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='{"title": "Claude Test"}')]
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            provider = AnthropicProvider(api_key="test-key")
            result = provider.generate("system prompt", "user prompt")

            assert result == '{"title": "Claude Test"}'
            mock_client.messages.create.assert_called_once()
            call_kwargs = mock_client.messages.create.call_args[1]
            assert call_kwargs["model"] == "claude-3-opus-20240229"
            assert call_kwargs["max_tokens"] == 4096
            assert call_kwargs["temperature"] == 0.2

    def test_generate_empty_response_raises_error(self):
        """Test error raised for empty response."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = []
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            provider = AnthropicProvider(api_key="test-key")

            with pytest.raises(ValueError, match="empty response"):
                provider.generate("system", "user")


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        provider = OllamaProvider()
        assert provider.base_url == "http://localhost:11434"
        assert provider.model == "llama2"

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        provider = OllamaProvider(base_url="http://custom:8080", model="mistral")
        assert provider.base_url == "http://custom:8080"
        assert provider.model == "mistral"

    def test_init_strips_trailing_slash(self):
        """Test base URL has trailing slash stripped."""
        provider = OllamaProvider(base_url="http://localhost:11434/")
        assert provider.base_url == "http://localhost:11434"

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_generate_calls_api(self, mock_request_class, mock_urlopen):
        """Test generate method calls Ollama API correctly."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": '{"title": "Ollama Test"}'}).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        mock_request = MagicMock()
        mock_request_class.return_value = mock_request

        provider = OllamaProvider()
        result = provider.generate("system prompt", "user prompt")

        assert result == '{"title": "Ollama Test"}'
        mock_request_class.assert_called_once()
        call_args = mock_request_class.call_args[0]
        assert call_args[0] == "http://localhost:11434/api/generate"

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_generate_empty_response_raises_error(self, mock_request_class, mock_urlopen):
        """Test error raised for empty response."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": ""}).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        mock_request = MagicMock()
        mock_request_class.return_value = mock_request

        provider = OllamaProvider()

        with pytest.raises(ValueError, match="empty response"):
            provider.generate("system", "user")

    @patch("urllib.request.urlopen")
    @patch("urllib.request.Request")
    def test_generate_request_failure_raises_error(self, mock_request_class, mock_urlopen):
        """Test error raised when request fails."""
        mock_urlopen.side_effect = Exception("Connection refused")

        mock_request = MagicMock()
        mock_request_class.return_value = mock_request

        provider = OllamaProvider()

        with pytest.raises(ValueError, match="Ollama request failed"):
            provider.generate("system", "user")


class TestProcessorFromConfig:
    """Tests for Processor.from_config factory method."""

    def test_from_config_with_openai(self):
        """Test creating Processor from config with OpenAI."""
        config = SimpleNamespace(
            provider="openai",
            api_key="test-key",
            default_model="gpt-4-turbo",
        )

        with patch("src.processor.OpenAIProvider") as mock_provider:
            mock_instance = MagicMock()
            mock_provider.return_value = mock_instance

            processor = Processor.from_config(config)

            mock_provider.assert_called_once_with(api_key="test-key", model="gpt-4-turbo")
            assert processor.provider == mock_instance

    def test_from_config_with_anthropic(self):
        """Test creating Processor from config with Anthropic."""
        config = SimpleNamespace(
            provider="anthropic",
            api_key="anthropic-key",
            default_model="claude-3-opus",
        )

        with patch("src.processor.AnthropicProvider") as mock_provider:
            mock_instance = MagicMock()
            mock_provider.return_value = mock_instance

            processor = Processor.from_config(config)

            mock_provider.assert_called_once_with(api_key="anthropic-key", model="claude-3-opus")
            assert processor.provider == mock_instance

    def test_from_config_with_ollama(self):
        """Test creating Processor from config with Ollama."""
        config = SimpleNamespace(
            provider="ollama",
            ollama_url="http://custom:11434",
            ollama_model="mistral",
        )

        with patch("src.processor.OllamaProvider") as mock_provider:
            mock_instance = MagicMock()
            mock_provider.return_value = mock_instance

            processor = Processor.from_config(config)

            mock_provider.assert_called_once_with(base_url="http://custom:11434", model="mistral")
            assert processor.provider == mock_instance

    def test_from_config_uses_defaults(self):
        """Test from_config uses default values when not specified."""
        config = SimpleNamespace()

        with patch("src.processor.OpenAIProvider") as mock_provider:
            mock_instance = MagicMock()
            mock_provider.return_value = mock_instance

            processor = Processor.from_config(config)

            mock_provider.assert_called_once_with(api_key=None, model="gpt-4o")


class TestProcessorProviderTypes:
    """Tests for Processor initialization with different provider types."""

    @patch("src.processor.OpenAIProvider")
    def test_init_with_openai_provider_type(self, mock_provider_class):
        """Test Processor with provider_type='openai'."""
        mock_instance = MagicMock()
        mock_provider_class.return_value = mock_instance

        processor = Processor(provider_type="openai", api_key="key", model="gpt-4")

        mock_provider_class.assert_called_once_with(api_key="key", model="gpt-4")
        assert processor.provider == mock_instance

    @patch("src.processor.AnthropicProvider")
    def test_init_with_anthropic_provider_type(self, mock_provider_class):
        """Test Processor with provider_type='anthropic'."""
        mock_instance = MagicMock()
        mock_provider_class.return_value = mock_instance

        processor = Processor(provider_type="anthropic", api_key="key", model="claude-3")

        mock_provider_class.assert_called_once_with(api_key="key", model="claude-3")
        assert processor.provider == mock_instance

    @patch("src.processor.OllamaProvider")
    def test_init_with_ollama_provider_type(self, mock_provider_class):
        """Test Processor with provider_type='ollama'."""
        mock_instance = MagicMock()
        mock_provider_class.return_value = mock_instance

        processor = Processor(
            provider_type="ollama",
            ollama_url="http://localhost:8080",
            ollama_model="llama3",
        )

        mock_provider_class.assert_called_once_with(base_url="http://localhost:8080", model="llama3")
        assert processor.provider == mock_instance

    def test_init_with_unknown_provider_type_raises_error(self):
        """Test Processor raises error for unknown provider_type."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            Processor(provider_type="unknown")

    def test_init_with_explicit_provider(self):
        """Test Processor with explicitly provided provider."""
        fake_provider = FakeLLMProvider(content='{"title": "Test"}')
        processor = Processor(provider=fake_provider)

        assert processor.provider == fake_provider

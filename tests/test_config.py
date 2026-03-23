"""Tests for configuration module."""

import json
import os
from pathlib import Path

import pytest
import yaml

from src.config import (
    Config,
    create_default_config,
    get_default_config_path,
    load_config,
    save_config,
)


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_values(self):
        """Test Config has correct default values."""
        config = Config()
        assert config.api_key is None
        assert config.default_model == "gpt-4o"
        assert config.whisper_model == "whisper-1"
        assert config.output_dir is None
        assert config.template_dir is None
        assert config.provider == "openai"
        assert config.ollama_url == "http://localhost:11434"
        assert config.ollama_model == "llama2"

    def test_from_dict(self):
        """Test creating Config from dictionary."""
        data = {
            "api_key": "test-key",
            "default_model": "claude-3-opus",
            "provider": "anthropic",
            "ollama_model": "mistral",
        }
        config = Config.from_dict(data)
        assert config.api_key == "test-key"
        assert config.default_model == "claude-3-opus"
        assert config.provider == "anthropic"
        assert config.ollama_model == "mistral"
        # Defaults should remain for unspecified fields
        assert config.whisper_model == "whisper-1"

    def test_from_dict_ignores_unknown_fields(self):
        """Test that unknown fields are ignored."""
        data = {
            "api_key": "test-key",
            "unknown_field": "should_be_ignored",
            "another_unknown": 123,
        }
        config = Config.from_dict(data)
        assert config.api_key == "test-key"

    def test_to_dict(self):
        """Test converting Config to dictionary."""
        config = Config(
            api_key="test-key",
            default_model="gpt-4-turbo",
            provider="openai",
        )
        data = config.to_dict()
        assert data["api_key"] == "test-key"
        assert data["default_model"] == "gpt-4-turbo"
        assert data["provider"] == "openai"
        assert data["whisper_model"] == "whisper-1"

    def test_merge_with_another_config(self):
        """Test merging with another Config."""
        base = Config(api_key="base-key", default_model="gpt-4o")
        override = Config(api_key="override-key", provider="anthropic")
        merged = base.merge(override)

        assert merged.api_key == "override-key"
        assert merged.provider == "anthropic"
        assert merged.default_model == "gpt-4o"  # Unchanged
        assert merged.whisper_model == "whisper-1"  # Unchanged

    def test_merge_with_kwargs(self):
        """Test merging with keyword arguments."""
        base = Config(api_key="base-key", default_model="gpt-4o")
        merged = base.merge(provider="ollama", ollama_model="llama3")

        assert merged.api_key == "base-key"  # Unchanged
        assert merged.provider == "ollama"
        assert merged.ollama_model == "llama3"
        assert merged.default_model == "gpt-4o"  # Unchanged

    def test_merge_kwargs_take_precedence(self):
        """Test that kwargs take precedence over config merge."""
        base = Config(api_key="base-key")
        override = Config(api_key="override-key")
        merged = base.merge(override, api_key="kwargs-key")

        assert merged.api_key == "kwargs-key"

    def test_merge_with_none(self):
        """Test merging with None values doesn't override."""
        base = Config(api_key="base-key", default_model="gpt-4o")
        merged = base.merge(api_key=None, provider="anthropic")

        assert merged.api_key == "base-key"  # Not overridden
        assert merged.provider == "anthropic"

    def test_immutability(self):
        """Test that Config is immutable."""
        config = Config(api_key="test-key")
        with pytest.raises(AttributeError):
            config.api_key = "new-key"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_file = tmp_path / "config.yaml"
        data = {
            "api_key": "yaml-api-key",
            "provider": "anthropic",
            "default_model": "claude-3-opus",
        }
        config_file.write_text(yaml.dump(data), encoding="utf-8")

        config = load_config(config_file)
        assert config.api_key == "yaml-api-key"
        assert config.provider == "anthropic"
        assert config.default_model == "claude-3-opus"

    def test_load_from_json(self, tmp_path):
        """Test loading configuration from JSON file."""
        config_file = tmp_path / "config.json"
        data = {
            "api_key": "json-api-key",
            "provider": "ollama",
            "ollama_model": "mistral",
        }
        config_file.write_text(json.dumps(data), encoding="utf-8")

        config = load_config(config_file)
        assert config.api_key == "json-api-key"
        assert config.provider == "ollama"
        assert config.ollama_model == "mistral"

    def test_load_from_yml_extension(self, tmp_path):
        """Test loading configuration from .yml file."""
        config_file = tmp_path / "config.yml"
        data = {"provider": "openai", "whisper_model": "whisper-1"}
        config_file.write_text(yaml.dump(data), encoding="utf-8")

        config = load_config(config_file)
        assert config.provider == "openai"
        assert config.whisper_model == "whisper-1"

    def test_load_nonexistent_file_returns_defaults(self, tmp_path):
        """Test that loading non-existent file returns default config."""
        config_file = tmp_path / "nonexistent.yaml"
        config = load_config(config_file)

        assert config.api_key is None
        assert config.provider == "openai"
        assert config.default_model == "gpt-4o"

    def test_load_without_path_uses_default(self, monkeypatch, tmp_path):
        """Test that loading without path checks default location."""
        # Create a config at the default location
        default_dir = tmp_path / ".voice-notes"
        default_dir.mkdir()
        default_file = default_dir / "config.yaml"

        data = {"api_key": "default-path-key", "provider": "anthropic"}
        default_file.write_text(yaml.dump(data), encoding="utf-8")

        # Monkeypatch the default path
        from src import config as config_module
        original_path = config_module.DEFAULT_CONFIG_PATH
        config_module.DEFAULT_CONFIG_PATH = default_file

        try:
            config = load_config()
            assert config.api_key == "default-path-key"
            assert config.provider == "anthropic"
        finally:
            config_module.DEFAULT_CONFIG_PATH = original_path

    def test_load_invalid_yaml_returns_defaults(self, tmp_path):
        """Test that invalid YAML returns default config."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [", encoding="utf-8")

        config = load_config(config_file)
        assert config.provider == "openai"  # Default value

    def test_load_invalid_json_returns_defaults(self, tmp_path):
        """Test that invalid JSON returns default config."""
        config_file = tmp_path / "config.json"
        config_file.write_text("not valid json", encoding="utf-8")

        config = load_config(config_file)
        assert config.provider == "openai"  # Default value


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_to_yaml(self, tmp_path):
        """Test saving configuration to YAML file."""
        config_file = tmp_path / "config.yaml"
        config = Config(
            api_key="test-key",
            provider="anthropic",
            default_model="claude-3-opus",
        )

        save_config(config, config_file)

        assert config_file.exists()
        content = config_file.read_text(encoding="utf-8")
        assert "api_key: test-key" in content
        assert "provider: anthropic" in content

    def test_save_to_json(self, tmp_path):
        """Test saving configuration to JSON file."""
        config_file = tmp_path / "config.json"
        config = Config(
            api_key="json-key",
            provider="ollama",
            ollama_model="llama3",
        )

        save_config(config, config_file)

        assert config_file.exists()
        data = json.loads(config_file.read_text(encoding="utf-8"))
        assert data["api_key"] == "json-key"
        assert data["provider"] == "ollama"
        assert data["ollama_model"] == "llama3"

    def test_save_creates_parent_directories(self, tmp_path):
        """Test that save_config creates parent directories."""
        config_file = tmp_path / "nested" / "dirs" / "config.yaml"
        config = Config(api_key="test-key")

        save_config(config, config_file)

        assert config_file.exists()

    def test_save_without_path_uses_default(self, monkeypatch, tmp_path):
        """Test saving without path uses default location."""
        default_dir = tmp_path / ".voice-notes"
        default_file = default_dir / "config.yaml"

        from src import config as config_module
        original_path = config_module.DEFAULT_CONFIG_PATH
        config_module.DEFAULT_CONFIG_PATH = default_file

        try:
            config = Config(api_key="default-save-key")
            save_config(config)

            assert default_file.exists()
            content = default_file.read_text(encoding="utf-8")
            assert "api_key: default-save-key" in content
        finally:
            config_module.DEFAULT_CONFIG_PATH = original_path


class TestDefaultConfigPath:
    """Tests for get_default_config_path function."""

    def test_returns_voice_notes_in_home(self):
        """Test that default path is ~/.voice-notes/config.yaml."""
        path = get_default_config_path()
        assert path.name == "config.yaml"
        assert path.parent.name == ".voice-notes"
        assert path.parent.parent == Path.home()


class TestCreateDefaultConfig:
    """Tests for create_default_config function."""

    def test_uses_environment_variables(self, monkeypatch):
        """Test that default config uses environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
        monkeypatch.setenv("GPT_MODEL", "gpt-4-turbo")
        monkeypatch.setenv("WHISPER_MODEL", "whisper-large")
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")

        config = create_default_config()

        assert config.api_key == "env-openai-key"
        assert config.default_model == "gpt-4-turbo"
        assert config.whisper_model == "whisper-large"
        assert config.provider == "anthropic"

    def test_anthropic_api_key_fallback(self, monkeypatch):
        """Test ANTHROPIC_API_KEY is used as api_key fallback."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-anthropic-key")

        config = create_default_config()

        assert config.api_key == "env-anthropic-key"

    def test_uses_hardcoded_defaults_when_no_env(self, monkeypatch):
        """Test hardcoded defaults when environment variables not set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GPT_MODEL", raising=False)
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        config = create_default_config()

        assert config.api_key is None
        assert config.default_model == "gpt-4o"
        assert config.whisper_model == "whisper-1"
        assert config.provider == "openai"

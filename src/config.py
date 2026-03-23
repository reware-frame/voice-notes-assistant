"""Configuration management for voice notes assistant.

Supports YAML and JSON configuration files with automatic loading
from default location or custom path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


DEFAULT_CONFIG_PATH = Path.home() / ".voice-notes" / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Application configuration model.

    Attributes:
        api_key: API key for the LLM provider
        default_model: Default model name for processing
        whisper_model: Whisper model for transcription
        output_dir: Default output directory for markdown files
        template_dir: Directory containing templates
        provider: LLM provider type (openai, anthropic, ollama)
        ollama_url: Ollama API URL (for ollama provider)
        ollama_model: Ollama model name (for ollama provider)
    """

    api_key: Optional[str] = None
    default_model: str = "gpt-4o"
    whisper_model: str = "whisper-1"
    output_dir: Optional[str] = None
    template_dir: Optional[str] = None
    provider: str = "openai"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create Config from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            Config instance
        """
        # Only use known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

    def merge(self, other: Optional[Config] = None, **kwargs: Any) -> Config:
        """Create merged config with overrides.

        Args:
            other: Another Config to merge (takes precedence)
            **kwargs: Individual field overrides (highest precedence)

        Returns:
            New merged Config instance
        """
        data = {
            "api_key": self.api_key,
            "default_model": self.default_model,
            "whisper_model": self.whisper_model,
            "output_dir": self.output_dir,
            "template_dir": self.template_dir,
            "provider": self.provider,
            "ollama_url": self.ollama_url,
            "ollama_model": self.ollama_model,
        }

        if other is not None:
            if other.api_key is not None:
                data["api_key"] = other.api_key
            if other.default_model != "gpt-4o":
                data["default_model"] = other.default_model
            if other.whisper_model != "whisper-1":
                data["whisper_model"] = other.whisper_model
            if other.output_dir is not None:
                data["output_dir"] = other.output_dir
            if other.template_dir is not None:
                data["template_dir"] = other.template_dir
            if other.provider != "openai":
                data["provider"] = other.provider
            if other.ollama_url != "http://localhost:11434":
                data["ollama_url"] = other.ollama_url
            if other.ollama_model != "llama2":
                data["ollama_model"] = other.ollama_model

        # kwargs take highest precedence
        for key, value in kwargs.items():
            if key in data and value is not None:
                data[key] = value

        return Config.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert Config to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "api_key": self.api_key,
            "default_model": self.default_model,
            "whisper_model": self.whisper_model,
            "output_dir": self.output_dir,
            "template_dir": self.template_dir,
            "provider": self.provider,
            "ollama_url": self.ollama_url,
            "ollama_model": self.ollama_model,
        }


def load_config(path: Optional[Path] = None) -> Config:
    """Load configuration from file.

    Supports both YAML and JSON formats. If no path is provided,
    uses the default config path.

    Args:
        path: Path to config file (optional)

    Returns:
        Config instance (returns default if file doesn't exist)
    """
    config_path = path or DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return Config()

    try:
        content = config_path.read_text(encoding="utf-8")

        if config_path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(content) or {}
        elif config_path.suffix == ".json":
            data = json.loads(content)
        else:
            # Try YAML first, then JSON
            try:
                data = yaml.safe_load(content) or {}
            except yaml.YAMLError:
                data = json.loads(content)

        return Config.from_dict(data)
    except (yaml.YAMLError, json.JSONDecodeError, OSError):
        return Config()


def save_config(config: Config, path: Optional[Path] = None) -> None:
    """Save configuration to file.

    Format is determined by file extension (.yaml, .yml, or .json).
    Defaults to YAML if no extension or unrecognized.

    Args:
        config: Config instance to save
        path: Path to save to (optional, defaults to DEFAULT_CONFIG_PATH)
    """
    config_path = path or DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()

    if config_path.suffix == ".json":
        content = json.dumps(data, indent=2, ensure_ascii=False)
    else:
        # Default to YAML
        content = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    config_path.write_text(content, encoding="utf-8")


def get_default_config_path() -> Path:
    """Get the default configuration file path.

    Returns:
        Path to default config file
    """
    return DEFAULT_CONFIG_PATH


def create_default_config() -> Config:
    """Create a default configuration with environment variable fallbacks.

    Returns:
        Config instance with defaults and env var overrides
    """
    return Config(
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
        default_model=os.getenv("GPT_MODEL", "gpt-4o"),
        whisper_model=os.getenv("WHISPER_MODEL", "whisper-1"),
        provider=os.getenv("LLM_PROVIDER", "openai"),
    )

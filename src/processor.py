"""LLM-based structuring module for voice notes.

Supports multiple providers: OpenAI, Anthropic Claude, and local Ollama.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

from dotenv import load_dotenv

load_dotenv()

ALLOWED_CATEGORIES = ("idea", "todo", "meeting", "inspiration")
ALLOWED_PRIORITIES = ("high", "medium", "low")


@dataclass(frozen=True)
class StructuredNote:
    """Structured output model used by formatter and CLI."""

    title: str
    category: str
    tags: List[str]
    priority: str
    summary: str
    action_items: List[str]


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text using the LLM.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt

        Returns:
            Generated text
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o") -> None:
        from openai import OpenAI

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OpenAI API key not configured")
        self.client = OpenAI(api_key=resolved_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned an empty response.")
        return content


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229") -> None:
        from anthropic import Anthropic

        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError("Anthropic API key not configured")
        self.client = Anthropic(api_key=resolved_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Claude doesn't have a json_object response format, so we instruct it in the prompt
        full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nYou must respond with valid JSON only."
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0.2,
            messages=[{"role": "user", "content": full_prompt}],
        )
        content = response.content[0].text if response.content else ""
        if not content:
            raise ValueError("Claude returned an empty response.")
        return content


class OllamaProvider(LLMProvider):
    """Local Ollama provider."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        import urllib.request

        full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nYou must respond with valid JSON only."
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
        }

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result.get("response", "")
                if not content:
                    raise ValueError("Ollama returned an empty response.")
                return content
        except Exception as exc:
            raise ValueError(f"Ollama request failed: {exc}") from exc


class Processor:
    """Processes transcripts using configurable LLM provider."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        provider_type: str = "openai",
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llama2",
    ) -> None:
        """Initialize processor with provider.

        Args:
            provider: Pre-configured LLM provider (takes precedence)
            provider_type: Type of provider to create (openai, anthropic, ollama)
            api_key: API key for the provider
            model: Model name
            ollama_url: Ollama base URL (for ollama provider)
            ollama_model: Ollama model name (for ollama provider)
        """
        if provider is not None:
            self.provider = provider
        elif provider_type == "openai":
            self.provider = OpenAIProvider(api_key=api_key, model=model)
        elif provider_type == "anthropic":
            self.provider = AnthropicProvider(api_key=api_key, model=model)
        elif provider_type == "ollama":
            self.provider = OllamaProvider(base_url=ollama_url, model=ollama_model)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

    def process(self, transcript: str, hint: Optional[str] = None) -> StructuredNote:
        """Convert transcript text into a structured note."""
        text = transcript.strip()
        if not text:
            raise ValueError("Transcript cannot be empty.")

        system_prompt = (
            "You are a voice-notes assistant. "
            "Return JSON only with these keys: "
            "title, category, tags, priority, summary, action_items. "
            "Allowed category values: idea, todo, meeting, inspiration. "
            "Allowed priority values: high, medium, low."
        )

        user_prompt = (
            "Transcript:\n"
            f"{text}\n\n"
            f"Hint: {hint or 'none'}\n\n"
            "Generate concise, practical structure. "
            "tags/action_items must be arrays."
        )

        content = self.provider.generate(system_prompt, user_prompt)

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response is not valid JSON.") from exc

        return self._normalize_payload(payload)

    def batch_process(self, transcripts: List[str]) -> List[StructuredNote]:
        """Process multiple transcripts sequentially."""
        return [self.process(item) for item in transcripts]

    def _normalize_payload(self, payload: dict) -> StructuredNote:
        title = str(payload.get("title") or "Untitled Note").strip()
        summary = str(payload.get("summary") or "No summary provided.").strip()

        raw_category = str(payload.get("category") or "idea").strip().lower()
        category = raw_category if raw_category in ALLOWED_CATEGORIES else "idea"

        raw_priority = str(payload.get("priority") or "medium").strip().lower()
        priority = raw_priority if raw_priority in ALLOWED_PRIORITIES else "medium"

        tags = self._clean_list(payload.get("tags"))
        action_items = self._clean_list(payload.get("action_items"))

        return StructuredNote(
            title=title,
            category=category,
            tags=tags,
            priority=priority,
            summary=summary,
            action_items=action_items,
        )

    @staticmethod
    def _clean_list(value: object) -> List[str]:
        if not isinstance(value, list):
            return []

        items: List[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in items:
                items.append(text)
        return items

    @classmethod
    def from_config(cls, config: Any) -> Processor:
        """Create Processor from config object.

        Args:
            config: Config object with provider, api_key, etc.

        Returns:
            Configured Processor instance
        """
        provider_type = getattr(config, "provider", "openai")
        api_key = getattr(config, "api_key", None)
        model = getattr(config, "default_model", "gpt-4o")
        ollama_url = getattr(config, "ollama_url", "http://localhost:11434")
        ollama_model = getattr(config, "ollama_model", "llama2")

        return cls(
            provider_type=provider_type,
            api_key=api_key,
            model=model,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
        )

"""GPT-4 based structuring module for voice notes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

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


class Processor:
    """Calls GPT-4 to structure transcribed text."""

    def __init__(self, client: Optional[OpenAI] = None, api_key: Optional[str] = None, model: str = "gpt-4o") -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = client or OpenAI(api_key=resolved_key)
        self.model = model

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
            raise ValueError("GPT-4 returned an empty response.")

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("GPT-4 response is not valid JSON.") from exc

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

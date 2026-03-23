import json
from types import SimpleNamespace

import pytest

from src.processor import Processor


class FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self, content: str) -> None:
        self.fake_completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.fake_completions)


def test_process_successfully_parses_json():
    payload = {
        "title": "Sprint planning",
        "category": "todo",
        "tags": ["product", "team"],
        "priority": "high",
        "summary": "Need to prepare sprint scope and owners.",
        "action_items": ["Draft ticket list", "Book meeting room"],
    }
    client = FakeClient(content=json.dumps(payload))
    processor = Processor(client=client)

    note = processor.process("Tomorrow we should plan sprint tasks")

    assert note.title == "Sprint planning"
    assert note.category == "todo"
    assert note.priority == "high"
    assert note.tags == ["product", "team"]
    assert len(note.action_items) == 2

    api_call = client.fake_completions.calls[0]
    assert api_call["model"] == "gpt-4o"


def test_process_invalid_json_raises_error():
    processor = Processor(client=FakeClient(content="not-json"))

    with pytest.raises(ValueError):
        processor.process("A quick note")


def test_process_normalizes_invalid_values():
    payload = {
        "title": "Random",
        "category": "anything",
        "tags": ["alpha", "alpha", ""],
        "priority": "urgent",
        "summary": "Summary",
        "action_items": "not-a-list",
    }
    processor = Processor(client=FakeClient(content=json.dumps(payload)))

    note = processor.process("test")

    assert note.category == "idea"
    assert note.priority == "medium"
    assert note.tags == ["alpha"]
    assert note.action_items == []

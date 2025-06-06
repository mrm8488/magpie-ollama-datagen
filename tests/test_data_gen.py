import json
import os
import sys
import urllib.request

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_gen import query_model


class FakeResponse:
    def __init__(self, lines):
        self._lines = lines
        self._index = 0

    def readline(self):
        if self._index >= len(self._lines):
            return b""
        line = self._lines[self._index]
        self._index += 1
        return line.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


def fake_urlopen(request):
    lines = [
        json.dumps({"message": {"content": "Hello"}, "done": False}),
        json.dumps({"done": True}),
    ]
    return FakeResponse(lines)


def test_query_model_handles_done(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda req: fake_urlopen(req))
    result = query_model("prompt", model="llama3")
    assert result == "Hello"

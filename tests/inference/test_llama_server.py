"""llama-server client request building and answer parsing, without any network."""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO

import numpy as np
import pytest

from retone.inference import llama_server
from retone.inference.errors import ModelUnavailableError
from retone.inference.llama_server import LlamaServerClient


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._buffer = BytesIO(json.dumps(payload).encode("utf-8"))

    def read(self) -> bytes:
        return self._buffer.read()

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _chat(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


def test_answer_is_parsed_into_a_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float) -> _Response:
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data)  # type: ignore[attr-defined]
        return _Response(_chat(json.dumps({"exposure": 30})))

    monkeypatch.setattr(llama_server.urllib.request, "urlopen", fake_urlopen)
    answer = LlamaServerClient().choose_params(np.zeros((4, 4, 3), dtype=np.uint8))
    assert answer == {"exposure": 30}
    body = captured["body"]
    assert body["temperature"] == 0  # type: ignore[index]
    assert body["response_format"]["type"] == "json_object"  # type: ignore[index]


def test_unreachable_server_raises_model_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_url_error(request: object, timeout: float) -> _Response:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(llama_server.urllib.request, "urlopen", raise_url_error)
    with pytest.raises(ModelUnavailableError):
        LlamaServerClient().choose_params(np.zeros((4, 4, 3), dtype=np.uint8))


def test_non_json_answer_raises_model_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: float) -> _Response:
        return _Response(_chat("not json"))

    monkeypatch.setattr(llama_server.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ModelUnavailableError):
        LlamaServerClient().choose_params(np.zeros((4, 4, 3), dtype=np.uint8))


def test_missing_content_raises_model_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: float) -> _Response:
        return _Response({"choices": []})

    monkeypatch.setattr(llama_server.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ModelUnavailableError):
        LlamaServerClient().choose_params(np.zeros((4, 4, 3), dtype=np.uint8))

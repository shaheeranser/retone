"""HTTP client for a running `llama-server`, sending the thumbnail and reading the JSON answer."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from retone.executor import INSTRUCTION, REQUEST_LIMITS, json_schema
from retone.inference.errors import ModelUnavailableError


@dataclass(frozen=True)
class LlamaServerClient:
    """Talks to llama.cpp's OpenAI-compatible endpoint with a JSON-Schema grammar.

    The grammar makes malformed JSON impossible at generation time; the caller still
    validates the values, because the server is a process boundary we do not trust.
    """

    base_url: str = "http://127.0.0.1:8080"
    timeout: float = 120.0

    def choose_params(self, thumbnail_rgb: NDArray[np.uint8]) -> Mapping[str, object]:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": INSTRUCTION},
                        {"type": "image_url", "image_url": {"url": _data_uri(thumbnail_rgb)}},
                    ],
                }
            ],
            "temperature": 0,
            "response_format": {"type": "json_object", "schema": json_schema(REQUEST_LIMITS)},
        }
        answer = _post_json(f"{self.base_url.rstrip('/')}/v1/chat/completions", body, self.timeout)
        return _content_mapping(answer)


def _data_uri(thumbnail_rgb: NDArray[np.uint8]) -> str:
    bgr = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise ModelUnavailableError("cannot encode the thumbnail for the model")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _post_json(url: str, body: Mapping[str, Any], timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as err:
        raise ModelUnavailableError(
            f"cannot reach the model server at {url}; is llama-server running? ({err.reason})"
        ) from err
    except json.JSONDecodeError as err:
        raise ModelUnavailableError(f"model server returned invalid JSON: {err.msg}") from err


def _content_mapping(answer: Any) -> Mapping[str, object]:
    try:
        content = answer["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as err:
        raise ModelUnavailableError("model server response has no message content") from err
    if not isinstance(content, str):
        raise ModelUnavailableError("model server message content is not text")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as err:
        raise ModelUnavailableError(f"model answer is not JSON: {err.msg}") from err
    if not isinstance(parsed, Mapping):
        raise ModelUnavailableError("model answer is not a JSON object")
    return parsed

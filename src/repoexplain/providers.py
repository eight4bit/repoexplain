"""Model backends.

Everything the rest of the app needs from a model server is behind the small
`Provider` interface, so Ollama is optional: any OpenAI-compatible local server
(LM Studio, llama.cpp server, vLLM, Jan, ...) works too.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

import requests

CONNECT_TIMEOUT = 5
READ_TIMEOUT = 600  # first call can be slow while a model loads into RAM


class BackendError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def _post(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None,
          stream: bool = False) -> requests.Response:
    try:
        resp = requests.post(url, json=payload, headers=headers, stream=stream,
                             timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except requests.ConnectionError as exc:
        raise BackendError(
            f"Cannot reach the model server at {url}. Is it running? "
            "(run `repoexplain doctor` for help)"
        ) from exc
    except requests.Timeout as exc:
        raise BackendError(f"Timed out waiting for {url}. The model may be too large for this machine.") from exc
    if resp.status_code >= 400:
        raise BackendError(f"{url} returned HTTP {resp.status_code}: {resp.text[:300]}", status=resp.status_code)
    return resp


def _get(url: str, headers: Optional[Dict[str, str]] = None) -> requests.Response:
    try:
        resp = requests.get(url, headers=headers, timeout=(CONNECT_TIMEOUT, 30))
    except requests.RequestException as exc:
        raise BackendError(f"Cannot reach the model server at {url}.") from exc
    if resp.status_code >= 400:
        raise BackendError(f"{url} returned HTTP {resp.status_code}: {resp.text[:300]}", status=resp.status_code)
    return resp


class Provider:
    name = "base"

    def chat_stream(self, model: str, messages: List[Dict[str, str]], options: Dict[str, Any]) -> Iterator[str]:
        raise NotImplementedError

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def unload(self, model: str, kind: str = "chat") -> None:
        """Free the model's RAM if the backend supports it. Best effort."""
        return None

    def list_models(self) -> List[str]:
        raise NotImplementedError


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    @staticmethod
    def _is_missing_model(exc: BackendError) -> bool:
        return exc.status == 404 and "model" in str(exc).lower()

    def _friendly(self, exc: BackendError, model: str) -> BackendError:
        if self._is_missing_model(exc):
            return BackendError(f"Model '{model}' is not installed in Ollama. Run: ollama pull {model}")
        return exc

    def chat_stream(self, model, messages, options):
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_ctx": options.get("num_ctx", 4096),
                "num_predict": options.get("num_predict", 400),
                "temperature": options.get("temperature", 0.1),
            },
        }
        try:
            resp = _post(f"{self.base}/api/chat", payload, stream=True)
        except BackendError as exc:
            raise self._friendly(exc, model)
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "error" in data:
                raise BackendError(str(data["error"]))
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                break

    def embed(self, model, texts):
        try:
            resp = _post(f"{self.base}/api/embed", {"model": model, "input": texts})
            return resp.json()["embeddings"]
        except BackendError as exc:
            if exc.status == 404 and not self._is_missing_model(exc):
                # Older Ollama versions only have the single-text endpoint.
                return [self._legacy_embed(model, t) for t in texts]
            raise self._friendly(exc, model)

    def _legacy_embed(self, model: str, text: str) -> List[float]:
        try:
            resp = _post(f"{self.base}/api/embeddings", {"model": model, "prompt": text})
        except BackendError as exc:
            raise self._friendly(exc, model)
        return resp.json()["embedding"]

    def unload(self, model, kind="chat"):
        try:
            if kind == "embed":
                _post(f"{self.base}/api/embed", {"model": model, "input": ".", "keep_alive": 0})
            else:
                _post(f"{self.base}/api/generate", {"model": model, "keep_alive": 0})
        except BackendError:
            pass  # unloading is only an optimisation

    def list_models(self):
        data = _get(f"{self.base}/api/tags").json()
        return [m.get("name", "") for m in data.get("models", [])]


class OpenAICompatProvider(Provider):
    """Any server exposing /v1/chat/completions and /v1/embeddings."""

    name = "openai_compatible"

    def __init__(self, base_url: str, api_key: str = ""):
        base = base_url.rstrip("/")
        self.base = base if base.endswith("/v1") else base + "/v1"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def chat_stream(self, model, messages, options):
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": options.get("temperature", 0.1),
            "max_tokens": options.get("num_predict", 400),
        }
        resp = _post(f"{self.base}/chat/completions", payload, headers=self.headers, stream=True)
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            choices = json.loads(body).get("choices") or [{}]
            piece = (choices[0].get("delta") or {}).get("content")
            if piece:
                yield piece

    def embed(self, model, texts):
        resp = _post(f"{self.base}/embeddings", {"model": model, "input": texts}, headers=self.headers)
        items = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in items]

    def list_models(self):
        data = _get(f"{self.base}/models", headers=self.headers).json()
        return [m.get("id", "") for m in data.get("data", [])]


def make_provider(section: Dict[str, Any]) -> Provider:
    kind = section.get("provider", "ollama")
    if kind == "ollama":
        return OllamaProvider(section["base_url"])
    if kind in ("openai_compatible", "openai", "lmstudio", "llamacpp"):
        return OpenAICompatProvider(section["base_url"], section.get("api_key", ""))
    raise BackendError(f"Unknown provider '{kind}'. Use 'ollama' or 'openai_compatible'.")

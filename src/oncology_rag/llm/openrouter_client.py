"""OpenRouter client with minimal chat completions support."""

from __future__ import annotations

import json
import os
import time
from http.client import IncompleteRead
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping


@dataclass(frozen=True)
class OpenRouterConfig:
    base_url: str
    api_key: str
    timeout_s: float = 60.0
    retries: int = 3
    headers: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "OpenRouterConfig":
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        return cls(base_url=base_url, api_key=api_key)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OpenRouterConfig":
        api_cfg = raw.get("api", raw)
        base_url = str(api_cfg.get("base_url") or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
        api_key = str(api_cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY", ""))
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        timeout_s = float(api_cfg.get("timeout_s", 60.0))
        retries = int(api_cfg.get("retries", 3))
        headers = api_cfg.get("headers", {}) or {}
        return cls(
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
            retries=retries,
            headers=headers,
        )


def _merge_headers(config: OpenRouterConfig) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    headers.update({str(k): str(v) for k, v in config.headers.items()})
    return headers


_KEY_LIMIT_WAIT_S = 60.0
_RETRYABLE_5XX = {429, 500, 502, 503, 504}


def _is_key_limit_error(exc: urllib.error.HTTPError) -> bool:
    """Return True if the 403 is a recoverable key-limit error (not an auth failure)."""
    if exc.code != 403:
        return False
    try:
        body = json.loads(exc.read().decode("utf-8"))
        msg = (body.get("error") or {}).get("message", "")
        return "limit exceeded" in msg.lower()
    except Exception:
        return False


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_s: float,
    retries: int,
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    server_error_attempts = 0

    while True:
        last_error: Exception | None = None
        retry_outer = False

        for attempt in range(retries + 1):
            try:
                request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
                with urllib.request.urlopen(request, timeout=timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if _is_key_limit_error(exc):
                    retry_outer = True
                    print(
                        f"[openrouter] Key limit exceeded — waiting {_KEY_LIMIT_WAIT_S:.0f}s before retry...",
                        flush=True,
                    )
                    time.sleep(_KEY_LIMIT_WAIT_S)
                    break
                if exc.code in _RETRYABLE_5XX:
                    retry_outer = True
                    wait = min(5.0 * (2 ** server_error_attempts), 60.0)
                    server_error_attempts += 1
                    print(
                        f"[openrouter] Server error {exc.code} — retrying in {wait:.0f}s (attempt {server_error_attempts})...",
                        flush=True,
                    )
                    time.sleep(wait)
                    break
                last_error = exc
            except (urllib.error.URLError, TimeoutError, IncompleteRead, json.JSONDecodeError) as exc:
                last_error = exc

            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))

        if retry_outer:
            continue

        raise RuntimeError(f"OpenRouter request failed: {last_error}") from last_error


def normalize_chat_response(raw: Mapping[str, Any]) -> Dict[str, Any]:
    choice = None
    if isinstance(raw.get("choices"), Iterable):
        choice = next(iter(raw.get("choices") or []), None)
    message = (choice or {}).get("message", {})
    return {
        "text": str(message.get("content") or ""),
        "tool_calls": message.get("tool_calls", []),
        "usage": raw.get("usage", {}),
        "raw": raw,
    }


class OpenRouterClient:
    def __init__(self, config: OpenRouterConfig) -> None:
        self._config = config
        self._headers = _merge_headers(config)

    def chat(
        self,
        *,
        model: str,
        messages: Iterable[Mapping[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": list(messages),
        }
        payload.update(kwargs)
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        try:
            raw = _post_json(
                url=url,
                payload=payload,
                headers=self._headers,
                timeout_s=self._config.timeout_s,
                retries=self._config.retries,
            )
        except RuntimeError as exc:
            # Some providers reject OpenRouter-specific provider routing options.
            # Retry once without provider hints to keep runs alive.
            if "HTTP Error 400" in str(exc) and "provider" in payload:
                fallback_payload = dict(payload)
                fallback_payload.pop("provider", None)
                raw = _post_json(
                    url=url,
                    payload=fallback_payload,
                    headers=self._headers,
                    timeout_s=self._config.timeout_s,
                    retries=self._config.retries,
                )
            else:
                raise
        return normalize_chat_response(raw)

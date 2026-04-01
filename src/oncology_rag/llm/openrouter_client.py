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
_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
# 4xx codes that are never recoverable (bad request, auth failure, not found…)
_FATAL_HTTP_CODES = {400, 401, 403, 404, 405, 422}


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    """Read and return the response body of an HTTPError as a string.

    The stream can only be read once; callers must use the returned string
    instead of reading ``exc`` again.  Returns an empty string on failure.
    """
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _is_key_limit_error(body_text: str) -> bool:
    """Return True if ``body_text`` indicates a recoverable key-limit 403."""
    try:
        body = json.loads(body_text)
        msg = (body.get("error") or {}).get("message", "")
        return "limit exceeded" in msg.lower()
    except Exception:
        return False


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_s: float,
    retries: int,  # NOTE: not used to cap transient retries — loop is intentionally infinite
) -> Dict[str, Any]:
    """POST JSON to url, retrying indefinitely on all transient errors.

    Only raises on fatal 4xx responses (bad request, auth failure, etc.).
    Timeouts, network errors, and 5xx responses retry with exponential backoff
    (capped at 60 s per wait). Key-limit 403s retry every 60 s indefinitely.

    The ``retries`` parameter is accepted for API compatibility but does not cap
    the number of transient retries — this is intentional for long-running
    reasoning models (e.g. qwen-thinking) that can take 35+ minutes per item.
    """
    body = json.dumps(payload).encode("utf-8")
    transient_attempt = 0

    while True:
        try:
            request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))

        except urllib.error.HTTPError as exc:
            if exc.code in _FATAL_HTTP_CODES:
                body_text = _read_http_error_body(exc)
                if exc.code == 403 and _is_key_limit_error(body_text):
                    print(
                        f"[openrouter] Key limit exceeded — waiting {_KEY_LIMIT_WAIT_S:.0f}s before retry...",
                        flush=True,
                    )
                    time.sleep(_KEY_LIMIT_WAIT_S)
                    continue
                detail = f" — body: {body_text}" if body_text else ""
                raise RuntimeError(
                    f"OpenRouter request failed with HTTP {exc.code}{detail}"
                ) from exc

            # 5xx or unexpected code: exponential backoff, indefinite retries
            wait = min(5.0 * (2 ** transient_attempt), 60.0)
            transient_attempt += 1
            print(
                f"[openrouter] HTTP {exc.code} — retrying in {wait:.0f}s (attempt {transient_attempt})...",
                flush=True,
            )
            time.sleep(wait)

        except (urllib.error.URLError, TimeoutError, IncompleteRead, json.JSONDecodeError) as exc:
            wait = min(5.0 * (2 ** transient_attempt), 60.0)
            transient_attempt += 1
            print(
                f"[openrouter] Transient error ({type(exc).__name__}) — retrying in {wait:.0f}s (attempt {transient_attempt})...",
                flush=True,
            )
            time.sleep(wait)


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
            # Some providers reject OpenRouter-specific provider routing options
            # with HTTP 400 (bad request) or HTTP 404 (no endpoints for routing
            # constraints like quantizations). Retry once without provider hints.
            _exc_str = str(exc)
            _is_routing_error = "provider" in payload and (
                "HTTP 400" in _exc_str
                or ("HTTP 404" in _exc_str and "endpoint" in _exc_str.lower())
            )
            if _is_routing_error:
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

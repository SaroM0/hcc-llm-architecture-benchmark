"""Helpers for writing evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_REDACTED = "${OPENROUTER_API_KEY}"
_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
}


def redact_secrets(value: Any) -> Any:
    """Return a copy of value with secret-looking fields redacted."""
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, nested in value.items():
            key_text = str(key).lower()
            if any(secret_key in key_text for secret_key in _SECRET_KEYS):
                redacted[str(key)] = _REDACTED
            else:
                redacted[str(key)] = redact_secrets(nested)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    return value


def write_config_snapshot(output_dir: Path, snapshot: Mapping[str, Any]) -> Path:
    """Write a configuration snapshot as YAML when possible."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config_snapshot.yaml"
    safe_snapshot = redact_secrets(snapshot)
    try:
        import yaml
    except ImportError:
        path.write_text(json.dumps(safe_snapshot, indent=2), encoding="utf-8")
        return path
    path.write_text(yaml.safe_dump(safe_snapshot, sort_keys=False), encoding="utf-8")
    return path

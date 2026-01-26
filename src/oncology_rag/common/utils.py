"""Shared utility helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


def hash_prompt_messages(messages: Iterable[Mapping[str, Any]]) -> str:
    """Create a stable hash for a list of prompt messages."""
    normalized = json.dumps(list(messages), sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"

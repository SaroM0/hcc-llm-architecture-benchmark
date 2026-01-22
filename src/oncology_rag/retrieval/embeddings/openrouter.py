"""OpenRouter embeddings client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class OpenRouterEmbeddingConfig:
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 60.0
    batch_size: int = 64


class OpenRouterEmbeddingModel:
    def __init__(self, config: OpenRouterEmbeddingConfig) -> None:
        self._config = config

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        batch_size = max(1, int(self._config.batch_size))
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        payload = json.dumps({"model": self._config.model, "input": list(texts)}).encode(
            "utf-8"
        )
        url = f"{self._config.base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self._config.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Embedding request failed: {exc}") from exc
        data = raw.get("data", [])
        return [item.get("embedding", []) for item in data]

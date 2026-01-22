"""Embedding model interface."""

from __future__ import annotations

from typing import Protocol, Sequence


class EmbeddingModel(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return embeddings for a batch of texts."""

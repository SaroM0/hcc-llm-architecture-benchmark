"""Vector store interface."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence


class VectorStore(Protocol):
    def upsert(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str] | None = None,
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Insert or update vectors."""

    def query(
        self,
        *,
        embedding: Sequence[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Query vectors and return raw store results."""

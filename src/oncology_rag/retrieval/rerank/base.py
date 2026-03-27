"""Base interface for rerankers."""

from __future__ import annotations

from typing import Protocol, Sequence


class Reranker(Protocol):
    """Reranks a list of documents given a query.

    Returns the indices of the documents sorted by relevance (descending).
    """

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int | None = None,
    ) -> list[int]:
        """Return document indices sorted by descending relevance score.

        Args:
            query: The search query.
            documents: Candidate documents to rerank.
            top_k: If set, return only the top-k indices.

        Returns:
            List of indices into ``documents``, best first.
        """
        ...

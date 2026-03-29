"""BM25 reranker — no extra dependencies required."""

from __future__ import annotations

import math
import re
from typing import Sequence


class BM25Reranker:
    """BM25F-style lexical reranker.

    Two-stage retrieval pattern:
      1. The vector store does a broad semantic first-pass (top_k * fetch_k).
      2. This reranker re-scores those candidates with BM25 and returns the
         best ``top_k`` by lexical relevance.

    BM25 is complementary to cosine-similarity retrieval: the vector store
    finds semantically related chunks; BM25 promotes chunks that share the
    exact terminology of the query (important for medical acronyms, drug names,
    staging codes, etc.).

    Parameters
    ----------
    k1 : float
        Term-frequency saturation parameter (typical range 1.2–2.0).
    b : float
        Length normalisation parameter (0 = no normalisation, 1 = full).
    fetch_k : int
        Multiplier applied to ``top_k`` when fetching from the vector store.
        The reranker re-scores ``top_k * fetch_k`` candidates and keeps the
        best ``top_k``.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        fetch_k: int = 3,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.fetch_k = fetch_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int | None = None,
    ) -> list[int]:
        """Return document indices sorted by descending BM25 score.

        Args:
            query: The search query.
            documents: Candidate documents (already retrieved).
            top_k: Number of top results to return; if None, returns all.

        Returns:
            Indices into ``documents`` ordered best-first.
        """
        if not documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return list(range(min(top_k or len(documents), len(documents))))

        doc_tokens: list[list[str]] = [self._tokenize(d) for d in documents]
        N = len(doc_tokens)
        avgdl = sum(len(d) for d in doc_tokens) / N

        # Document frequency per unique query term
        df: dict[str, int] = {}
        for token in set(query_tokens):
            df[token] = sum(1 for d in doc_tokens if token in d)

        scores: list[float] = []
        for doc in doc_tokens:
            dl = len(doc)
            tf_map: dict[str, int] = {}
            for t in doc:
                tf_map[t] = tf_map.get(t, 0) + 1

            score = 0.0
            for token in query_tokens:
                tf = tf_map.get(token, 0)
                if tf == 0:
                    continue
                idf = math.log(
                    (N - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5) + 1.0
                )
                norm_tf = (tf * (self.k1 + 1.0)) / (
                    tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                )
                score += idf * norm_tf
            scores.append(score)

        ranked = sorted(range(N), key=lambda i: scores[i], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase word tokeniser.

        Keeps all alphanumeric tokens including single-character ones so that
        clinical staging notation (e.g. "T", "N", "M" in TNM staging, "B" in
        "BCLC B") is preserved.  Empty strings produced by the regex are still
        excluded.
        """
        return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t]

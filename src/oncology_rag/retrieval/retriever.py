"""Retriever that uses an embedding model with a vector store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from oncology_rag.retrieval.embeddings import EmbeddingModel
from oncology_rag.retrieval.vectorstores.base import VectorStore


@dataclass(frozen=True)
class RetrievalResult:
    ids: Sequence[str]
    documents: Sequence[str]
    metadatas: Sequence[Mapping[str, Any]]
    distances: Sequence[float]
    raw: Mapping[str, Any]


class Retriever:
    def __init__(
        self,
        *,
        embedding_model: EmbeddingModel,
        store: VectorStore,
        reranker: Any | None = None,
    ) -> None:
        self._embedding_model = embedding_model
        self._store = store
        self._reranker = reranker

    def retrieve(
        self,
        *,
        query: str,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        # When a reranker is present, fetch a wider candidate set first.
        fetch_k = top_k
        if self._reranker is not None:
            fetch_k = top_k * getattr(self._reranker, "fetch_k", 3)

        embeddings = self._embedding_model.embed_texts([query])
        if not embeddings:
            raise RuntimeError("embed_texts returned an empty list for the query")
        embedding = embeddings[0]
        raw = self._store.query(embedding=embedding, top_k=fetch_k, filters=filters)

        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        if self._reranker is not None and documents:
            ranked_indices = self._reranker.rerank(query, documents, top_k=top_k)
            ids = [ids[i] for i in ranked_indices]
            documents = [documents[i] for i in ranked_indices]
            metadatas = [metadatas[i] for i in ranked_indices]
            distances = [distances[i] for i in ranked_indices]

        if len(documents) < top_k:
            print(
                f"[retriever] WARNING: requested top_k={top_k} but store returned "
                f"{len(documents)} document(s). The index may have fewer documents than expected.",
                flush=True,
            )

        return RetrievalResult(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            distances=distances,
            raw=raw,
        )

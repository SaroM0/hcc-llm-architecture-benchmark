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
    def __init__(self, *, embedding_model: EmbeddingModel, store: VectorStore) -> None:
        self._embedding_model = embedding_model
        self._store = store

    def retrieve(
        self,
        *,
        query: str,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        embedding = self._embedding_model.embed_texts([query])[0]
        raw = self._store.query(embedding=embedding, top_k=top_k, filters=filters)
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        return RetrievalResult(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            distances=distances,
            raw=raw,
        )

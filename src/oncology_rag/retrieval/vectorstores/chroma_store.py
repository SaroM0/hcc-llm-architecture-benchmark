"""ChromaDB vector store factory.

This module hides the client mode (persistent or HTTP) behind a single config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence


@dataclass(frozen=True)
class ChromaConfig:
    mode: str
    collection: str
    tenant: str
    database: str
    persistent_path: str
    http_host: str
    http_port: int
    http_ssl: bool
    http_headers: Mapping[str, str]

    @staticmethod
    def _coerce_headers(value: Any) -> Mapping[str, str]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return {str(k): str(v) for k, v in value.items()}
        raise TypeError("http.headers must be a mapping")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ChromaConfig":
        chroma_cfg = raw.get("chroma", raw)
        mode = str(chroma_cfg.get("mode", "persistent"))
        collection = str(chroma_cfg.get("collection", "default"))
        tenant = str(chroma_cfg.get("tenant", "default_tenant"))
        database = str(chroma_cfg.get("database", "default_database"))

        persistent = chroma_cfg.get("persistent", {})
        http = chroma_cfg.get("http", {})

        return cls(
            mode=mode,
            collection=collection,
            tenant=tenant,
            database=database,
            persistent_path=str(persistent.get("path", "data/indexes/chroma")),
            http_host=str(http.get("host", "localhost")),
            http_port=int(http.get("port", 8000)),
            http_ssl=bool(http.get("ssl", False)),
            http_headers=cls._coerce_headers(http.get("headers")),
        )


def create_chroma_client(config: Mapping[str, Any]) -> Any:
    """Create a Chroma client based on config mode."""
    cfg = ChromaConfig.from_mapping(config)
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError("chromadb is required to use ChromaStore") from exc

    if cfg.mode == "persistent":
        return chromadb.PersistentClient(
            path=cfg.persistent_path,
            tenant=cfg.tenant,
            database=cfg.database,
        )
    if cfg.mode == "http":
        return chromadb.HttpClient(
            host=cfg.http_host,
            port=cfg.http_port,
            ssl=cfg.http_ssl,
            headers=dict(cfg.http_headers),
            tenant=cfg.tenant,
            database=cfg.database,
        )
    raise ValueError(f"Unsupported Chroma mode: {cfg.mode}")


def get_or_create_collection(client: Any, config: Mapping[str, Any]) -> Any:
    """Return a collection based on the configured name."""
    cfg = ChromaConfig.from_mapping(config)
    return client.get_or_create_collection(cfg.collection)


class ChromaStore:
    """Thin wrapper that stores the client and collection name."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)
        self._client = create_chroma_client(self._config)
        self._collection = get_or_create_collection(self._client, self._config)

    @property
    def client(self) -> Any:
        return self._client

    @property
    def collection(self) -> Any:
        return self._collection

    @property
    def config(self) -> MutableMapping[str, Any]:
        return dict(self._config)

    def upsert(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str] | None = None,
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=list(ids),
            embeddings=[list(vec) for vec in embeddings],
            documents=list(documents) if documents is not None else None,
            metadatas=[dict(meta) for meta in metadatas] if metadatas is not None else None,
        )

    def query(
        self,
        *,
        embedding: Sequence[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=int(top_k),
            where=dict(filters) if filters else None,
        )

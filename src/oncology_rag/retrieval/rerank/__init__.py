"""Reranking utilities."""

from oncology_rag.retrieval.rerank.base import Reranker
from oncology_rag.retrieval.rerank.bm25 import BM25Reranker

__all__ = ["Reranker", "BM25Reranker"]


def build_reranker(cfg: dict) -> "BM25Reranker | None":
    """Instantiate a reranker from a retrieval config dict.

    Expected config shape (``configs/rag/retrieval.yaml``)::

        rerank:
          enabled: true
          model: bm25          # only supported value for now
          fetch_k: 3           # optional, default 3
          k1: 1.5              # optional BM25 param
          b: 0.75              # optional BM25 param

    Returns ``None`` when reranking is disabled or the model is unknown.
    """
    rerank_cfg = cfg.get("rerank") or {}
    if not rerank_cfg.get("enabled", False):
        return None

    model = str(rerank_cfg.get("model") or "bm25").lower()
    if model == "bm25":
        return BM25Reranker(
            k1=float(rerank_cfg.get("k1", 1.5)),
            b=float(rerank_cfg.get("b", 0.75)),
            fetch_k=int(rerank_cfg.get("fetch_k", 3)),
        )

    raise ValueError(f"Unknown reranker model: {model!r}. Only 'bm25' is supported.")

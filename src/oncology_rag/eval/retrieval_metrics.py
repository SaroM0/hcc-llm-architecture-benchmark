"""Retrieval evaluation metrics for ranked evidence lists."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RetrievalQrels:
    """Gold evidence for one query.

    ``chunk_relevance`` maps chunk ids to integer relevance grades. Binary
    qrels can use grade 1 for every relevant chunk. ``gold_sources`` is kept
    separate because some datasets can identify the correct guideline/source
    without identifying exact chunks.
    """

    chunk_relevance: Mapping[str, int] = field(default_factory=dict)
    gold_sources: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RetrievedEvidence:
    chunk_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    distance: float | None = None

    @property
    def source(self) -> str | None:
        value = self.metadata.get("source") or self.metadata.get("guideline_name")
        return str(value) if value is not None else None

    @property
    def source_values(self) -> frozenset[str]:
        values = set()
        for key in ("source", "guideline_name", "title", "source_path"):
            value = self.metadata.get(key)
            if value is not None and str(value).strip():
                values.add(str(value).strip())
        return frozenset(values)


def _relevance_at(ranked: Sequence[RetrievedEvidence], qrels: RetrievalQrels, k: int) -> list[int]:
    return [
        int(qrels.chunk_relevance.get(item.chunk_id, 0) or 0)
        for item in ranked[: max(0, k)]
    ]


def hit_at_k(ranked: Sequence[RetrievedEvidence], qrels: RetrievalQrels, k: int) -> float | None:
    if not qrels.chunk_relevance:
        return None
    return float(any(score > 0 for score in _relevance_at(ranked, qrels, k)))


def precision_at_k(ranked: Sequence[RetrievedEvidence], qrels: RetrievalQrels, k: int) -> float | None:
    if not qrels.chunk_relevance:
        return None
    if k <= 0:
        return None
    scores = _relevance_at(ranked, qrels, k)
    return sum(1 for score in scores if score > 0) / float(k)


def recall_at_k(ranked: Sequence[RetrievedEvidence], qrels: RetrievalQrels, k: int) -> float | None:
    if not qrels.chunk_relevance:
        return None
    total_relevant = sum(1 for score in qrels.chunk_relevance.values() if score > 0)
    if total_relevant <= 0:
        return None
    found = sum(1 for score in _relevance_at(ranked, qrels, k) if score > 0)
    return found / float(total_relevant)


def mrr_at_k(ranked: Sequence[RetrievedEvidence], qrels: RetrievalQrels, k: int) -> float | None:
    if not qrels.chunk_relevance:
        return None
    for idx, score in enumerate(_relevance_at(ranked, qrels, k), start=1):
        if score > 0:
            return 1.0 / idx
    return 0.0


def _dcg(scores: Sequence[int]) -> float:
    return sum(
        ((2.0**score) - 1.0) / math.log2(idx + 2.0)
        for idx, score in enumerate(scores)
    )


def ndcg_at_k(ranked: Sequence[RetrievedEvidence], qrels: RetrievalQrels, k: int) -> float | None:
    if not qrels.chunk_relevance:
        return None
    observed = _relevance_at(ranked, qrels, k)
    ideal = sorted((int(v) for v in qrels.chunk_relevance.values()), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg <= 0:
        return None
    return _dcg(observed) / ideal_dcg


def evidence_coverage_at_k(
    ranked: Sequence[RetrievedEvidence],
    qrels: RetrievalQrels,
    k: int,
) -> float | None:
    """Coverage of expected sources in the top-k retrieved evidence."""

    if not qrels.gold_sources:
        return None
    retrieved_sources = set()
    for item in ranked[: max(0, k)]:
        retrieved_sources.update(item.source_values)
    return len(retrieved_sources & set(qrels.gold_sources)) / float(len(qrels.gold_sources))


def source_accuracy_at_k(
    ranked: Sequence[RetrievedEvidence],
    qrels: RetrievalQrels,
    k: int,
) -> float | None:
    """Whether the highest-ranked source-bearing item in top-k is a gold source."""

    if not qrels.gold_sources:
        return None
    for item in ranked[: max(0, k)]:
        if item.source_values:
            return float(bool(item.source_values & set(qrels.gold_sources)))
    return 0.0


def evaluate_ranked_evidence(
    ranked: Sequence[RetrievedEvidence],
    qrels: RetrievalQrels,
    *,
    context_relevance: float | None = None,
) -> dict[str, float | None]:
    """Compute the retrieval metric suite requested for one query."""

    return {
        "hit_at_5": hit_at_k(ranked, qrels, 5),
        "recall_at_10": recall_at_k(ranked, qrels, 10),
        "mrr_at_10": mrr_at_k(ranked, qrels, 10),
        "ndcg_at_10": ndcg_at_k(ranked, qrels, 10),
        "precision_at_5": precision_at_k(ranked, qrels, 5),
        "context_relevance": context_relevance,
        "gold_chunk_recall_at_10": recall_at_k(ranked, qrels, 10),
        "evidence_coverage_at_10": evidence_coverage_at_k(ranked, qrels, 10),
        "source_accuracy_at_10": source_accuracy_at_k(ranked, qrels, 10),
    }


def aggregate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Macro-average metric rows, ignoring None values per metric."""

    metric_names = [
        "hit_at_5",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "precision_at_5",
        "context_relevance",
        "gold_chunk_recall_at_10",
        "evidence_coverage_at_10",
        "source_accuracy_at_10",
    ]
    summary: dict[str, Any] = {"num_items": len(rows), "metrics": {}}
    for name in metric_names:
        values = [
            float(row[name])
            for row in rows
            if row.get(name) is not None
        ]
        summary["metrics"][name] = {
            "mean": (sum(values) / len(values)) if values else None,
            "evaluated_items": len(values),
        }
    return summary

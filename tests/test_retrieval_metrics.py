from oncology_rag.eval.retrieval_metrics import (
    RetrievedEvidence,
    RetrievalQrels,
    evaluate_ranked_evidence,
)


def test_retrieval_metrics_with_chunk_and_source_gold() -> None:
    ranked = [
        RetrievedEvidence("c0", "noise", {"source": "wrong.md"}),
        RetrievedEvidence("c1", "relevant", {"source": "guide-a.md", "guideline_name": "AASLD"}),
        RetrievedEvidence("c2", "also relevant", {"source": "guide-b.md"}),
        RetrievedEvidence("c3", "noise", {"source": "guide-c.md"}),
    ]
    qrels = RetrievalQrels(
        chunk_relevance={"c1": 2, "c2": 1, "missing": 1},
        gold_sources=frozenset({"AASLD", "guide-b.md"}),
    )

    metrics = evaluate_ranked_evidence(ranked, qrels, context_relevance=0.75)

    assert metrics["hit_at_5"] == 1.0
    assert metrics["precision_at_5"] == 0.4
    assert metrics["recall_at_10"] == 2 / 3
    assert metrics["gold_chunk_recall_at_10"] == 2 / 3
    assert metrics["mrr_at_10"] == 0.5
    assert metrics["ndcg_at_10"] is not None
    assert 0.0 < metrics["ndcg_at_10"] <= 1.0
    assert metrics["evidence_coverage_at_10"] == 1.0
    assert metrics["source_accuracy_at_10"] == 0.0
    assert metrics["context_relevance"] == 0.75


def test_gold_metrics_are_none_without_qrels() -> None:
    ranked = [RetrievedEvidence("c0", "text", {"source": "guide-a.md"})]
    metrics = evaluate_ranked_evidence(ranked, RetrievalQrels())

    assert metrics["hit_at_5"] is None
    assert metrics["recall_at_10"] is None
    assert metrics["mrr_at_10"] is None
    assert metrics["ndcg_at_10"] is None
    assert metrics["precision_at_5"] is None
    assert metrics["gold_chunk_recall_at_10"] is None
    assert metrics["evidence_coverage_at_10"] is None
    assert metrics["source_accuracy_at_10"] is None

"""Isolated retrieval evaluation over the configured vector database."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping

from oncology_rag.common.types import QAItem
from oncology_rag.eval.retrieval_metrics import (
    RetrievedEvidence,
    RetrievalQrels,
    aggregate_metric_rows,
    evaluate_ranked_evidence,
)
from oncology_rag.llm.openrouter_client import OpenRouterClient, OpenRouterConfig
from oncology_rag.llm.router import ModelRouter
from oncology_rag.retrieval.embeddings import build_embedding_model
from oncology_rag.retrieval.rerank import build_reranker
from oncology_rag.retrieval.retriever import Retriever
from oncology_rag.retrieval.vectorstores.chroma_store import ChromaStore


_METRIC_NAMES = [
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
_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), match.group(0)), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(val) for key, val in value.items()}
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load config files") from exc
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _expand_env(raw)


def _detect_dataset_format(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
            delimiter = "\t" if "\t" in first_line else ","
            handle.seek(0)
            header = next(csv.reader(handle, delimiter=delimiter), [])
        if {"question_id", "hypothesis", "new_information", "validated_answer"}.issubset(set(header)):
            return "sct_validated_csv"
    if path.suffix.lower() == ".json":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "jsonl"
        if isinstance(raw, list) and raw and isinstance(raw[0], Mapping) and "vignette" in raw[0]:
            return "sct"
        if isinstance(raw, Mapping) and ("vignette" in raw or "vignettes" in raw):
            return "sct"
    return "jsonl"


def _load_dataset(path: Path) -> list[QAItem]:
    fmt = _detect_dataset_format(path)
    if fmt == "sct_validated_csv":
        from oncology_rag.eval.sct.loader import load_validated_csv_as_qa_items

        return list(load_validated_csv_as_qa_items(path))
    if fmt == "sct":
        from oncology_rag.eval.sct.loader import load_sct_as_qa_items

        return list(load_sct_as_qa_items(path))

    items: list[QAItem] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        raw = json.loads(line)
        items.append(
            QAItem(
                question_id=str(raw.get("question_id", idx)),
                question=str(raw.get("question", "")),
                metadata=raw.get("metadata", {}) or {},
                rubric_id=raw.get("rubric_id"),
            )
        )
    return items


def _split_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in re.split(r"[|;,]", text) if part.strip()]


def _parse_relevance_map(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): int(v) for k, v in value.items()}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, Mapping):
        return {str(k): int(v) for k, v in parsed.items()}
    return {chunk_id: 1 for chunk_id in _split_list(text)}


def _qrels_from_row(row: Mapping[str, Any]) -> RetrievalQrels:
    relevance: dict[str, int] = {}
    for key in ("chunk_relevance", "gold_chunk_relevance", "qrels"):
        relevance.update(_parse_relevance_map(row.get(key)))
    for key in ("gold_chunk_ids", "relevant_chunk_ids", "gold_chunks"):
        for chunk_id in _split_list(row.get(key)):
            relevance.setdefault(chunk_id, 1)
    sources = set()
    for key in ("gold_sources", "relevant_sources", "source_ids"):
        sources.update(_split_list(row.get(key)))
    return RetrievalQrels(chunk_relevance=relevance, gold_sources=frozenset(sources))


def _load_qrels_from_dataset(path: Path) -> dict[str, RetrievalQrels]:
    if path.suffix.lower() != ".csv":
        return {}
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
        delimiter = "\t" if "\t" in first_line else ","
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames or "question_id" not in reader.fieldnames:
            return {}
        qrels: dict[str, RetrievalQrels] = {}
        for row in reader:
            question_id = str(row.get("question_id") or "").strip()
            if not question_id:
                continue
            parsed = _qrels_from_row(row)
            if parsed.chunk_relevance or parsed.gold_sources:
                qrels[question_id] = parsed
        return qrels


def _load_qrels_file(path: Path | None) -> dict[str, RetrievalQrels]:
    if path is None:
        return {}
    if path.suffix.lower() == ".jsonl":
        qrels: dict[str, RetrievalQrels] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row.get("question_id") or "").strip()
            if question_id:
                qrels[question_id] = _qrels_from_row(row)
        return qrels
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("items", raw) if isinstance(raw, Mapping) else raw
        return {
            str(row["question_id"]): _qrels_from_row(row)
            for row in rows
            if isinstance(row, Mapping) and row.get("question_id")
        }
    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
        delimiter = "\t" if "\t" in first_line else ","
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        return {
            str(row["question_id"]): _qrels_from_row(row)
            for row in reader
            if row.get("question_id")
        }


def _merge_qrels(
    dataset_qrels: Mapping[str, RetrievalQrels],
    file_qrels: Mapping[str, RetrievalQrels],
) -> dict[str, RetrievalQrels]:
    merged = dict(dataset_qrels)
    merged.update(file_qrels)
    return merged


def _build_retriever(
    *,
    embeddings_config_path: Path,
    chroma_config_path: Path,
    retrieval_config_path: Path,
) -> Retriever:
    embeddings_cfg = _load_yaml(embeddings_config_path)
    chroma_cfg = _load_yaml(chroma_config_path)
    retrieval_cfg = _load_yaml(retrieval_config_path)
    return Retriever(
        embedding_model=build_embedding_model(embeddings_cfg),
        store=ChromaStore(chroma_cfg),
        reranker=build_reranker(retrieval_cfg),
    )


def _chat_json(
    client: OpenRouterClient,
    *,
    model_id: str,
    messages: list[dict[str, str]],
    llm_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    response = client.chat(model=model_id, messages=messages, **dict(llm_params or {}))
    text = response.get("text", "")
    fenced = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        obj = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if obj:
            text = obj.group(0)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": response.get("text", "")}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _generate_query_with_llm(
    *,
    client: OpenRouterClient,
    model_id: str,
    item_question: str,
    llm_params: Mapping[str, Any] | None = None,
) -> str:
    parsed = _chat_json(
        client,
        model_id=model_id,
        llm_params=llm_params,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write search queries for a hepatocellular carcinoma guideline "
                    "vector database. Return only JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Rewrite this SCT case into one concise retrieval query. Include the "
                    "clinical decision, key disease state, imaging/staging/treatment terms, "
                    "and the new information. Do not answer the case.\n\n"
                    f"CASE:\n{item_question}\n\n"
                    'Return JSON exactly like: {"query": "..."}'
                ),
            },
        ],
    )
    query = str(parsed.get("query") or "").strip()
    return query or item_question


def _judge_context_relevance(
    *,
    client: OpenRouterClient,
    model_id: str,
    item_question: str,
    evidence: list[RetrievedEvidence],
    llm_params: Mapping[str, Any] | None = None,
) -> tuple[float | None, list[dict[str, Any]]]:
    if not evidence:
        return None, []
    evidence_block = "\n\n".join(
        f"{idx}. id={ev.chunk_id}\nsource={ev.source or 'Unknown'}\ntext={ev.text[:1800]}"
        for idx, ev in enumerate(evidence, start=1)
    )
    parsed = _chat_json(
        client,
        model_id=model_id,
        llm_params=llm_params,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict clinical retrieval judge. Score whether each evidence "
                    "chunk is useful for answering the SCT case from HCC guidelines. "
                    "Return only JSON. Scores: 0 irrelevant, 1 partially relevant, 2 directly relevant."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"SCT CASE:\n{item_question}\n\n"
                    f"RETRIEVED EVIDENCE:\n{evidence_block}\n\n"
                    'Return JSON exactly like: {"scores":[{"rank":1,"score":2,"reason":"..."}, ...]}'
                ),
            },
        ],
    )
    raw_scores = parsed.get("scores", [])
    scored: list[dict[str, Any]] = []
    if isinstance(raw_scores, list):
        for entry in raw_scores:
            if not isinstance(entry, Mapping):
                continue
            try:
                rank = int(entry.get("rank"))
                score = max(0, min(2, int(entry.get("score"))))
            except (TypeError, ValueError):
                continue
            if 1 <= rank <= len(evidence):
                scored.append({
                    "rank": rank,
                    "chunk_id": evidence[rank - 1].chunk_id,
                    "score": score,
                    "reason": str(entry.get("reason") or "")[:500],
                })
    if not scored:
        return None, []
    return sum(item["score"] for item in scored) / (2.0 * len(scored)), scored


def _write_qrels_template(items: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question_id", "gold_chunk_ids", "chunk_relevance", "gold_sources"],
        )
        writer.writeheader()
        for item in items:
            writer.writerow({
                "question_id": item.question_id,
                "gold_chunk_ids": "",
                "chunk_relevance": "",
                "gold_sources": "",
            })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate isolated vector retrieval with gold and LLM-judge metrics."
    )
    parser.add_argument("--dataset", required=True, help="SCT CSV/JSON/JSONL dataset.")
    parser.add_argument("--qrels", default=None, help="Optional qrels CSV/JSON/JSONL file.")
    parser.add_argument("--provider-config", default="configs/providers/openrouter.yaml")
    parser.add_argument("--embeddings-config", default="configs/rag/embeddings.yaml")
    parser.add_argument("--chroma-config", default="configs/rag/chroma.yaml")
    parser.add_argument("--retrieval-config", default="configs/rag/retrieval.yaml")
    parser.add_argument("--output-dir", default="runs/retrieval_eval")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--query-mode",
        choices=["item", "llm"],
        default="llm",
        help="'llm' asks --query-model to write the retrieval query; 'item' uses the full SCT prompt.",
    )
    parser.add_argument("--query-model", default=None, help="Provider model key for LLM query generation.")
    parser.add_argument("--judge-model", default=None, help="Provider model key for Context Relevance judging.")
    parser.add_argument("--llm-params", default=None, help="Optional JSON object passed to query/judge model calls.")
    parser.add_argument(
        "--write-qrels-template",
        default=None,
        help="Write a qrels CSV template for the loaded dataset and exit.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_path = Path(args.dataset)
    items = list(_load_dataset(dataset_path))
    if args.limit is not None:
        items = items[: max(0, int(args.limit))]

    if args.write_qrels_template:
        _write_qrels_template(items, Path(args.write_qrels_template))
        print(f"Wrote qrels template for {len(items)} items to {args.write_qrels_template}")
        return

    if args.query_mode == "llm" and not args.query_model:
        raise ValueError("--query-mode llm requires --query-model")

    llm_params = json.loads(args.llm_params) if args.llm_params else {}
    provider_cfg = _load_yaml(Path(args.provider_config))
    router = ModelRouter(provider_cfg)
    client = OpenRouterClient(OpenRouterConfig.from_mapping(provider_cfg))
    query_model_id = router.registry.get(args.query_model).model_id if args.query_model else None
    judge_model_id = router.registry.get(args.judge_model).model_id if args.judge_model else None

    retriever = _build_retriever(
        embeddings_config_path=Path(args.embeddings_config),
        chroma_config_path=Path(args.chroma_config),
        retrieval_config_path=Path(args.retrieval_config),
    )
    qrels = _merge_qrels(
        _load_qrels_from_dataset(dataset_path),
        _load_qrels_file(Path(args.qrels) if args.qrels else None),
    )

    run_id = time.strftime("%Y%m%d_%H%M%S_retrieval_eval")
    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "retrieval_results.jsonl"

    metric_rows: list[dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(items, start=1):
            print(f"[{index}/{len(items)}] retrieval eval {item.question_id}", flush=True)
            if args.query_mode == "llm":
                query = _generate_query_with_llm(
                    client=client,
                    model_id=str(query_model_id),
                    item_question=item.question,
                    llm_params=llm_params,
                )
            else:
                query = item.question

            result = retriever.retrieve(query=query, top_k=max(int(args.top_k), 10), filters={})
            ranked = [
                RetrievedEvidence(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata or {},
                    distance=distance,
                )
                for chunk_id, text, metadata, distance in zip(
                    result.ids,
                    result.documents,
                    result.metadatas,
                    result.distances,
                    strict=True,
                )
            ]
            context_relevance = None
            relevance_judgments: list[dict[str, Any]] = []
            if judge_model_id:
                context_relevance, relevance_judgments = _judge_context_relevance(
                    client=client,
                    model_id=str(judge_model_id),
                    item_question=item.question,
                    evidence=ranked[:10],
                    llm_params=llm_params,
                )

            item_qrels = qrels.get(item.question_id, RetrievalQrels())
            metrics = evaluate_ranked_evidence(
                ranked,
                item_qrels,
                context_relevance=context_relevance,
            )
            record = {
                "question_id": item.question_id,
                "query_mode": args.query_mode,
                "query": query,
                "has_gold_chunks": bool(item_qrels.chunk_relevance),
                "has_gold_sources": bool(item_qrels.gold_sources),
                "gold_chunk_ids": list(item_qrels.chunk_relevance.keys()),
                "gold_sources": sorted(item_qrels.gold_sources),
                "retrieved": [
                    {
                        "rank": rank,
                        "chunk_id": ev.chunk_id,
                        "source": ev.source,
                        "distance": ev.distance,
                        "metadata": dict(ev.metadata),
                        "text": ev.text,
                        "gold_relevance": item_qrels.chunk_relevance.get(ev.chunk_id, 0),
                    }
                    for rank, ev in enumerate(ranked, start=1)
                ],
                "context_relevance_judgments": relevance_judgments,
                "metrics": metrics,
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            row = {"question_id": item.question_id, **metrics}
            metric_rows.append(row)

    summary = aggregate_metric_rows(metric_rows)
    summary.update({
        "run_id": run_id,
        "dataset": str(dataset_path),
        "qrels": str(args.qrels) if args.qrels else None,
        "query_mode": args.query_mode,
        "query_model": args.query_model,
        "judge_model": args.judge_model,
        "top_k": int(args.top_k),
        "items_with_gold_chunks": sum(1 for row in metric_rows if qrels.get(row["question_id"], RetrievalQrels()).chunk_relevance),
        "items_with_gold_sources": sum(1 for row in metric_rows if qrels.get(row["question_id"], RetrievalQrels()).gold_sources),
        "metric_names": _METRIC_NAMES,
        "notes": [
            "Gold-dependent metrics are null for items without qrels.",
            "Context Relevance is null unless --judge-model is provided and returns parseable scores.",
            "Gold Chunk Recall@10 is exact chunk-id Recall@10.",
            "Evidence Coverage@10 and Source Accuracy@10 require gold_sources.",
        ],
    })
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Wrote retrieval results to {results_path}")
    print(f"Wrote summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

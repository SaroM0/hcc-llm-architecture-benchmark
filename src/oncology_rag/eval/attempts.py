"""Attempt-level evaluation logging utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO

from oncology_rag.common.types import Prediction, QAItem, RunContext
from oncology_rag.eval.sct.metrics import NUMERIC_TO_SCORE, SCORE_MAP, extract_score_from_response


SCHEMA_VERSION = "eval.v1"
SCT_ALLOWED_LABELS = ["+2", "+1", "0", "-1", "-2"]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _extract_gold(item: QAItem) -> dict[str, Any]:
    metadata = item.metadata or {}
    label = metadata.get("gold_label") or metadata.get("label") or metadata.get("expected_answer")
    allowed = metadata.get("allowed_labels") or metadata.get("labels")
    rationale = metadata.get("rationale")
    if allowed is None and label in SCT_ALLOWED_LABELS:
        allowed = SCT_ALLOWED_LABELS
    return {
        "label": _coerce_str(label),
        "allowed_labels": list(allowed) if isinstance(allowed, Iterable) and not isinstance(allowed, (str, bytes)) else allowed,
        "rationale": _coerce_str(rationale),
    }


def _extract_pred_label(prediction: Prediction, item: QAItem) -> str | None:
    structured = prediction.structured or {}
    for key in ("pred_label", "label", "answer", "prediction"):
        if key in structured:
            return _coerce_str(structured.get(key))
    expected_answer = item.metadata.get("expected_answer")
    if expected_answer in SCORE_MAP:
        predicted_score = extract_score_from_response(prediction.answer_text)
        if predicted_score is not None:
            return NUMERIC_TO_SCORE.get(predicted_score, str(predicted_score))
    return None


def _extract_confidence(prediction: Prediction) -> float | None:
    structured = prediction.structured or {}
    confidence = structured.get("confidence")
    if confidence is None:
        return None
    try:
        return float(confidence)
    except (TypeError, ValueError):
        return None


def _aggregate_usage(events: Iterable[Any]) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    total_cost_usd = 0.0
    latency_ms = 0.0
    provider_errors: list[str] = []

    for event in events:
        data = event.to_dict()
        if data.get("event_type") != "llm_call":
            continue
        usage = data.get("usage", {}) or {}
        prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        total_tokens += int(
            usage.get("total_tokens", (usage.get("prompt_tokens", 0) or 0) + (usage.get("completion_tokens", 0) or 0)) or 0
        )
        cost = data.get("cost_usd")
        if cost is None:
            cost = usage.get("cost", 0.0) or 0.0
        total_cost_usd += float(cost or 0.0)
        latency_ms += float(data.get("latency_ms") or 0.0)
        if data.get("error"):
            provider_errors.append(str(data.get("error")))

    return {
        "prompt_tokens": prompt_tokens or None,
        "completion_tokens": completion_tokens or None,
        "total_tokens": total_tokens or None,
        "cost_usd": total_cost_usd if total_cost_usd > 0 else None,
        "latency_ms": latency_ms if latency_ms > 0 else None,
        "provider_error": provider_errors[0] if provider_errors else None,
    }


def _score_attempt(item: QAItem, prediction: Prediction) -> dict[str, Any]:
    expected_answer = item.metadata.get("expected_answer")
    predicted_score = extract_score_from_response(prediction.answer_text)
    gold = _extract_gold(item)
    pred_label = _extract_pred_label(prediction, item)

    is_correct: int | None = None
    is_partial: int | None = None
    parse_error = False

    if expected_answer in SCORE_MAP:
        expected_score = SCORE_MAP.get(expected_answer)
        if predicted_score is None:
            parse_error = True
        else:
            is_correct = int(predicted_score == expected_score)
            is_partial = int(abs(predicted_score - expected_score) <= 1)
        if pred_label is None and predicted_score is not None:
            pred_label = NUMERIC_TO_SCORE.get(predicted_score)
    elif gold.get("label") and pred_label:
        is_correct = int(pred_label == gold.get("label"))
        is_partial = is_correct
    elif gold.get("label") and pred_label is None:
        parse_error = True

    return {
        "is_correct": is_correct,
        "is_partial": is_partial,
        "pred_label": pred_label,
        "confidence": _extract_confidence(prediction),
        "error_type": None,
        "parse_error": parse_error,
    }


def build_attempt_record(
    *,
    context: RunContext,
    item: QAItem,
    prediction: Prediction,
    events: Iterable[Any],
    model_key: str,
    model_id: str,
    model_class: str | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    usage = _aggregate_usage(events)
    scoring = _score_attempt(item, prediction)
    prompt_hash = None
    if prediction.debug:
        prompt_hash = prediction.debug.get("prompt_hash")

    errors = {
        "provider_error": usage.get("provider_error"),
        "timeout": bool(usage.get("provider_error") and "timeout" in str(usage.get("provider_error")).lower()),
        "parse_error": bool(scoring.get("parse_error")),
    }

    return {
        "schema_version": schema_version,
        "timestamp_utc": _utc_timestamp(),
        "experiment_id": context.experiment_id,
        "run_id": context.run_id,
        "model_key": model_key,
        "model_id": model_id,
        "model_class": model_class,
        "question_id": item.question_id,
        "prompt_hash": prompt_hash,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "latency_ms": usage.get("latency_ms"),
        "cost_usd": usage.get("cost_usd"),
        "raw_answer": prediction.answer_text,
        "gold": _extract_gold(item),
        "scoring": {
            "is_correct": scoring.get("is_correct"),
            "is_partial": scoring.get("is_partial"),
            "pred_label": scoring.get("pred_label"),
            "confidence": scoring.get("confidence"),
            "error_type": scoring.get("error_type"),
        },
        "errors": errors,
    }


class AttemptLogger:
    """Append attempt records to versioned JSONL files."""

    def __init__(self, results_root: Path) -> None:
        self._results_root = results_root
        self._handles: dict[tuple[str, str, str], TextIO] = {}

    def _open_handle(self, arm_id: str, run_id: str, model_key: str) -> TextIO:
        key = (arm_id, run_id, model_key)
        if key in self._handles:
            return self._handles[key]
        dir_path = self._results_root / f"arm={arm_id}" / f"run={run_id}"
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / f"model={model_key}.jsonl"
        handle = path.open("a", encoding="utf-8")
        self._handles[key] = handle
        return handle

    def write_attempt(
        self,
        *,
        arm_id: str,
        run_id: str,
        model_key: str,
        record: Mapping[str, Any],
    ) -> None:
        handle = self._open_handle(arm_id, run_id, model_key)
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

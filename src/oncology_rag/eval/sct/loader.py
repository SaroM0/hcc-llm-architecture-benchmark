"""Loader for SCT datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from oncology_rag.common.types import QAItem
from oncology_rag.eval.sct.metrics import NUMERIC_TO_SCORE
from oncology_rag.eval.sct.types import SCTItem, SCTQuestion, SCTVignette


def _normalize_expected_answer(value: Any) -> str | None:
    """Normalize expected answers to the canonical SCT string format."""
    if value is None:
        return None
    if isinstance(value, int):
        return NUMERIC_TO_SCORE.get(value)
    if isinstance(value, float):
        if value.is_integer():
            return NUMERIC_TO_SCORE.get(int(value))
        return None
    if isinstance(value, str):
        text = value.strip()
        if text in {"+2", "+1", "0", "-1", "-2"}:
            return text
        try:
            num = int(text)
        except ValueError:
            return None
        return NUMERIC_TO_SCORE.get(num)
    return None


def _parse_question(raw: Mapping[str, Any], index: int) -> SCTQuestion:
    """Parse a single SCT question from raw JSON."""
    return SCTQuestion(
        question_type=raw.get("question_type", "diagnosis"),
        hypothesis=str(raw.get("hypothesis", "")),
        new_information=str(raw.get("new_information", "")),
        effect_phrase=str(raw.get("effect_phrase", "this hypothesis becomes")),
        options=list(raw.get("options", ["+2", "+1", "0", "-1", "-2"])),
        author_notes=str(raw.get("author_notes", "")),
        expected_answer=_normalize_expected_answer(raw.get("expected_answer")),
    )


def _parse_vignette(raw: Mapping[str, Any], vignette_id: str) -> SCTVignette:
    """Parse a single SCT vignette from raw JSON."""
    questions = [
        _parse_question(q, i)
        for i, q in enumerate(raw.get("questions", []))
    ]
    return SCTVignette(
        vignette_id=vignette_id,
        domain=str(raw.get("domain", "HCC")),
        guideline=str(raw.get("guideline", "unknown")),
        vignette=str(raw.get("vignette", "")),
        questions=questions,
        metadata={k: v for k, v in raw.items() if k not in {"domain", "guideline", "vignette", "questions"}},
    )


def load_sct_dataset(path: Path) -> list[SCTVignette]:
    """Load SCT vignettes from a JSON file.

    Supports two formats:
    1. JSON array of vignettes
    2. JSON object with "vignettes" key containing array

    Args:
        path: Path to the JSON file.

    Returns:
        List of SCTVignette objects.
    """
    content = path.read_text(encoding="utf-8")
    data = json.loads(content)

    if isinstance(data, list):
        vignettes_raw = data
    elif isinstance(data, dict):
        vignettes_raw = data.get("vignettes", data.get("items", [data]))
    else:
        raise ValueError(f"Unexpected JSON structure in {path}")

    vignettes = []
    for i, raw in enumerate(vignettes_raw):
        vignette_id = str(raw.get("id", raw.get("vignette_id", f"v_{i:04d}")))
        vignettes.append(_parse_vignette(raw, vignette_id))

    return vignettes


def expand_vignettes_to_items(vignettes: list[SCTVignette]) -> list[SCTItem]:
    """Expand vignettes into individual SCT items.

    Each vignette may contain multiple questions. This function
    creates one SCTItem per question.

    Args:
        vignettes: List of SCTVignette objects.

    Returns:
        List of SCTItem objects (one per question).
    """
    items = []
    for vignette in vignettes:
        for q_idx, question in enumerate(vignette.questions):
            item_id = f"{vignette.vignette_id}_q{q_idx:02d}"
            items.append(
                SCTItem(
                    item_id=item_id,
                    vignette_id=vignette.vignette_id,
                    domain=vignette.domain,
                    guideline=vignette.guideline,
                    vignette_text=vignette.vignette,
                    question=question,
                    question_index=q_idx,
                )
            )
    return items


def sct_to_qa_items(sct_items: list[SCTItem]) -> Iterable[QAItem]:
    """Convert SCT items to QAItem format for the runner.

    This adapter allows SCT items to be processed by the existing
    arm infrastructure.

    Args:
        sct_items: List of SCT items.

    Yields:
        QAItem objects compatible with the runner.
    """
    for item in sct_items:
        yield QAItem(
            question_id=item.item_id,
            question=item.format_full_prompt(),
            metadata={
                "vignette_id": item.vignette_id,
                "domain": item.domain,
                "guideline": item.guideline,
                "question_type": item.question.question_type,
                "hypothesis": item.question.hypothesis,
                "new_information": item.question.new_information,
                "expected_answer": item.expected_answer,
                "author_notes": item.question.author_notes,
            },
            rubric_id=item.question.question_type,
        )


def load_sct_as_qa_items(path: Path) -> list[QAItem]:
    """Load SCT dataset and convert directly to QAItems.

    Convenience function that combines loading and conversion.

    Args:
        path: Path to the SCT JSON file.

    Returns:
        List of QAItem objects ready for evaluation.
    """
    vignettes = load_sct_dataset(path)
    sct_items = expand_vignettes_to_items(vignettes)
    return list(sct_to_qa_items(sct_items))


def load_validated_csv_as_qa_items(path: Path) -> list[QAItem]:
    """Load validated SCT ground truth CSV and convert to QAItems."""
    items: list[SCTItem] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            question_id = (row.get("question_id") or f"q_{idx:05d}").strip()
            vignette_id = (row.get("vignette_id") or "v_0000").strip()
            vignette_text = row.get("vignette", "") or ""
            question_type = (row.get("question_type") or "diagnosis").strip().lower()
            hypothesis = row.get("hypothesis", "") or ""
            new_information = row.get("new_information", "") or ""
            validated_answer = _normalize_expected_answer(row.get("validated_answer"))

            question_index = 0
            if "_q" in question_id:
                try:
                    question_index = int(question_id.split("_q", 1)[1])
                except ValueError:
                    question_index = 0

            question = SCTQuestion(
                question_type=question_type,
                hypothesis=hypothesis,
                new_information=new_information,
                effect_phrase="this hypothesis becomes",
                expected_answer=validated_answer,
            )

            items.append(
                SCTItem(
                    item_id=question_id,
                    vignette_id=vignette_id,
                    domain="HCC",
                    guideline="validated_ground_truth",
                    vignette_text=vignette_text,
                    question=question,
                    question_index=question_index,
                )
            )

    return list(sct_to_qa_items(items))

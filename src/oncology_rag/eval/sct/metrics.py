"""Metrics and scoring for SCT evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping


# Map string scores to numeric values
SCORE_MAP: dict[str, int] = {
    "+2": 2,
    "+1": 1,
    "0": 0,
    "-1": -1,
    "-2": -2,
    "2": 2,
    "1": 1,
    "-": 0,  # Sometimes models output just "-" for 0
}

# Reverse map for display
NUMERIC_TO_SCORE: dict[int, str] = {
    2: "+2",
    1: "+1",
    0: "0",
    -1: "-1",
    -2: "-2",
}


def extract_score_from_response(response: str) -> int | None:
    """Extract the numeric score from a model response.

    Looks for patterns like "+2", "-1", "0", etc. at the start
    of the response or after common prefixes.

    Args:
        response: The model's response text.

    Returns:
        Integer score (-2 to +2) or None if not found.
    """
    if not response:
        return None

    # Clean and normalize
    text = response.strip()

    # Try to find score at the very beginning
    patterns = [
        r"^([+-]?[0-2])\b",  # Score at start: "+2", "-1", "0"
        r"^\*\*([+-]?[0-2])\*\*",  # Bold markdown: **+2**
        r"^Score:\s*([+-]?[0-2])\b",  # "Score: +2"
        r"^Answer:\s*([+-]?[0-2])\b",  # "Answer: +1"
        r"^My (?:answer|score) is:?\s*([+-]?[0-2])\b",  # "My answer is: 0"
        r"^(?:I would (?:choose|select|rate|give))\s*([+-]?[0-2])\b",  # "I would choose +1"
        r"\b([+-]?[0-2])\s*[:\-–]\s*(?:much|somewhat)",  # "+2: Much more likely"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            score_str = match.group(1)
            if score_str in SCORE_MAP:
                return SCORE_MAP[score_str]
            try:
                val = int(score_str)
                if -2 <= val <= 2:
                    return val
            except ValueError:
                continue

    # Fallback: look for any score-like pattern in first 100 chars
    first_chunk = text[:100]
    match = re.search(r"([+-]?[0-2])", first_chunk)
    if match:
        try:
            val = int(match.group(1))
            if -2 <= val <= 2:
                return val
        except ValueError:
            pass

    return None


@dataclass
class SCTScoreResult:
    """Result of scoring a single SCT item."""

    item_id: str
    predicted_score: int | None
    expected_score: int | None
    is_exact_match: bool
    distance: int | None  # Absolute difference
    raw_response: str
    parsed_successfully: bool


@dataclass
class SCTMetrics:
    """Aggregated metrics for SCT evaluation."""

    total_items: int
    parsed_items: int
    exact_matches: int
    within_one: int  # Score within 1 point
    mean_absolute_error: float | None
    accuracy: float  # Exact match rate
    partial_accuracy: float  # Within 1 point rate
    parse_rate: float  # Successfully parsed responses

    # Breakdown by question type
    by_question_type: dict[str, dict[str, float]] = field(default_factory=dict)

    # Confusion-like analysis
    score_distribution: dict[str, int] = field(default_factory=dict)


class SCTScorer:
    """Scorer for SCT evaluation results."""

    def __init__(self) -> None:
        self._results: list[SCTScoreResult] = []

    def score_item(
        self,
        item_id: str,
        response: str,
        expected_answer: str | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SCTScoreResult:
        """Score a single SCT item response.

        Args:
            item_id: Unique identifier for the item.
            response: Model's response text.
            expected_answer: Expected answer string (e.g., "+1").
            metadata: Optional metadata for the item.

        Returns:
            SCTScoreResult with scoring details.
        """
        predicted = extract_score_from_response(response)
        expected = None
        if expected_answer and expected_answer in SCORE_MAP:
            expected = SCORE_MAP[expected_answer]

        parsed_ok = predicted is not None
        is_exact = False
        distance = None

        if predicted is not None and expected is not None:
            is_exact = predicted == expected
            distance = abs(predicted - expected)

        result = SCTScoreResult(
            item_id=item_id,
            predicted_score=predicted,
            expected_score=expected,
            is_exact_match=is_exact,
            distance=distance,
            raw_response=response[:500],  # Truncate for storage
            parsed_successfully=parsed_ok,
        )

        self._results.append(result)
        return result

    def compute_metrics(self) -> SCTMetrics:
        """Compute aggregated metrics from all scored items."""
        if not self._results:
            return SCTMetrics(
                total_items=0,
                parsed_items=0,
                exact_matches=0,
                within_one=0,
                mean_absolute_error=None,
                accuracy=0.0,
                partial_accuracy=0.0,
                parse_rate=0.0,
            )

        total = len(self._results)
        parsed = sum(1 for r in self._results if r.parsed_successfully)
        exact = sum(1 for r in self._results if r.is_exact_match)
        within_one = sum(
            1 for r in self._results
            if r.distance is not None and r.distance <= 1
        )

        # MAE only for items with both predicted and expected
        scorable = [r for r in self._results if r.distance is not None]
        mae = None
        if scorable:
            mae = sum(r.distance for r in scorable) / len(scorable)

        # Score distribution
        score_dist: dict[str, int] = {}
        for r in self._results:
            if r.predicted_score is not None:
                key = NUMERIC_TO_SCORE.get(r.predicted_score, str(r.predicted_score))
                score_dist[key] = score_dist.get(key, 0) + 1

        return SCTMetrics(
            total_items=total,
            parsed_items=parsed,
            exact_matches=exact,
            within_one=within_one,
            mean_absolute_error=mae,
            accuracy=exact / total if total > 0 else 0.0,
            partial_accuracy=within_one / total if total > 0 else 0.0,
            parse_rate=parsed / total if total > 0 else 0.0,
            score_distribution=score_dist,
        )

    def clear(self) -> None:
        """Clear all results."""
        self._results = []

    @property
    def results(self) -> list[SCTScoreResult]:
        """Get all scoring results."""
        return list(self._results)


def calculate_sct_metrics(
    predictions: list[Mapping[str, Any]],
    expected_key: str = "expected_answer",
) -> SCTMetrics:
    """Calculate SCT metrics from a list of predictions.

    Convenience function for batch evaluation.

    Args:
        predictions: List of prediction dicts with answer_text and metadata.
        expected_key: Key in metadata containing expected answer.

    Returns:
        SCTMetrics with aggregated results.
    """
    scorer = SCTScorer()

    for pred in predictions:
        item_id = pred.get("question_id", "unknown")
        response = pred.get("answer_text", "")
        metadata = pred.get("metadata", {}) or {}

        # Try to get expected answer from debug or metadata
        debug = pred.get("debug", {}) or {}
        expected = (
            metadata.get(expected_key)
            or debug.get(expected_key)
            or pred.get(expected_key)
        )

        scorer.score_item(item_id, response, expected, metadata)

    return scorer.compute_metrics()

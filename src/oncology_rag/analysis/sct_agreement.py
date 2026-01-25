"""SCT expert response agreement analysis.

This module compares expert responses collected via SCT assessments with
model predictions from evaluation runs. It calculates agreement metrics
and generates detailed manifests for analysis.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ExpertResponse:
    """A single expert response to an SCT item."""

    doctor_name: str
    doctor_email: str
    item_set: str
    row_index: int
    domain: str
    question_type: str
    vignette: str
    hypothesis: str
    new_information: str
    response_value: int
    responded_at: str

    @property
    def response_label(self) -> str:
        """Return human-readable label for response value."""
        labels = {
            -2: "Much less likely/appropriate",
            -1: "Less likely/appropriate",
            0: "Neither more nor less likely/appropriate",
            1: "More likely/appropriate",
            2: "Much more likely/appropriate",
        }
        return labels.get(self.response_value, "Unknown")


@dataclass
class MatchedItem:
    """A matched pair of expert response and dataset item."""

    question_id: str
    vignette_id: str
    question_type: str
    hypothesis: str
    new_information: str
    expert_response: int
    expected_answer: int | None
    expert_name: str
    expert_email: str
    row_index: int


@dataclass
class AgreementResult:
    """Agreement analysis for a single expert response."""

    question_id: str
    question_type: str
    expert_response: int
    expected_answer: int | None
    model_prediction: int | None
    expert_agrees_with_expected: bool | None
    expert_agrees_with_model: bool | None
    model_agrees_with_expected: bool | None
    expert_distance_from_expected: int | None
    model_distance_from_expected: int | None
    expert_name: str


@dataclass
class AgreementMetrics:
    """Aggregated agreement metrics."""

    total_responses: int
    matched_responses: int
    unmatched_responses: int
    unique_questions: int
    questions_with_multiple_experts: int

    # Expert vs Expected (gold standard)
    expert_exact_agreement: float
    expert_partial_agreement: float  # Within 1 point
    expert_mean_absolute_error: float

    # Model vs Expected (gold standard)
    model_exact_agreement: float
    model_partial_agreement: float
    model_mean_absolute_error: float

    # Expert vs Model
    expert_model_exact_agreement: float
    expert_model_partial_agreement: float
    expert_model_mean_absolute_error: float

    # Inter-expert agreement (when multiple experts answer same question)
    inter_expert_exact_agreement: float
    inter_expert_partial_agreement: float
    inter_expert_mean_absolute_error: float

    # Breakdown by question type
    by_question_type: dict[str, dict[str, float]] = field(default_factory=dict)

    # Breakdown by expert
    by_expert: dict[str, dict[str, float]] = field(default_factory=dict)

    # Questions where experts disagreed
    expert_disagreements: list[dict] = field(default_factory=list)


def _normalize_text(text: str) -> str:
    """Normalize text for comparison by removing extra whitespace and lowercasing."""
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def _parse_response_value(value: str) -> int | None:
    """Parse response value from string, handling various formats."""
    if not value:
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _parse_int(value: str, default: int = 0) -> int:
    """Safely parse an integer from string."""
    if not value or not value.strip():
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def load_expert_responses(csv_path: Path) -> list[ExpertResponse]:
    """Load expert responses from a CSV file.

    Args:
        csv_path: Path to the CSV file containing expert responses.

    Returns:
        List of ExpertResponse objects.
    """
    responses = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            response_value = _parse_response_value(row.get("Response Value", ""))
            if response_value is None:
                continue  # Skip rows without valid response

            responses.append(
                ExpertResponse(
                    doctor_name=row.get("Doctor Name", ""),
                    doctor_email=row.get("Doctor Email", ""),
                    item_set=row.get("Item Set", ""),
                    row_index=_parse_int(row.get("Row Index", ""), 0),
                    domain=row.get("Domain", ""),
                    question_type=row.get("Question Type", ""),
                    vignette=row.get("Vignette", ""),
                    hypothesis=row.get("Hypothesis", ""),
                    new_information=row.get("New Information", ""),
                    response_value=response_value,
                    responded_at=row.get("Responded At", ""),
                )
            )
    return responses


def load_dataset_items(dataset_path: Path) -> dict[str, dict[str, Any]]:
    """Load dataset items and create a lookup index.

    Args:
        dataset_path: Path to the SCT dataset JSON file.

    Returns:
        Dictionary mapping (normalized_key) -> item data.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items_by_key: dict[str, dict[str, Any]] = {}
    vignettes = data if isinstance(data, list) else data.get("vignettes", [data])

    for vignette in vignettes:
        vignette_id = vignette.get("id", "")
        vignette_text = vignette.get("vignette", "")
        questions = vignette.get("questions", [])

        for q_idx, question in enumerate(questions):
            question_id = f"{vignette_id}_q{q_idx:02d}"
            hypothesis = question.get("hypothesis", "")
            new_info = question.get("new_information", "")

            # Create normalized key for matching
            key = _normalize_text(f"{vignette_text}|{hypothesis}|{new_info}")

            items_by_key[key] = {
                "question_id": question_id,
                "vignette_id": vignette_id,
                "question_type": question.get("question_type", ""),
                "hypothesis": hypothesis,
                "new_information": new_info,
                "expected_answer": question.get("expected_answer"),
                "vignette_text": vignette_text,
            }

    return items_by_key


def load_model_predictions(
    runs_dir: Path,
    experiment_filter: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load model predictions from evaluation runs.

    Args:
        runs_dir: Path to the runs directory.
        experiment_filter: Optional filter to select specific experiment
            (e.g., "A1" for arm A1, or "A1_llama" for specific model).

    Returns:
        Dictionary mapping question_id -> prediction data.
    """
    predictions: dict[str, dict[str, Any]] = {}

    # Search in matrix runs (latest per experiment)
    matrix_dir = runs_dir / "matrix"
    if not matrix_dir.exists():
        return predictions

    # Find latest run for each experiment
    latest_runs: dict[str, tuple[float, Path]] = {}
    for run_dir in matrix_dir.iterdir():
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            exp_id = manifest.get("experiment_id", run_dir.name)

            # Apply experiment filter if specified
            if experiment_filter and experiment_filter not in exp_id:
                continue

            mtime = run_dir.stat().st_mtime
            if exp_id not in latest_runs or mtime > latest_runs[exp_id][0]:
                latest_runs[exp_id] = (mtime, run_dir)
        except (json.JSONDecodeError, OSError):
            continue

    # Load predictions from latest runs, preferring non-None answers
    for exp_id, (_, run_dir) in latest_runs.items():
        predictions_path = run_dir / "predictions.jsonl"
        if not predictions_path.exists():
            continue

        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                pred = json.loads(line)
                question_id = pred.get("question_id")
                if not question_id:
                    continue

                # Store prediction, preferring ones with valid answers
                existing = predictions.get(question_id)
                pred_answer = pred.get("predicted_answer")

                if existing is None:
                    predictions[question_id] = pred
                elif pred_answer is not None and existing.get("predicted_answer") is None:
                    # Replace None answer with valid answer
                    predictions[question_id] = pred
            except json.JSONDecodeError:
                continue

    return predictions


def match_expert_responses(
    responses: list[ExpertResponse],
    dataset_items: dict[str, dict[str, Any]],
) -> tuple[list[MatchedItem], list[ExpertResponse]]:
    """Match expert responses to dataset items.

    Args:
        responses: List of expert responses.
        dataset_items: Dictionary of dataset items indexed by normalized key.

    Returns:
        Tuple of (matched items, unmatched responses).
    """
    matched = []
    unmatched = []

    for response in responses:
        # Create normalized key for matching
        key = _normalize_text(
            f"{response.vignette}|{response.hypothesis}|{response.new_information}"
        )

        item = dataset_items.get(key)
        if item:
            matched.append(
                MatchedItem(
                    question_id=item["question_id"],
                    vignette_id=item["vignette_id"],
                    question_type=item["question_type"],
                    hypothesis=item["hypothesis"],
                    new_information=item["new_information"],
                    expert_response=response.response_value,
                    expected_answer=item["expected_answer"],
                    expert_name=response.doctor_name,
                    expert_email=response.doctor_email,
                    row_index=response.row_index,
                )
            )
        else:
            unmatched.append(response)

    return matched, unmatched


def _parse_predicted_answer(pred: dict[str, Any]) -> int | None:
    """Parse predicted answer from prediction record."""
    answer = pred.get("predicted_answer")
    if answer is None:
        return None
    if isinstance(answer, str):
        # Handle formats like "+2", "-1", "0"
        answer = answer.strip().lstrip("+")
        try:
            return int(answer)
        except ValueError:
            return None
    return int(answer)


def calculate_agreement(
    matched_items: list[MatchedItem],
    predictions: dict[str, dict[str, Any]],
) -> tuple[list[AgreementResult], AgreementMetrics]:
    """Calculate agreement metrics between experts, expected answers, and models.

    Args:
        matched_items: List of matched expert responses.
        predictions: Dictionary of model predictions.

    Returns:
        Tuple of (individual results, aggregated metrics).
    """
    results = []

    # Counters for metrics
    expert_exact = 0
    expert_partial = 0
    expert_total_distance = 0
    expert_valid = 0

    model_exact = 0
    model_partial = 0
    model_total_distance = 0
    model_valid = 0

    expert_model_exact = 0
    expert_model_partial = 0
    expert_model_total_distance = 0
    expert_model_valid = 0

    by_question_type: dict[str, dict[str, list]] = {}
    by_expert: dict[str, dict[str, list]] = {}

    # Group responses by question_id to find multi-expert questions
    responses_by_question: dict[str, list[MatchedItem]] = {}
    for item in matched_items:
        if item.question_id not in responses_by_question:
            responses_by_question[item.question_id] = []
        responses_by_question[item.question_id].append(item)

    for item in matched_items:
        pred = predictions.get(item.question_id, {})
        model_answer = _parse_predicted_answer(pred)

        expected = item.expected_answer
        if isinstance(expected, str):
            expected = int(expected.strip().lstrip("+"))

        # Calculate agreements
        expert_agrees_expected = None
        expert_distance = None
        if expected is not None:
            expert_agrees_expected = item.expert_response == expected
            expert_distance = abs(item.expert_response - expected)
            expert_valid += 1
            expert_total_distance += expert_distance
            if expert_agrees_expected:
                expert_exact += 1
            if expert_distance <= 1:
                expert_partial += 1

        model_agrees_expected = None
        model_distance = None
        if model_answer is not None and expected is not None:
            model_agrees_expected = model_answer == expected
            model_distance = abs(model_answer - expected)
            model_valid += 1
            model_total_distance += model_distance
            if model_agrees_expected:
                model_exact += 1
            if model_distance <= 1:
                model_partial += 1

        expert_agrees_model = None
        if model_answer is not None:
            expert_agrees_model = item.expert_response == model_answer
            em_distance = abs(item.expert_response - model_answer)
            expert_model_valid += 1
            expert_model_total_distance += em_distance
            if expert_agrees_model:
                expert_model_exact += 1
            if em_distance <= 1:
                expert_model_partial += 1

        result = AgreementResult(
            question_id=item.question_id,
            question_type=item.question_type,
            expert_response=item.expert_response,
            expected_answer=expected,
            model_prediction=model_answer,
            expert_agrees_with_expected=expert_agrees_expected,
            expert_agrees_with_model=expert_agrees_model,
            model_agrees_with_expected=model_agrees_expected,
            expert_distance_from_expected=expert_distance,
            model_distance_from_expected=model_distance,
            expert_name=item.expert_name,
        )
        results.append(result)

        # Track by question type
        qtype = item.question_type or "unknown"
        if qtype not in by_question_type:
            by_question_type[qtype] = {
                "expert_exact": [],
                "expert_distance": [],
                "model_exact": [],
                "model_distance": [],
            }
        if expert_agrees_expected is not None:
            by_question_type[qtype]["expert_exact"].append(1 if expert_agrees_expected else 0)
            by_question_type[qtype]["expert_distance"].append(expert_distance)
        if model_agrees_expected is not None:
            by_question_type[qtype]["model_exact"].append(1 if model_agrees_expected else 0)
            by_question_type[qtype]["model_distance"].append(model_distance)

        # Track by expert
        expert = item.expert_name or "unknown"
        if expert not in by_expert:
            by_expert[expert] = {
                "expert_exact": [],
                "expert_distance": [],
                "model_exact": [],
            }
        if expert_agrees_expected is not None:
            by_expert[expert]["expert_exact"].append(1 if expert_agrees_expected else 0)
            by_expert[expert]["expert_distance"].append(expert_distance)

    # Calculate inter-expert agreement for questions with multiple experts
    inter_expert_exact = 0
    inter_expert_partial = 0
    inter_expert_total_distance = 0
    inter_expert_count = 0
    expert_disagreements = []

    multi_expert_questions = {
        qid: items for qid, items in responses_by_question.items() if len(items) > 1
    }

    for qid, items in multi_expert_questions.items():
        # Compare all pairs of experts
        responses = [item.expert_response for item in items]
        experts = [item.expert_name for item in items]

        # For simplicity, compare first two experts (typical case)
        if len(responses) >= 2:
            r1, r2 = responses[0], responses[1]
            e1, e2 = experts[0], experts[1]
            distance = abs(r1 - r2)
            inter_expert_count += 1
            inter_expert_total_distance += distance

            if r1 == r2:
                inter_expert_exact += 1
            if distance <= 1:
                inter_expert_partial += 1
            else:
                # Track disagreements
                expected = items[0].expected_answer
                if isinstance(expected, str):
                    expected = int(expected.strip().lstrip("+"))
                expert_disagreements.append({
                    "question_id": qid,
                    "question_type": items[0].question_type,
                    f"{e1}_response": r1,
                    f"{e2}_response": r2,
                    "expected_answer": expected,
                    "distance": distance,
                })

    # Aggregate metrics
    def safe_mean(values: list) -> float:
        return sum(values) / len(values) if values else 0.0

    def safe_div(num: int, denom: int) -> float:
        return num / denom if denom > 0 else 0.0

    metrics = AgreementMetrics(
        total_responses=len(matched_items),
        matched_responses=len(matched_items),
        unmatched_responses=0,  # Will be set by caller
        unique_questions=len(responses_by_question),
        questions_with_multiple_experts=len(multi_expert_questions),
        expert_exact_agreement=safe_div(expert_exact, expert_valid),
        expert_partial_agreement=safe_div(expert_partial, expert_valid),
        expert_mean_absolute_error=safe_div(expert_total_distance, expert_valid),
        model_exact_agreement=safe_div(model_exact, model_valid),
        model_partial_agreement=safe_div(model_partial, model_valid),
        model_mean_absolute_error=safe_div(model_total_distance, model_valid),
        expert_model_exact_agreement=safe_div(expert_model_exact, expert_model_valid),
        expert_model_partial_agreement=safe_div(expert_model_partial, expert_model_valid),
        expert_model_mean_absolute_error=safe_div(expert_model_total_distance, expert_model_valid),
        inter_expert_exact_agreement=safe_div(inter_expert_exact, inter_expert_count),
        inter_expert_partial_agreement=safe_div(inter_expert_partial, inter_expert_count),
        inter_expert_mean_absolute_error=safe_div(inter_expert_total_distance, inter_expert_count),
        by_question_type={
            qtype: {
                "expert_exact_agreement": safe_mean(data["expert_exact"]),
                "expert_mae": safe_mean(data["expert_distance"]),
                "model_exact_agreement": safe_mean(data["model_exact"]),
                "model_mae": safe_mean(data["model_distance"]),
                "count": len(data["expert_exact"]),
            }
            for qtype, data in by_question_type.items()
        },
        by_expert={
            expert: {
                "exact_agreement": safe_mean(data["expert_exact"]),
                "mae": safe_mean(data["expert_distance"]),
                "count": len(data["expert_exact"]),
            }
            for expert, data in by_expert.items()
        },
        expert_disagreements=expert_disagreements,
    )

    return results, metrics


def generate_manifest(
    metrics: AgreementMetrics,
    results: list[AgreementResult],
    csv_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate a manifest with analysis results.

    Args:
        metrics: Aggregated agreement metrics.
        results: Individual agreement results.
        csv_path: Path to the input CSV file.
        output_dir: Output directory for manifest.

    Returns:
        Manifest dictionary.
    """
    manifest = {
        "analysis_type": "sct_agreement",
        "generated_at": datetime.now().isoformat(),
        "input_file": str(csv_path),
        "output_dir": str(output_dir),
        "summary": {
            "total_responses": metrics.total_responses,
            "matched_responses": metrics.matched_responses,
            "unmatched_responses": metrics.unmatched_responses,
            "unique_questions": metrics.unique_questions,
            "questions_with_multiple_experts": metrics.questions_with_multiple_experts,
        },
        "expert_vs_expected": {
            "exact_agreement": round(metrics.expert_exact_agreement, 4),
            "partial_agreement": round(metrics.expert_partial_agreement, 4),
            "mean_absolute_error": round(metrics.expert_mean_absolute_error, 4),
        },
        "model_vs_expected": {
            "exact_agreement": round(metrics.model_exact_agreement, 4),
            "partial_agreement": round(metrics.model_partial_agreement, 4),
            "mean_absolute_error": round(metrics.model_mean_absolute_error, 4),
        },
        "expert_vs_model": {
            "exact_agreement": round(metrics.expert_model_exact_agreement, 4),
            "partial_agreement": round(metrics.expert_model_partial_agreement, 4),
            "mean_absolute_error": round(metrics.expert_model_mean_absolute_error, 4),
        },
        "inter_expert_agreement": {
            "exact_agreement": round(metrics.inter_expert_exact_agreement, 4),
            "partial_agreement": round(metrics.inter_expert_partial_agreement, 4),
            "mean_absolute_error": round(metrics.inter_expert_mean_absolute_error, 4),
            "questions_compared": metrics.questions_with_multiple_experts,
            "disagreements_count": len(metrics.expert_disagreements),
        },
        "by_question_type": {
            qtype: {k: round(v, 4) if isinstance(v, float) else v for k, v in data.items()}
            for qtype, data in metrics.by_question_type.items()
        },
        "by_expert": {
            expert: {k: round(v, 4) if isinstance(v, float) else v for k, v in data.items()}
            for expert, data in metrics.by_expert.items()
        },
        "expert_disagreements": metrics.expert_disagreements,
    }

    return manifest


def run_agreement_analysis(
    csv_path: Path,
    dataset_path: Path,
    runs_dir: Path,
    output_dir: Path,
    experiment_filter: str | None = None,
) -> dict[str, Any]:
    """Run the full agreement analysis pipeline.

    Args:
        csv_path: Path to expert responses CSV.
        dataset_path: Path to SCT dataset JSON.
        runs_dir: Path to evaluation runs directory.
        output_dir: Output directory for results.
        experiment_filter: Optional filter to select specific experiment
            (e.g., "A1" for arm A1, or "A1_llama" for specific model).

    Returns:
        Manifest dictionary with analysis results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    expert_responses = load_expert_responses(csv_path)
    dataset_items = load_dataset_items(dataset_path)
    predictions = load_model_predictions(runs_dir, experiment_filter)

    # Match and analyze
    matched, unmatched = match_expert_responses(expert_responses, dataset_items)
    results, metrics = calculate_agreement(matched, predictions)
    metrics.unmatched_responses = len(unmatched)
    metrics.total_responses = len(matched) + len(unmatched)

    # Generate outputs
    manifest = generate_manifest(metrics, results, csv_path, output_dir)

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Write detailed results
    results_data = [
        {
            "question_id": r.question_id,
            "question_type": r.question_type,
            "expert_name": r.expert_name,
            "expert_response": r.expert_response,
            "expected_answer": r.expected_answer,
            "model_prediction": r.model_prediction,
            "expert_agrees_expected": r.expert_agrees_with_expected,
            "expert_agrees_model": r.expert_agrees_with_model,
            "model_agrees_expected": r.model_agrees_with_expected,
            "expert_distance": r.expert_distance_from_expected,
            "model_distance": r.model_distance_from_expected,
        }
        for r in results
    ]
    results_path = output_dir / "agreement_details.json"
    results_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")

    # Write CSV summary
    csv_lines = [
        "question_id,question_type,expert_name,expert_response,expected_answer,"
        "model_prediction,expert_agrees_expected,model_agrees_expected"
    ]
    for r in results:
        csv_lines.append(
            f"{r.question_id},{r.question_type},{r.expert_name},{r.expert_response},"
            f"{r.expected_answer},{r.model_prediction},"
            f"{r.expert_agrees_with_expected},{r.model_agrees_with_expected}"
        )
    csv_summary_path = output_dir / "agreement_summary.csv"
    csv_summary_path.write_text("\n".join(csv_lines), encoding="utf-8")

    # Write unmatched responses for debugging
    if unmatched:
        unmatched_data = [
            {
                "row_index": r.row_index,
                "doctor_name": r.doctor_name,
                "question_type": r.question_type,
                "hypothesis": r.hypothesis[:100] + "..." if len(r.hypothesis) > 100 else r.hypothesis,
            }
            for r in unmatched
        ]
        unmatched_path = output_dir / "unmatched_responses.json"
        unmatched_path.write_text(json.dumps(unmatched_data, indent=2), encoding="utf-8")

    return manifest

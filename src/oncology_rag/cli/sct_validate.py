"""CLI to generate validated SCT ground truth from expert responses.

Validation rules:
1. Responses from verified experts (e.g., user9) are accepted directly
2. Responses from other experts (e.g., user5, user7) are only accepted
   when they agree on the same question

Output: A CSV file with validated ground truth items that can be used
for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExpertResponse:
    """A single expert response."""

    doctor_email: str
    question_key: str  # Normalized key for matching
    question_id: str | None  # Matched question ID from dataset
    vignette: str
    hypothesis: str
    new_information: str
    question_type: str
    response_value: int
    row_index: int


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    import re
    return re.sub(r"\s+", " ", text.strip().lower())


def load_responses(csv_path: Path) -> list[ExpertResponse]:
    """Load all expert responses from CSV."""
    responses = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            response_val = row.get("Response Value", "")
            if not response_val:
                continue
            try:
                value = int(float(response_val))
            except (ValueError, TypeError):
                continue

            vignette = row.get("Vignette", "")
            hypothesis = row.get("Hypothesis", "")
            new_info = row.get("New Information", "")
            key = _normalize_text(f"{vignette}|{hypothesis}|{new_info}")

            row_idx = row.get("Row Index", "")
            try:
                row_index = int(row_idx) if row_idx else 0
            except ValueError:
                row_index = 0

            responses.append(
                ExpertResponse(
                    doctor_email=row.get("Doctor Email", ""),
                    question_key=key,
                    question_id=None,
                    vignette=vignette,
                    hypothesis=hypothesis,
                    new_information=new_info,
                    question_type=row.get("Question Type", ""),
                    response_value=value,
                    row_index=row_index,
                )
            )
    return responses


def load_dataset_index(dataset_path: Path | None) -> dict[str, dict[str, Any]]:
    """Load dataset and create index by normalized key.

    If dataset_path is None or doesn't exist, returns empty dict.
    """
    if dataset_path is None or not dataset_path.exists():
        return {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    index: dict[str, dict[str, Any]] = {}
    vignettes = data if isinstance(data, list) else data.get("vignettes", [data])

    for vignette in vignettes:
        vignette_id = vignette.get("id", "")
        vignette_text = vignette.get("vignette", "")
        questions = vignette.get("questions", [])

        for q_idx, question in enumerate(questions):
            question_id = f"{vignette_id}_q{q_idx:02d}"
            hypothesis = question.get("hypothesis", "")
            new_info = question.get("new_information", "")
            key = _normalize_text(f"{vignette_text}|{hypothesis}|{new_info}")

            index[key] = {
                "question_id": question_id,
                "vignette_id": vignette_id,
                "question_type": question.get("question_type", ""),
                "hypothesis": hypothesis,
                "new_information": new_info,
                "expected_answer": question.get("expected_answer"),
                "vignette_text": vignette_text,
            }

    return index


def build_index_from_responses(responses: list[ExpertResponse]) -> dict[str, dict[str, Any]]:
    """Build a dataset index from the responses themselves.

    Groups by vignette and assigns question IDs based on question type.
    """
    # Group by vignette
    vignettes: dict[str, dict[str, Any]] = {}
    vignette_keys: dict[str, str] = {}  # normalized vignette -> vignette_id

    vignette_counter = 1
    for resp in responses:
        vignette_norm = _normalize_text(resp.vignette)

        if vignette_norm not in vignette_keys:
            vignette_id = f"v_{vignette_counter:04d}"
            vignette_keys[vignette_norm] = vignette_id
            vignettes[vignette_id] = {
                "vignette_text": resp.vignette,
                "questions": {},
            }
            vignette_counter += 1

        vignette_id = vignette_keys[vignette_norm]
        qtype = resp.question_type or "unknown"

        # Create question key
        q_key = _normalize_text(f"{resp.hypothesis}|{resp.new_information}")

        if q_key not in vignettes[vignette_id]["questions"]:
            vignettes[vignette_id]["questions"][q_key] = {
                "question_type": qtype,
                "hypothesis": resp.hypothesis,
                "new_information": resp.new_information,
            }

    # Now create the index with proper question IDs
    index: dict[str, dict[str, Any]] = {}

    for vignette_id, v_data in vignettes.items():
        vignette_text = v_data["vignette_text"]

        # Sort questions by type for consistent ordering
        type_order = {"diagnosis": 0, "management": 1, "followup": 2, "unknown": 99}
        sorted_questions = sorted(
            v_data["questions"].items(),
            key=lambda x: type_order.get(x[1]["question_type"], 99)
        )

        for q_idx, (q_key, q_data) in enumerate(sorted_questions):
            question_id = f"{vignette_id}_q{q_idx:02d}"
            full_key = _normalize_text(
                f"{vignette_text}|{q_data['hypothesis']}|{q_data['new_information']}"
            )

            index[full_key] = {
                "question_id": question_id,
                "vignette_id": vignette_id,
                "question_type": q_data["question_type"],
                "hypothesis": q_data["hypothesis"],
                "new_information": q_data["new_information"],
                "expected_answer": None,
                "vignette_text": vignette_text,
            }

    return index


def validate_responses(
    responses: list[ExpertResponse],
    dataset_index: dict[str, dict[str, Any]],
    verified_emails: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate responses and generate ground truth.

    Args:
        responses: All expert responses.
        dataset_index: Dataset indexed by question key.
        verified_emails: List of verified expert emails.

    Returns:
        Tuple of (validated items, statistics).
    """
    # Group responses by question key
    by_question: dict[str, list[ExpertResponse]] = defaultdict(list)
    for resp in responses:
        # Match to dataset
        item = dataset_index.get(resp.question_key)
        if item:
            resp.question_id = item["question_id"]
        by_question[resp.question_key].append(resp)

    validated_items: list[dict[str, Any]] = {}
    stats = {
        "total_questions": len(by_question),
        "verified_expert_questions": 0,
        "agreed_questions": 0,
        "disagreed_questions": 0,
        "single_expert_questions": 0,
        "unmatched_questions": 0,
        "by_expert": defaultdict(int),
    }

    for key, resps in by_question.items():
        item = dataset_index.get(key)
        if not item:
            stats["unmatched_questions"] += 1
            continue

        question_id = item["question_id"]

        # Check if any verified expert responded
        verified_resps = [r for r in resps if r.doctor_email in verified_emails]
        other_resps = [r for r in resps if r.doctor_email not in verified_emails]

        if verified_resps:
            # Use verified expert's response
            resp = verified_resps[0]
            stats["verified_expert_questions"] += 1
            stats["by_expert"][resp.doctor_email] += 1
            validated_items[question_id] = {
                "question_id": question_id,
                "vignette_id": item["vignette_id"],
                "question_type": item["question_type"],
                "vignette": item["vignette_text"],
                "hypothesis": item["hypothesis"],
                "new_information": item["new_information"],
                "validated_answer": resp.response_value,
                "original_expected": item["expected_answer"],
                "validation_source": "verified_expert",
                "expert_email": resp.doctor_email,
            }
        elif len(other_resps) >= 2:
            # Check if multiple experts agree
            answers = [r.response_value for r in other_resps]
            experts = [r.doctor_email for r in other_resps]

            # Check if first two agree
            if answers[0] == answers[1]:
                stats["agreed_questions"] += 1
                for email in experts[:2]:
                    stats["by_expert"][email] += 1
                validated_items[question_id] = {
                    "question_id": question_id,
                    "vignette_id": item["vignette_id"],
                    "question_type": item["question_type"],
                    "vignette": item["vignette_text"],
                    "hypothesis": item["hypothesis"],
                    "new_information": item["new_information"],
                    "validated_answer": answers[0],
                    "original_expected": item["expected_answer"],
                    "validation_source": "expert_agreement",
                    "expert_emails": ",".join(experts[:2]),
                }
            else:
                stats["disagreed_questions"] += 1
        else:
            stats["single_expert_questions"] += 1

    stats["by_expert"] = dict(stats["by_expert"])
    stats["validated_total"] = len(validated_items)

    return list(validated_items.values()), stats


def write_validated_csv(items: list[dict[str, Any]], output_path: Path) -> None:
    """Write validated items to CSV."""
    if not items:
        print("No validated items to write.")
        return

    fieldnames = [
        "question_id",
        "vignette_id",
        "question_type",
        "vignette",
        "hypothesis",
        "new_information",
        "validated_answer",
        "original_expected",
        "validation_source",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in sorted(items, key=lambda x: x["question_id"]):
            writer.writerow(item)


def write_validated_json(items: list[dict[str, Any]], output_path: Path) -> None:
    """Write validated items as JSON for use in evaluation."""
    # Group by vignette to create proper SCT format
    vignettes: dict[str, dict] = {}

    for item in items:
        vid = item["vignette_id"]
        if vid not in vignettes:
            vignettes[vid] = {
                "id": vid,
                "vignette": item["vignette"],
                "domain": "Hepatocellular Carcinoma",
                "questions": [],
            }

        vignettes[vid]["questions"].append({
            "question_type": item["question_type"],
            "hypothesis": item["hypothesis"],
            "new_information": item["new_information"],
            "expected_answer": item["validated_answer"],
            "original_expected": item["original_expected"],
            "validation_source": item["validation_source"],
        })

    # Sort vignettes by ID and questions within
    output = []
    for vid in sorted(vignettes.keys()):
        v = vignettes[vid]
        v["questions"].sort(
            key=lambda q: ["diagnosis", "management", "followup"].index(q["question_type"])
            if q["question_type"] in ["diagnosis", "management", "followup"]
            else 99
        )
        output.append(v)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate validated SCT ground truth from expert responses."
    )
    parser.add_argument(
        "--responses",
        required=True,
        help="Path to expert responses CSV file.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to original SCT dataset JSON file (optional, will build from responses if not provided).",
    )
    parser.add_argument(
        "--verified-emails",
        nargs="+",
        default=["user9@gmail.com"],
        help="Email addresses of verified experts (default: user9@gmail.com).",
    )
    parser.add_argument(
        "--output-csv",
        default="data/eval/sct_validated_ground_truth.csv",
        help="Output CSV path for validated items.",
    )
    parser.add_argument(
        "--output-json",
        default="data/eval/sct_validated_ground_truth.json",
        help="Output JSON path for validated dataset.",
    )
    return parser


def main() -> None:
    """Run the validation pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    print(f"Loading responses from: {args.responses}")
    responses = load_responses(Path(args.responses))
    print(f"  Total responses: {len(responses)}")

    # Load or build dataset index
    dataset_path = Path(args.dataset) if args.dataset else None
    if dataset_path and dataset_path.exists():
        print(f"Loading dataset from: {args.dataset}")
        dataset_index = load_dataset_index(dataset_path)
        print(f"  Total questions in dataset: {len(dataset_index)}")
    else:
        print("Building dataset index from responses...")
        dataset_index = build_index_from_responses(responses)
        print(f"  Built index with {len(dataset_index)} unique questions")

    print(f"Verified experts: {args.verified_emails}")
    print("Validating responses...")

    validated, stats = validate_responses(
        responses, dataset_index, args.verified_emails
    )

    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"\nTotal unique questions with responses: {stats['total_questions']}")
    print(f"  Unmatched to dataset: {stats['unmatched_questions']}")
    print(f"\nValidated items: {stats['validated_total']}")
    print(f"  From verified expert: {stats['verified_expert_questions']}")
    print(f"  From expert agreement: {stats['agreed_questions']}")
    print(f"\nRejected:")
    print(f"  Experts disagreed: {stats['disagreed_questions']}")
    print(f"  Single expert only: {stats['single_expert_questions']}")

    print(f"\nContributions by expert:")
    for email, count in sorted(stats["by_expert"].items()):
        print(f"  {email}: {count} validated items")

    # Write outputs
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    write_validated_csv(validated, output_csv)
    print(f"\nCSV written to: {output_csv}")

    write_validated_json(validated, output_json)
    print(f"JSON written to: {output_json}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

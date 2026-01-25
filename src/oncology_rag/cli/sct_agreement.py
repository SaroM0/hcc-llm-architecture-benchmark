"""CLI entrypoint for SCT expert agreement analysis.

This command compares expert responses collected via SCT assessments with
model predictions from evaluation runs, generating agreement metrics and
a detailed manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oncology_rag.analysis.sct_agreement import run_agreement_analysis


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for SCT agreement analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze agreement between expert responses and model predictions."
    )
    parser.add_argument(
        "--responses",
        required=True,
        help="Path to expert responses CSV file.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to SCT dataset JSON file.",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory containing evaluation runs (default: runs).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: runs/agreement/<timestamp>).",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Filter predictions by experiment ID (e.g., 'A1' or 'A1_llama').",
    )
    return parser


def _print_summary(manifest: dict) -> None:
    """Print a formatted summary of the analysis results."""
    print("\n" + "=" * 60)
    print("SCT Expert Agreement Analysis")
    print("=" * 60)

    summary = manifest["summary"]
    print(f"\nResponses analyzed: {summary['total_responses']}")
    print(f"  Matched: {summary['matched_responses']}")
    print(f"  Unmatched: {summary['unmatched_responses']}")
    print(f"  Unique questions: {summary['unique_questions']}")
    print(f"  Questions with multiple experts: {summary['questions_with_multiple_experts']}")

    print("\n--- Inter-Expert Agreement ---")
    inter = manifest["inter_expert_agreement"]
    print(f"  Questions compared: {inter['questions_compared']}")
    print(f"  Exact agreement:    {inter['exact_agreement']:.1%}")
    print(f"  Partial agreement:  {inter['partial_agreement']:.1%} (within 1 point)")
    print(f"  Mean Abs Error:     {inter['mean_absolute_error']:.3f}")
    print(f"  Disagreements:      {inter['disagreements_count']}")

    print("\n--- Expert vs Expected (Gold Standard) ---")
    expert = manifest["expert_vs_expected"]
    print(f"  Exact agreement:   {expert['exact_agreement']:.1%}")
    print(f"  Partial agreement: {expert['partial_agreement']:.1%} (within 1 point)")
    print(f"  Mean Abs Error:    {expert['mean_absolute_error']:.3f}")

    print("\n--- Model vs Expected (Gold Standard) ---")
    model = manifest["model_vs_expected"]
    print(f"  Exact agreement:   {model['exact_agreement']:.1%}")
    print(f"  Partial agreement: {model['partial_agreement']:.1%} (within 1 point)")
    print(f"  Mean Abs Error:    {model['mean_absolute_error']:.3f}")

    print("\n--- Expert vs Model ---")
    em = manifest["expert_vs_model"]
    print(f"  Exact agreement:   {em['exact_agreement']:.1%}")
    print(f"  Partial agreement: {em['partial_agreement']:.1%} (within 1 point)")
    print(f"  Mean Abs Error:    {em['mean_absolute_error']:.3f}")

    print("\n--- By Question Type ---")
    for qtype, data in manifest.get("by_question_type", {}).items():
        print(f"  {qtype}:")
        print(f"    Expert agreement: {data['expert_exact_agreement']:.1%} (n={data['count']})")
        print(f"    Model agreement:  {data['model_exact_agreement']:.1%}")

    print("\n--- By Expert ---")
    for expert_name, data in manifest.get("by_expert", {}).items():
        print(f"  {expert_name}:")
        print(f"    Agreement: {data['exact_agreement']:.1%} (n={data['count']})")
        print(f"    MAE: {data['mae']:.3f}")

    print("\n" + "=" * 60)


def main() -> None:
    """Run the SCT agreement analysis CLI."""
    parser = build_parser()
    args = parser.parse_args()

    responses_path = Path(args.responses)
    dataset_path = Path(args.dataset)
    runs_dir = Path(args.runs_dir)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = runs_dir / "agreement" / timestamp

    print(f"Analyzing expert responses from: {responses_path}")
    print(f"Using dataset: {dataset_path}")
    print(f"Loading predictions from: {runs_dir}")
    if args.experiment:
        print(f"Filtering to experiment: {args.experiment}")
    print(f"Output directory: {output_dir}")

    manifest = run_agreement_analysis(
        csv_path=responses_path,
        dataset_path=dataset_path,
        runs_dir=runs_dir,
        output_dir=output_dir,
        experiment_filter=args.experiment,
    )

    _print_summary(manifest)

    print(f"\nManifest written to: {output_dir / 'manifest.json'}")
    print(f"Detailed results: {output_dir / 'agreement_details.json'}")
    print(f"CSV summary: {output_dir / 'agreement_summary.csv'}")


if __name__ == "__main__":
    main()

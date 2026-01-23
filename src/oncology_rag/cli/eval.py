"""CLI entrypoint for evaluation runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from oncology_rag.eval.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evaluation experiments.")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single experiment
    single = subparsers.add_parser("single", help="Run a single experiment")
    single.add_argument(
        "--experiment",
        required=True,
        help="Path to experiment config YAML.",
    )
    single.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset (JSONL or SCT JSON).",
    )
    single.add_argument(
        "--provider-config",
        default="configs/providers/openrouter.yaml",
        help="Provider config YAML.",
    )
    single.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory for run artifacts.",
    )

    # Full matrix experiment
    matrix = subparsers.add_parser("matrix", help="Run full experimental matrix")
    matrix.add_argument(
        "--dataset",
        required=True,
        help="Path to SCT dataset JSON.",
    )
    matrix.add_argument(
        "--provider-config",
        default="configs/providers/openrouter.yaml",
        help="Provider config YAML.",
    )
    matrix.add_argument(
        "--embeddings-config",
        default="configs/rag/embeddings.yaml",
        help="Embeddings config YAML (for RAG arms).",
    )
    matrix.add_argument(
        "--chroma-config",
        default="configs/rag/chroma.yaml",
        help="ChromaDB config YAML (for RAG arms).",
    )
    matrix.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory for run artifacts.",
    )
    matrix.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help="Arms to run (default: A1 A2 A3 A4).",
    )
    matrix.add_argument(
        "--model-groups",
        nargs="+",
        default=None,
        help="Model groups to run (default: large small).",
    )
    matrix.add_argument(
        "--resume-from",
        default=None,
        help="Experiment ID to resume from (e.g., A1_gpt52).",
    )
    matrix.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to evaluate (for testing).",
    )

    # Legacy support: if no subcommand, treat as single
    parser.add_argument(
        "--experiment",
        help="Path to experiment config YAML (legacy mode).",
    )
    parser.add_argument(
        "--dataset",
        help="Path to dataset (legacy mode).",
    )
    parser.add_argument(
        "--provider-config",
        default="configs/providers/openrouter.yaml",
        help="Provider config YAML (legacy mode).",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory for run artifacts (legacy mode).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Legacy mode: if --experiment is provided without subcommand
    if args.command is None and args.experiment:
        run_experiment(
            experiment_path=Path(args.experiment),
            dataset_path=Path(args.dataset),
            provider_config_path=Path(args.provider_config),
            runs_dir=Path(args.runs_dir),
        )
        return

    if args.command == "single":
        run_experiment(
            experiment_path=Path(args.experiment),
            dataset_path=Path(args.dataset),
            provider_config_path=Path(args.provider_config),
            runs_dir=Path(args.runs_dir),
        )
    elif args.command == "matrix":
        from oncology_rag.eval.orchestrator import run_full_matrix

        run_full_matrix(
            provider_config_path=Path(args.provider_config),
            dataset_path=Path(args.dataset),
            runs_dir=Path(args.runs_dir),
            embeddings_config_path=Path(args.embeddings_config),
            chroma_config_path=Path(args.chroma_config),
            arms=args.arms,
            model_groups=args.model_groups,
            limit=args.limit,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

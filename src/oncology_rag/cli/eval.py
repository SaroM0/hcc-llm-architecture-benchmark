"""CLI entrypoint for evaluation runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from oncology_rag.eval.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evaluation experiments.")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSONL dataset.",
    )
    parser.add_argument(
        "--provider-config",
        default="configs/providers/openrouter.yaml",
        help="Provider config YAML.",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory for run artifacts.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(
        experiment_path=Path(args.experiment),
        dataset_path=Path(args.dataset),
        provider_config_path=Path(args.provider_config),
        runs_dir=Path(args.runs_dir),
    )


if __name__ == "__main__":
    main()

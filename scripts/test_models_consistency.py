#!/usr/bin/env python3
"""Test each model with 1 item per arm to verify LLM calls work with consistent params.

Runs A1, A2, A3 with limit=1 for each model in the provider config.
Exits with code 1 if any model fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oncology_rag.eval.runner import _load_yaml
from oncology_rag.eval.orchestrator import run_full_matrix


def get_all_model_keys(provider_config: dict) -> list[str]:
    """Extract all model keys from provider config groups.all or models."""
    groups = provider_config.get("groups", {})
    if "all" in groups:
        return list(groups["all"])
    models = provider_config.get("models", {})
    return list(models.keys())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test each model with 1 item per arm for consistency validation."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to dataset (default: data/eval/sct_validated_ground_truth.csv).",
    )
    parser.add_argument(
        "--provider-config",
        default="configs/providers/openrouter.yaml",
        help="Provider config YAML.",
    )
    parser.add_argument(
        "--embeddings-config",
        default="configs/rag/embeddings.yaml",
        help="Embeddings config (for A2, A3).",
    )
    parser.add_argument(
        "--chroma-config",
        default="configs/rag/chroma.yaml",
        help="ChromaDB config (for A2, A3).",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory for run artifacts.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Specific models to test (default: all from provider config).",
    )
    args = parser.parse_args()

    provider_config = _load_yaml(Path(args.provider_config))
    model_keys = args.models if args.models else get_all_model_keys(provider_config)

    dataset_paths = [
        Path("data/eval/sct_validated_ground_truth.csv"),
        Path("data/eval/sct_validated_ground_truth.json"),
    ]
    dataset_path = Path(args.dataset) if args.dataset else None
    if not dataset_path:
        for p in dataset_paths:
            if p.exists():
                dataset_path = p
                break
    if not dataset_path or not dataset_path.exists():
        print("Error: No dataset found. Specify --dataset or ensure sct_validated_ground_truth exists.")
        return 1

    runs_dir = Path(args.runs_dir)
    provider_path = Path(args.provider_config)
    embeddings_path = Path(args.embeddings_config)
    chroma_path = Path(args.chroma_config)

    failed: list[str] = []
    passed: list[str] = []

    print(f"Testing {len(model_keys)} models with 1 item on A1, A2, A3...")
    print()

    for model_key in model_keys:
        print(f"--- {model_key} ---")
        try:
            results = run_full_matrix(
                provider_config_path=provider_path,
                dataset_path=dataset_path,
                runs_dir=runs_dir,
                embeddings_config_path=embeddings_path,
                chroma_config_path=chroma_path,
                arms=["A1", "A2", "A3"],
                models=[model_key],
                limit=1,
            )
            expected_configs = 3
            if len(results) == expected_configs:
                accs = [r.metrics.get("accuracy", 0) for r in results]
                print(f"  OK A1/A2/A3 (acc: {accs[0]:.0%}/{accs[1]:.0%}/{accs[2]:.0%})")
                passed.append(model_key)
            elif results:
                print(f"  PARTIAL: {len(results)}/{expected_configs} configs (some arms failed)")
                failed.append(model_key)
            else:
                print("  FAILED: no configs completed")
                failed.append(model_key)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(model_key)
        print()

    print("=" * 50)
    print(f"Passed: {len(passed)}/{len(model_keys)}")
    if failed:
        print(f"Failed models: {', '.join(failed)}")
        return 1
    print("All models OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

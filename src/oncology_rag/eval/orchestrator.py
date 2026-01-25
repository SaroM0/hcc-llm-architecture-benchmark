"""Experiment orchestrator for running the full evaluation matrix.

This module orchestrates the complete experimental design:
- 10 models (5 large + 5 small)
- 4 arms (A1, A2, A3, A4)
- 200 SCT items

Total: 40 model-architecture configurations, 8000 evaluations.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from oncology_rag.eval.runner import (
    _load_yaml,
    _load_dataset,
    _resolve_arm,
    _make_run_id,
)
from oncology_rag.eval.sct.metrics import (
    SCTScorer,
    calculate_sct_metrics,
    extract_score_from_response,
    NUMERIC_TO_SCORE,
)
from oncology_rag.arms.base import ArmOutput
from oncology_rag.common.types import RunContext
from oncology_rag.llm.openrouter_client import OpenRouterClient, OpenRouterConfig
from oncology_rag.llm.router import ModelRouter
from oncology_rag.retrieval.embeddings import build_embedding_model
from oncology_rag.retrieval.retriever import Retriever
from oncology_rag.retrieval.vectorstores.chroma_store import ChromaStore


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    arm_id: str
    model_key: str
    model_class: str  # "large" or "small"


@dataclass
class MatrixConfig:
    """Configuration for the full experimental matrix."""

    arms: list[str] = field(default_factory=lambda: ["A1", "A2", "A3", "A4"])
    model_groups: list[str] = field(default_factory=lambda: ["large", "small"])
    consensus_config: dict[str, Any] = field(default_factory=lambda: {
        "num_doctors": 4,
        "max_rounds": 3,
        "consensus_threshold": 0.75,
    })
    retrieval_config: dict[str, Any] = field(default_factory=lambda: {
        "top_k": 5,
        "filters": {},
    })
    llm_params: dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.3,
        "max_tokens": 1024,
    })


@dataclass
class ExperimentResult:
    """Result from a single experiment configuration."""

    arm_id: str
    model_key: str
    model_class: str
    model_id: str
    run_id: str
    total_items: int
    predictions_path: Path
    events_path: Path
    metrics: dict[str, Any]
    latency_seconds: float


def build_experiment_matrix(
    provider_config: Mapping[str, Any],
    matrix_config: MatrixConfig,
) -> list[ExperimentConfig]:
    """Build the list of all experiment configurations.

    Returns:
        List of ExperimentConfig for each model-arm combination.
    """
    experiments = []
    groups = provider_config.get("groups", {})

    for arm_id in matrix_config.arms:
        for group_name in matrix_config.model_groups:
            model_keys = groups.get(group_name, [])
            for model_key in model_keys:
                experiments.append(
                    ExperimentConfig(
                        arm_id=arm_id,
                        model_key=model_key,
                        model_class=group_name,
                    )
                )

    return experiments


class ExperimentOrchestrator:
    """Orchestrates the full experimental matrix."""

    def __init__(
        self,
        provider_config_path: Path,
        dataset_path: Path,
        runs_dir: Path,
        embeddings_config_path: Path | None = None,
        chroma_config_path: Path | None = None,
        limit: int | None = None,
    ) -> None:
        self._provider_config_path = provider_config_path
        self._dataset_path = dataset_path
        self._runs_dir = runs_dir
        self._embeddings_config_path = embeddings_config_path
        self._chroma_config_path = chroma_config_path
        self._limit = limit

        # Load configs
        self._provider_config = _load_yaml(provider_config_path)
        self._llm_router = ModelRouter(self._provider_config)
        self._client = OpenRouterClient(
            OpenRouterConfig.from_mapping(self._provider_config)
        )

        # Lazy-load retriever (only for A2/A4)
        self._retriever: Retriever | None = None

    def _ensure_retriever(self) -> Retriever:
        """Ensure retriever is initialized for RAG arms."""
        if self._retriever is not None:
            return self._retriever

        if not self._embeddings_config_path or not self._chroma_config_path:
            raise ValueError(
                "embeddings_config_path and chroma_config_path required for RAG arms"
            )

        embeddings_cfg = _load_yaml(self._embeddings_config_path)
        chroma_cfg = _load_yaml(self._chroma_config_path)
        embedding_model = build_embedding_model(embeddings_cfg)
        store = ChromaStore(chroma_cfg)
        self._retriever = Retriever(embedding_model=embedding_model, store=store)
        return self._retriever

    def run_single_experiment(
        self,
        config: ExperimentConfig,
        matrix_config: MatrixConfig,
    ) -> ExperimentResult:
        """Run a single experiment configuration.

        Args:
            config: The experiment configuration.
            matrix_config: Global matrix settings.

        Returns:
            ExperimentResult with paths and metrics.
        """
        experiment_id = f"{config.arm_id}_{config.model_key}"
        run_id = _make_run_id(experiment_id)
        run_dir = self._runs_dir / "matrix" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Get model info
        model_spec = self._llm_router.registry.get(config.model_key)

        # Build retriever if needed
        retriever = None
        top_k = matrix_config.retrieval_config.get("top_k", 5)
        filters = matrix_config.retrieval_config.get("filters", {})
        # A2, A3, A4 all use RAG now
        if config.arm_id in ("A2", "A3", "A4"):
            retriever = self._ensure_retriever()

        # Resolve arm
        arm = _resolve_arm(
            config.arm_id,
            llm_router=self._llm_router,
            client=self._client,
            retriever=retriever,
            top_k=top_k,
            filters=filters,
        )

        # Prepare output files
        predictions_path = run_dir / "predictions.jsonl"
        events_path = run_dir / "events.jsonl"

        # Load dataset (with optional limit)
        all_items = list(_load_dataset(self._dataset_path))
        items = all_items[: self._limit] if self._limit else all_items

        # Run experiment
        started = time.monotonic()
        scorer = SCTScorer()

        with predictions_path.open("w", encoding="utf-8") as pred_file, \
             events_path.open("w", encoding="utf-8") as events_file:

            for item in items:
                # Build role overrides for this model
                overrides = {
                    "oneshot": config.model_key,
                    "oneshot_rag": config.model_key,
                    "consensus": config.model_key,
                    "consensus_rag": config.model_key,
                }

                context = RunContext(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    role_overrides=overrides,
                    output_schema=None,
                    llm_params=matrix_config.llm_params,
                )

                output: ArmOutput = arm.run_one(context, item)

                # Extract scores for comparison
                expected_answer = item.metadata.get("expected_answer")
                predicted_score = extract_score_from_response(output.prediction.answer_text)

                # Convert to display format
                predicted_answer = NUMERIC_TO_SCORE.get(predicted_score) if predicted_score is not None else None

                # Write prediction with expected/predicted scores
                pred_dict = asdict(output.prediction)
                pred_dict["expected_answer"] = expected_answer
                pred_dict["predicted_answer"] = predicted_answer
                pred_dict["is_correct"] = (predicted_answer == expected_answer) if predicted_answer and expected_answer else None
                pred_file.write(json.dumps(pred_dict) + "\n")

                # Write events
                for event in output.events:
                    events_file.write(json.dumps(event.to_dict()) + "\n")

                # Score for SCT metrics
                scorer.score_item(
                    item_id=item.question_id,
                    response=output.prediction.answer_text,
                    expected_answer=expected_answer,
                    metadata=item.metadata,
                )

        elapsed = time.monotonic() - started
        metrics = scorer.compute_metrics()

        # Write manifest
        manifest = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "arm_id": config.arm_id,
            "model_key": config.model_key,
            "model_class": config.model_class,
            "model_id": model_spec.model_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_items": len(items),
            "latency_seconds": elapsed,
            "matrix_config": {
                "consensus": matrix_config.consensus_config,
                "retrieval": matrix_config.retrieval_config,
                "llm_params": matrix_config.llm_params,
            },
            "metrics": {
                "total_items": metrics.total_items,
                "parsed_items": metrics.parsed_items,
                "exact_matches": metrics.exact_matches,
                "within_one": metrics.within_one,
                "accuracy": metrics.accuracy,
                "partial_accuracy": metrics.partial_accuracy,
                "parse_rate": metrics.parse_rate,
                "mean_absolute_error": metrics.mean_absolute_error,
                "score_distribution": metrics.score_distribution,
            },
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        return ExperimentResult(
            arm_id=config.arm_id,
            model_key=config.model_key,
            model_class=config.model_class,
            model_id=model_spec.model_id,
            run_id=run_id,
            total_items=len(items),
            predictions_path=predictions_path,
            events_path=events_path,
            metrics=manifest["metrics"],
            latency_seconds=elapsed,
        )

    def run_matrix(
        self,
        matrix_config: MatrixConfig | None = None,
        resume_from: str | None = None,
    ) -> list[ExperimentResult]:
        """Run the full experimental matrix.

        Args:
            matrix_config: Matrix configuration (uses defaults if None).
            resume_from: Optional experiment_id to resume from.

        Returns:
            List of ExperimentResult for all configurations.
        """
        config = matrix_config or MatrixConfig()
        experiments = build_experiment_matrix(self._provider_config, config)

        # Handle resume
        start_idx = 0
        if resume_from:
            for i, exp in enumerate(experiments):
                if f"{exp.arm_id}_{exp.model_key}" == resume_from:
                    start_idx = i
                    break

        results = []
        total = len(experiments)

        print(f"Running experimental matrix: {total} configurations")
        print(f"  Arms: {config.arms}")
        print(f"  Model groups: {config.model_groups}")
        print()

        for i, exp_config in enumerate(experiments[start_idx:], start=start_idx + 1):
            print(f"[{i}/{total}] Running {exp_config.arm_id} with {exp_config.model_key} ({exp_config.model_class})...")

            try:
                result = self.run_single_experiment(exp_config, config)
                results.append(result)
                print(f"  ✓ Completed in {result.latency_seconds:.1f}s | Accuracy: {result.metrics['accuracy']:.2%}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                # Continue with other experiments

        # Write summary
        self._write_matrix_summary(results, config)

        return results

    def _write_matrix_summary(
        self,
        results: list[ExperimentResult],
        config: MatrixConfig,
    ) -> None:
        """Write a summary of the matrix results."""
        summary_dir = self._runs_dir / "matrix"
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_experiments": len(results),
            "config": {
                "arms": config.arms,
                "model_groups": config.model_groups,
            },
            "results": [
                {
                    "arm_id": r.arm_id,
                    "model_key": r.model_key,
                    "model_class": r.model_class,
                    "model_id": r.model_id,
                    "accuracy": r.metrics["accuracy"],
                    "partial_accuracy": r.metrics["partial_accuracy"],
                    "latency_seconds": r.latency_seconds,
                }
                for r in results
            ],
        }

        # Aggregate by arm
        by_arm: dict[str, list[float]] = {}
        for r in results:
            by_arm.setdefault(r.arm_id, []).append(r.metrics["accuracy"])

        summary["by_arm"] = {
            arm: {"mean_accuracy": sum(acc) / len(acc), "count": len(acc)}
            for arm, acc in by_arm.items()
        }

        # Aggregate by model class
        by_class: dict[str, list[float]] = {}
        for r in results:
            by_class.setdefault(r.model_class, []).append(r.metrics["accuracy"])

        summary["by_model_class"] = {
            cls: {"mean_accuracy": sum(acc) / len(acc), "count": len(acc)}
            for cls, acc in by_class.items()
        }

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_path = summary_dir / f"summary_{timestamp}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nMatrix summary written to: {summary_path}")


def run_full_matrix(
    provider_config_path: Path,
    dataset_path: Path,
    runs_dir: Path,
    embeddings_config_path: Path | None = None,
    chroma_config_path: Path | None = None,
    arms: list[str] | None = None,
    model_groups: list[str] | None = None,
    limit: int | None = None,
) -> list[ExperimentResult]:
    """Convenience function to run the full experimental matrix.

    Args:
        provider_config_path: Path to provider config YAML.
        dataset_path: Path to SCT dataset JSON.
        runs_dir: Directory to store results.
        embeddings_config_path: Path to embeddings config (for RAG).
        chroma_config_path: Path to ChromaDB config (for RAG).
        arms: List of arms to run (default: all).
        model_groups: List of model groups to run (default: all).
        limit: Optional limit on number of items to evaluate.

    Returns:
        List of ExperimentResult for all configurations.
    """
    orchestrator = ExperimentOrchestrator(
        provider_config_path=provider_config_path,
        dataset_path=dataset_path,
        runs_dir=runs_dir,
        embeddings_config_path=embeddings_config_path,
        chroma_config_path=chroma_config_path,
        limit=limit,
    )

    config = MatrixConfig()
    if arms:
        config.arms = arms
    if model_groups:
        config.model_groups = model_groups

    return orchestrator.run_matrix(config)

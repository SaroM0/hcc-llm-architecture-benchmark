"""Evaluation runner for arms."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from oncology_rag.arms.a1_oneshot import A1OneShot
from oncology_rag.arms.a2_oneshot_rag import A2OneShotRag
from oncology_rag.arms.a3_consensus import A3Consensus
from oncology_rag.arms.a4_consensus_rag import A4ConsensusRag
from oncology_rag.arms.base import Arm, ArmOutput
from oncology_rag.common.types import QAItem, RunContext
from oncology_rag.llm.openrouter_client import OpenRouterClient, OpenRouterConfig
from oncology_rag.llm.router import ModelRouter
from oncology_rag.retrieval.embeddings import build_embedding_model
from oncology_rag.retrieval.retriever import Retriever
from oncology_rag.retrieval.vectorstores.chroma_store import ChromaStore


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")

# Dataset format detection
DATASET_FORMAT_JSONL = "jsonl"
DATASET_FORMAT_SCT = "sct"


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1), match.group(0))

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(val) for key, val in value.items()}
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency wiring
        raise ImportError("PyYAML is required to load config files") from exc
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _expand_env(raw)


def _load_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _detect_dataset_format(path: Path) -> str:
    """Detect the format of a dataset file."""
    if path.suffix == ".jsonl":
        return DATASET_FORMAT_JSONL

    # Try to detect SCT format by reading the file
    content = path.read_text(encoding="utf-8").strip()
    if content.startswith("[") or content.startswith("{"):
        try:
            data = json.loads(content)
            # Check if it looks like SCT format
            if isinstance(data, list) and data:
                first = data[0]
                if "vignette" in first and "questions" in first:
                    return DATASET_FORMAT_SCT
            elif isinstance(data, dict):
                if "vignette" in data or "vignettes" in data:
                    return DATASET_FORMAT_SCT
        except json.JSONDecodeError:
            pass

    return DATASET_FORMAT_JSONL


def _load_dataset(path: Path, dataset_format: str | None = None) -> Iterable[QAItem]:
    """Load a dataset file and yield QAItems.

    Supports both JSONL (line-delimited) and SCT JSON formats.
    """
    fmt = dataset_format or _detect_dataset_format(path)

    if fmt == DATASET_FORMAT_SCT:
        # Import SCT loader
        from oncology_rag.eval.sct.loader import load_sct_as_qa_items
        yield from load_sct_as_qa_items(path)
    else:
        # Standard JSONL format
        for idx, raw in enumerate(_load_jsonl(path)):
            yield QAItem(
                question_id=str(raw.get("question_id", idx)),
                question=str(raw.get("question", "")),
                metadata=raw.get("metadata", {}) or {},
                rubric_id=raw.get("rubric_id"),
            )


def _make_run_id(prefix: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{prefix}"


def _resolve_arm(
    arm_id: str,
    *,
    llm_router: ModelRouter,
    client: OpenRouterClient,
    retriever: Retriever | None = None,
    top_k: int | None = None,
    filters: Mapping[str, Any] | None = None,
    consensus_config: Mapping[str, Any] | None = None,
) -> Arm:
    if arm_id == "A1":
        return A1OneShot(llm_router=llm_router, client=client)
    if arm_id == "A2":
        if retriever is None or top_k is None:
            raise ValueError("A2 requires a retriever and top_k")
        return A2OneShotRag(
            llm_router=llm_router,
            client=client,
            retriever=retriever,
            top_k=top_k,
            filters=filters,
        )
    if arm_id == "A3":
        cfg = consensus_config or {}
        return A3Consensus(
            llm_router=llm_router,
            client=client,
            num_doctors=int(cfg.get("num_doctors", 4)),
            max_rounds=int(cfg.get("max_rounds", 3)),
            consensus_threshold=float(cfg.get("consensus_threshold", 0.75)),
        )
    if arm_id == "A4":
        if retriever is None:
            raise ValueError("A4 requires a retriever")
        cfg = consensus_config or {}
        return A4ConsensusRag(
            llm_router=llm_router,
            client=client,
            retriever=retriever,
            top_k=int(top_k or cfg.get("top_k", 5)),
            filters=filters,
            num_doctors=int(cfg.get("num_doctors", 4)),
            max_rounds=int(cfg.get("max_rounds", 3)),
            consensus_threshold=float(cfg.get("consensus_threshold", 0.75)),
        )
    raise ValueError(f"Unsupported arm: {arm_id}")


def run_experiment(
    *,
    experiment_path: Path,
    dataset_path: Path,
    provider_config_path: Path,
    runs_dir: Path,
) -> Path:
    experiment = _load_yaml(experiment_path)
    provider_cfg = _load_yaml(provider_config_path)
    run_id = _make_run_id(experiment.get("id", "run"))
    run_dir = runs_dir / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    role_overrides = experiment.get("model_roles", {}) or {}
    llm_params = experiment.get("llm_params", {}) or {}
    output_schema = experiment.get("output_schema")

    llm_router = ModelRouter(provider_cfg)
    client = OpenRouterClient(OpenRouterConfig.from_mapping(provider_cfg))

    arm_id = str(experiment.get("arm", "A1"))
    retriever = None
    retrieval_cfg = experiment.get("retrieval", {}) or {}
    top_k = retrieval_cfg.get("top_k")
    filters = retrieval_cfg.get("filters", {}) or {}
    consensus_cfg = experiment.get("consensus", {}) or {}
    if arm_id in ("A2", "A4"):
        embeddings_cfg_path = Path(
            experiment.get("embeddings_config", "configs/rag/embeddings.yaml")
        )
        chroma_cfg_path = Path(experiment.get("chroma_config", "configs/rag/chroma.yaml"))
        embeddings_cfg = _load_yaml(embeddings_cfg_path)
        chroma_cfg = _load_yaml(chroma_cfg_path)
        embedding_model = build_embedding_model(embeddings_cfg)
        store = ChromaStore(chroma_cfg)
        retriever = Retriever(embedding_model=embedding_model, store=store)
        if top_k is None:
            retrieval_defaults = _load_yaml(
                Path(experiment.get("retrieval_config", "configs/rag/retrieval.yaml"))
            )
            top_k = int(retrieval_defaults.get("top_k", 5))
    arm = _resolve_arm(
        arm_id,
        llm_router=llm_router,
        client=client,
        retriever=retriever,
        top_k=top_k,
        filters=filters,
        consensus_config=consensus_cfg,
    )

    model_keys = list(experiment.get("model_keys", []) or [])
    model_group = experiment.get("model_group")
    if model_group:
        model_keys = [spec.key for spec in llm_router.registry.list(str(model_group))]
    if not model_keys:
        model_keys = [str(role_overrides.get("oneshot", "default_large"))]

    predictions_path = run_dir / "predictions.jsonl"
    events_path = run_dir / "events.jsonl"

    # Detect dataset format from experiment config or auto-detect
    dataset_format = experiment.get("dataset_format")

    with predictions_path.open("w", encoding="utf-8") as pred_file, events_path.open(
        "w", encoding="utf-8"
    ) as events_file:
        for item in _load_dataset(dataset_path, dataset_format):
            for model_key in model_keys:
                overrides = dict(role_overrides)
                overrides["oneshot"] = model_key
                # Also set consensus roles for A3/A4
                overrides["consensus"] = model_key
                overrides["consensus_rag"] = model_key
                context = RunContext(
                    run_id=run_id,
                    experiment_id=str(experiment.get("id", "run")),
                    role_overrides=overrides,
                    output_schema=output_schema,
                    llm_params=llm_params,
                )
                output: ArmOutput = arm.run_one(context, item)
                pred_file.write(json.dumps(asdict(output.prediction)) + "\n")
                for event in output.events:
                    events_file.write(json.dumps(event.to_dict()) + "\n")

    manifest = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment": experiment,
        "provider_config": provider_cfg.get("api", {}),
        "model_roles": role_overrides,
        "model_keys": model_keys,
        "model_group": model_group,
        "retrieval": retrieval_cfg,
        "dataset_path": str(dataset_path),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return run_dir

"""LLM parameter resolution for consistent behavior across all models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

_DEFAULT_PARAMS: dict[str, Any] = {
    "temperature": 0.2,
    "max_tokens": 2048,
    "max_completion_tokens": 2048,
    "provider": {
        "allow_fallbacks": False,
        "quantization": "fp16",
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(raw) if isinstance(raw, dict) else {}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge override into base recursively. Override values take precedence."""
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, (dict, Mapping))
        ):
            result[key] = _deep_merge(dict(result[key]), dict(value))
        else:
            result[key] = value
    return result


def resolve_llm_params(
    override: Mapping[str, Any] | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve LLM parameters for consistent behavior across all models.

    Uses configs/llm_params.yaml as base, then applies override. Guarantees
    all arms (A1, A2, A3) and all models receive the same parameter structure
    unless explicitly overridden.

    Args:
        override: Experiment or matrix-level overrides (optional).
        config_path: Path to llm_params.yaml (default: configs/llm_params.yaml).

    Returns:
        Merged parameters ready to pass to client.chat(**params).
    """
    path = config_path or Path("configs/llm_params.yaml")
    base = _load_yaml(path) or _DEFAULT_PARAMS
    if not override:
        return dict(base)
    return _deep_merge(base, override)

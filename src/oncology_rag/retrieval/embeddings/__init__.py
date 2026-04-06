"""Embedding model interfaces and factory."""

from __future__ import annotations

import os
from typing import Any, Mapping

from .base import EmbeddingModel
from .openrouter import OpenRouterEmbeddingConfig, OpenRouterEmbeddingModel


def _resolve_env_value(value: Any, env_name: str, default: str = "") -> str:
    raw = str(value or "").strip()
    if raw in {"", f"${{{env_name}}}"}:
        return os.environ.get(env_name, default)
    return raw


def build_embedding_model(config: Mapping[str, Any]) -> EmbeddingModel:
    provider = str(config.get("provider", "openrouter"))
    if provider == "openrouter":
        openrouter_cfg = config.get("openrouter", {}) or {}
        api_key = _resolve_env_value(openrouter_cfg.get("api_key"), "OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        return OpenRouterEmbeddingModel(
            OpenRouterEmbeddingConfig(
                base_url=_resolve_env_value(
                    openrouter_cfg.get("base_url"),
                    "OPENROUTER_BASE_URL",
                    "https://openrouter.ai/api/v1",
                ),
                api_key=api_key,
                model=str(config.get("model", "")),
                timeout_s=float(openrouter_cfg.get("timeout_s", 60.0)),
                batch_size=int(config.get("batch_size", 64)),
            )
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")

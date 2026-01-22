"""Embedding model interfaces and factory."""

from __future__ import annotations

from typing import Any, Mapping

from .base import EmbeddingModel
from .openrouter import OpenRouterEmbeddingConfig, OpenRouterEmbeddingModel


def build_embedding_model(config: Mapping[str, Any]) -> EmbeddingModel:
    provider = str(config.get("provider", "openrouter"))
    if provider == "openrouter":
        openrouter_cfg = config.get("openrouter", {}) or {}
        return OpenRouterEmbeddingModel(
            OpenRouterEmbeddingConfig(
                base_url=str(openrouter_cfg.get("base_url", "")),
                api_key=str(openrouter_cfg.get("api_key", "")),
                model=str(config.get("model", "")),
                timeout_s=float(openrouter_cfg.get("timeout_s", 60.0)),
                batch_size=int(config.get("batch_size", 64)),
            )
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")

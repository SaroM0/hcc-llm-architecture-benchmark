"""Chunking strategies and factory."""

from __future__ import annotations

from typing import Any, Mapping

from .base import Chunker
from .fixed_token import FixedTokenChunker, FixedTokenConfig
from .structured_token import StructuredTokenChunker, StructuredTokenConfig


def build_chunker(config: Mapping[str, Any]) -> Chunker:
    strategy = str(config.get("strategy", "fixed_token"))
    if strategy == "fixed_token":
        return FixedTokenChunker(
            FixedTokenConfig(
                chunk_size=int(config.get("chunk_size", 512)),
                chunk_overlap=int(config.get("chunk_overlap", 64)),
            )
        )
    if strategy in {"structured_token", "structure_first"}:
        return StructuredTokenChunker(
            StructuredTokenConfig(
                chunk_size=int(config.get("chunk_size", 450)),
                chunk_overlap=int(config.get("chunk_overlap", 80)),
            )
        )
    raise ValueError(f"Unsupported chunking strategy: {strategy}")

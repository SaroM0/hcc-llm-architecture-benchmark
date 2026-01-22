"""Fixed-size token chunker."""

from __future__ import annotations

from dataclasses import dataclass

from .base import TextChunk


@dataclass(frozen=True)
class FixedTokenConfig:
    chunk_size: int
    chunk_overlap: int


class FixedTokenChunker:
    def __init__(self, config: FixedTokenConfig) -> None:
        chunk_size = int(config.chunk_size)
        chunk_overlap = int(config.chunk_overlap)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[TextChunk]:
        tokens = text.split()
        if not tokens:
            return []
        chunks: list[TextChunk] = []
        step = max(1, self._chunk_size - self._chunk_overlap)
        for start in range(0, len(tokens), step):
            end = min(start + self._chunk_size, len(tokens))
            chunk_text = " ".join(tokens[start:end])
            chunks.append(TextChunk(text=chunk_text, start=start, end=end))
            if end >= len(tokens):
                break
        return chunks

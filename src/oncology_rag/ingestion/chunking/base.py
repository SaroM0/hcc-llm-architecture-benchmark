"""Chunking interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class TextChunk:
    text: str
    start: int
    end: int
    metadata: dict[str, str] = field(default_factory=dict)


class Chunker(Protocol):
    def chunk(self, text: str) -> list[TextChunk]:
        """Split text into chunks."""

"""Structure-first chunking with token-sized windows."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base import TextChunk


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")


@dataclass(frozen=True)
class StructuredTokenConfig:
    chunk_size: int
    chunk_overlap: int


class StructuredTokenChunker:
    def __init__(self, config: StructuredTokenConfig) -> None:
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
        sections = _split_sections(text)
        chunks: list[TextChunk] = []
        for section_path, section_text in sections:
            chunks.extend(self._chunk_section(section_text, section_path))
        return chunks

    def _chunk_section(self, text: str, section_path: str) -> list[TextChunk]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return []

        chunks: list[TextChunk] = []
        cursor = 0
        current_paragraphs: list[str] = []
        current_tokens = 0
        chunk_start = 0

        def finalize_chunk(start: int, end: int) -> None:
            if not current_paragraphs:
                return
            chunk_text = "\n\n".join(current_paragraphs).strip()
            if not chunk_text:
                return
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    start=start,
                    end=end,
                    metadata={"section_path": section_path} if section_path else {},
                )
            )

        for paragraph in paragraphs:
            tokens = paragraph.split()
            token_len = len(tokens)
            if token_len == 0:
                continue
            if token_len > self._chunk_size:
                if current_paragraphs:
                    finalize_chunk(chunk_start, cursor)
                    current_paragraphs = []
                    current_tokens = 0
                for subchunk in _split_tokens(
                    paragraph, self._chunk_size, self._chunk_overlap
                ):
                    sub_tokens = subchunk.split()
                    sub_len = len(sub_tokens)
                    chunks.append(
                        TextChunk(
                            text=subchunk,
                            start=cursor,
                            end=cursor + sub_len,
                            metadata={"section_path": section_path} if section_path else {},
                        )
                    )
                    cursor += sub_len
                chunk_start = cursor
                continue

            if current_tokens + token_len > self._chunk_size:
                finalize_chunk(chunk_start, cursor)
                overlap_text = _tail_tokens(
                    "\n\n".join(current_paragraphs), self._chunk_overlap
                )
                current_paragraphs = [overlap_text] if overlap_text else []
                current_tokens = len(overlap_text.split()) if overlap_text else 0
                chunk_start = cursor

            current_paragraphs.append(paragraph)
            current_tokens += token_len
            cursor += token_len

        finalize_chunk(chunk_start, cursor)
        return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_lines: list[str] = []
    current_path: list[str] = []

    lines = text.splitlines()
    for line in lines:
        match = _HEADING_RE.match(line.strip())
        if match:
            if current_lines:
                sections.append((" > ".join(current_path), "\n".join(current_lines).strip()))
                current_lines = []
            level = len(match.group(1))
            title = match.group(2).strip()
            if level <= len(current_path):
                current_path = current_path[: level - 1]
            current_path.append(title)
            current_lines.append(line.strip())
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((" > ".join(current_path), "\n".join(current_lines).strip()))
    if not sections:
        return [("", text.strip())] if text.strip() else []
    return sections


def _split_tokens(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(tokens), step):
        end = min(start + chunk_size, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        if end >= len(tokens):
            break
    return chunks


def _tail_tokens(text: str, count: int) -> str:
    if count <= 0:
        return ""
    tokens = text.split()
    if not tokens:
        return ""
    return " ".join(tokens[-count:])

"""Load plain text and markdown documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping
import re


@dataclass(frozen=True)
class TextDocument:
    doc_id: str
    source_path: str
    text: str
    title: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return "\n".join(lines[idx + 1 :]).lstrip()
    return text


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.lstrip("#").strip()
    return None


def iter_markdown_files(root: Path, pattern: str = "**/*.md") -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    for path in sorted(root.glob(pattern)):
        if path.is_file() and path.suffix.lower() == ".md":
            yield path


def load_markdown_documents(root: Path, pattern: str = "**/*.md") -> list[TextDocument]:
    documents: list[TextDocument] = []
    for path in iter_markdown_files(root, pattern=pattern):
        raw = path.read_text(encoding="utf-8")
        cleaned = _strip_frontmatter(raw).strip()
        if not cleaned:
            continue
        rel_path = path.relative_to(root).as_posix()
        title = _first_heading(cleaned)
        guideline_name, version_year = _parse_guideline_metadata(path.stem)
        metadata: dict[str, str] = {
            "guideline_name": guideline_name,
        }
        if version_year:
            metadata["version_year"] = version_year
        documents.append(
            TextDocument(
                doc_id=rel_path,
                source_path=str(path),
                text=cleaned,
                title=title,
                metadata=metadata,
            )
        )
    return documents


def _parse_guideline_metadata(stem: str) -> tuple[str, str | None]:
    match = re.search(r"(19|20)\d{2}", stem)
    if not match:
        return stem.strip(), None
    year = match.group(0)
    name = stem.replace(year, "").replace("_", " ").replace("-", " ").strip()
    name = " ".join(name.split())
    return name, year

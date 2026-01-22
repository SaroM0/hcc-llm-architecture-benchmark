"""Prompt construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _read_template(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_oneshot(
    *,
    question: str,
    output_schema: Mapping[str, Any] | None = None,
    safety_policy: str | None = None,
    templates_dir: Path | None = None,
) -> list[dict[str, str]]:
    """Build a oneshot prompt using shared templates."""
    templates_path = templates_dir or Path(__file__).parent / "templates"
    system_template = _read_template(templates_path / "system.md")
    rag_contract = _read_template(templates_path / "rag_contract.md")
    system_parts = [system_template, rag_contract]
    if safety_policy:
        system_parts.append(safety_policy)
    if output_schema:
        system_parts.append(f"Output JSON schema: {output_schema}")
    system_content = "\n".join(system_parts).strip()
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]


def build_oneshot_rag(
    *,
    question: str,
    evidences: list[tuple[str, str]],
    output_schema: Mapping[str, Any] | None = None,
    safety_policy: str | None = None,
    templates_dir: Path | None = None,
) -> list[dict[str, str]]:
    """Build a oneshot RAG prompt with injected evidence."""
    templates_path = templates_dir or Path(__file__).parent / "templates"
    system_template = _read_template(templates_path / "system.md")
    rag_contract = _read_template(templates_path / "rag_contract.md")
    system_parts = [system_template, rag_contract]
    if safety_policy:
        system_parts.append(safety_policy)
    if output_schema:
        system_parts.append(f"Output JSON schema: {output_schema}")
    system_content = "\n".join(system_parts).strip()

    evidence_lines = ["Evidence:"]
    for chunk_id, text in evidences:
        evidence_lines.append(f"[{chunk_id}] {text}")
    evidence_block = "\n".join(evidence_lines)
    user_content = f"{evidence_block}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

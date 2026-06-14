"""Prompt construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from oncology_rag.prompts.xml_loader import load_prompt


def build_oneshot(
    *,
    question: str,
    output_schema: Mapping[str, Any] | None = None,
    safety_policy: str | None = None,
    templates_dir: Path | None = None,
) -> list[dict[str, str]]:
    """Build a oneshot prompt using shared templates."""
    templates_path = templates_dir or Path(__file__).parent / "templates"
    prompts_path = templates_path / "prompts.xml"
    system_template = load_prompt(prompts_path, "system")
    rag_contract = load_prompt(prompts_path, "rag_contract")
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

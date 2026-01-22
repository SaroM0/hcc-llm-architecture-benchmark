"""XML-backed prompt loader."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET


@lru_cache(maxsize=None)
def _load_prompts(path_str: str) -> dict[str, str]:
    path = Path(path_str)
    tree = ET.parse(path)
    root = tree.getroot()
    prompts: dict[str, str] = {}
    for prompt in root.findall("prompt"):
        prompt_id = prompt.get("id")
        if not prompt_id:
            continue
        prompts[prompt_id] = (prompt.text or "").strip()
    return prompts


def load_prompt(path: Path, prompt_id: str) -> str:
    """Load a prompt by id from an XML file."""
    prompts = _load_prompts(str(path))
    try:
        return prompts[prompt_id]
    except KeyError as exc:
        raise KeyError(f"Prompt id '{prompt_id}' not found in {path}") from exc

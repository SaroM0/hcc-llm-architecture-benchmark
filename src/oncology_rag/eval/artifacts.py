"""Helpers for writing evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_config_snapshot(output_dir: Path, snapshot: Mapping[str, Any]) -> Path:
    """Write a configuration snapshot as YAML when possible."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config_snapshot.yaml"
    try:
        import yaml
    except ImportError:
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        return path
    path.write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    return path

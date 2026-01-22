"""Model registry and specs loaded from provider config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    model_class: str
    defaults: Mapping[str, Any]


class ModelRegistry:
    """Resolve model specs from provider configuration."""

    def __init__(self, provider_config: Mapping[str, Any]) -> None:
        self._config = dict(provider_config)
        self._models: Dict[str, ModelSpec] = {}
        self._load_models()

    def _load_models(self) -> None:
        models = self._config.get("models", {})
        for key, raw in models.items():
            self._models[key] = ModelSpec(
                key=key,
                model_id=str(raw.get("id", "")),
                model_class=str(raw.get("class", "unspecified")),
                defaults=dict(raw.get("defaults", {})),
            )

    def get(self, key: str) -> ModelSpec:
        if key not in self._models:
            raise KeyError(f"Unknown model key: {key}")
        return self._models[key]

    def list(self, group: str | None = None) -> Iterable[ModelSpec]:
        if group is None:
            return list(self._models.values())
        groups = self._config.get("groups", {})
        keys = groups.get(group, [])
        return [self._models[key] for key in keys if key in self._models]

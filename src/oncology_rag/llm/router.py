"""Resolve models by role using provider config and registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .models import ModelRegistry, ModelSpec


@dataclass(frozen=True)
class RoleResolution:
    role: str
    model: ModelSpec


class ModelRouter:
    """Map roles to model specs with optional overrides."""

    def __init__(self, provider_config: Mapping[str, Any]) -> None:
        self._config = dict(provider_config)
        self._registry = ModelRegistry(self._config)
        self._roles = dict(self._config.get("roles", {}))

    def for_role(self, role: str, overrides: Mapping[str, str] | None = None) -> RoleResolution:
        role_map = dict(overrides or {})
        model_key = role_map.get(role) or self._roles.get(role)
        if not model_key:
            raise KeyError(f"No model mapped for role: {role}")
        model = self._registry.get(str(model_key))
        return RoleResolution(role=role, model=model)

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

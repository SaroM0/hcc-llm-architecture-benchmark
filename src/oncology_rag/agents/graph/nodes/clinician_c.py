"""Clinician C node."""

from __future__ import annotations

from typing import Any, Mapping

from oncology_rag.llm.router import ModelRouter, RoleResolution

ROLE = "clinician_c"


def resolve_model(
    llm_router: ModelRouter, overrides: Mapping[str, str] | None = None
) -> RoleResolution:
    return llm_router.for_role(ROLE, overrides=overrides)


def run(
    state: Mapping[str, Any],
    llm_router: ModelRouter,
    overrides: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    resolution = resolve_model(llm_router, overrides)
    return {
        "role": ROLE,
        "model_key": resolution.model.key,
        "model_id": resolution.model.model_id,
    }

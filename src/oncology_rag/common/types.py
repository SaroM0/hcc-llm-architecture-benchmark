"""Shared dataclasses for evaluation and arms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class QAItem:
    question_id: str
    question: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    rubric_id: str | None = None


@dataclass(frozen=True)
class EvidenceRef:
    source_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UsedModel:
    role: str
    model_key: str
    model_id: str


@dataclass(frozen=True)
class Prediction:
    question_id: str
    arm: str
    answer_text: str
    structured: Mapping[str, Any] | None = None
    citations: list[str] = field(default_factory=list)
    evidence_used: list[EvidenceRef] = field(default_factory=list)
    used_models: list[UsedModel] = field(default_factory=list)
    debug: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunContext:
    run_id: str
    experiment_id: str
    role_overrides: Mapping[str, str] = field(default_factory=dict)
    output_schema: Mapping[str, Any] | None = None
    llm_params: Mapping[str, Any] = field(default_factory=dict)

"""Event schema for observability sinks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class LLMCallEvent:
    event_type: str
    run_id: str
    question_id: str
    role: str
    model_key: str
    model_id: str
    latency_ms: float | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "run_id": self.run_id,
            "question_id": self.question_id,
            "role": self.role,
            "model_key": self.model_key,
            "model_id": self.model_id,
            "latency_ms": self.latency_ms,
            "usage": dict(self.usage),
            "cost_usd": self.cost_usd,
        }

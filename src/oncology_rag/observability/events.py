"""Event schema for observability sinks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


class Event(Protocol):
    def to_dict(self) -> dict[str, Any]:
        """Serialize event to a dict."""


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
    error: str | None = None

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
            "error": self.error,
        }


@dataclass(frozen=True)
class RetrievalRequestEvent:
    event_type: str
    run_id: str
    question_id: str
    arm: str
    query: str
    top_k: int
    filters: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "run_id": self.run_id,
            "question_id": self.question_id,
            "arm": self.arm,
            "query": self.query,
            "top_k": self.top_k,
            "filters": dict(self.filters),
        }


@dataclass(frozen=True)
class RetrievalResponseEvent:
    event_type: str
    run_id: str
    question_id: str
    arm: str
    n_results: int
    chunk_ids: list[str]
    latency_ms: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "run_id": self.run_id,
            "question_id": self.question_id,
            "arm": self.arm,
            "n_results": self.n_results,
            "chunk_ids": list(self.chunk_ids),
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass(frozen=True)
class ConsensusRoundEvent:
    """Event emitted at the start of each consensus round."""

    event_type: str
    run_id: str
    question_id: str
    arm: str
    round_number: int
    max_rounds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "run_id": self.run_id,
            "question_id": self.question_id,
            "arm": self.arm,
            "round_number": self.round_number,
            "max_rounds": self.max_rounds,
        }


@dataclass(frozen=True)
class DoctorResponseEvent:
    """Event emitted when a doctor agent produces output."""

    event_type: str
    run_id: str
    question_id: str
    arm: str
    doctor_id: str
    round_number: int
    hypothesis: str
    confidence: float
    has_criticisms: bool
    latency_ms: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "run_id": self.run_id,
            "question_id": self.question_id,
            "arm": self.arm,
            "doctor_id": self.doctor_id,
            "round_number": self.round_number,
            "hypothesis": self.hypothesis,
            "confidence": self.confidence,
            "has_criticisms": self.has_criticisms,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass(frozen=True)
class SupervisorDecisionEvent:
    """Event emitted when supervisor makes a decision."""

    event_type: str
    run_id: str
    question_id: str
    arm: str
    round_number: int
    consensus_reached: bool
    winner: str | None
    confidence: float
    open_issues_count: int
    continue_deliberation: bool
    latency_ms: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "run_id": self.run_id,
            "question_id": self.question_id,
            "arm": self.arm,
            "round_number": self.round_number,
            "consensus_reached": self.consensus_reached,
            "winner": self.winner,
            "confidence": self.confidence,
            "open_issues_count": self.open_issues_count,
            "continue_deliberation": self.continue_deliberation,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass(frozen=True)
class ConsensusCompleteEvent:
    """Event emitted when the consensus process completes."""

    event_type: str
    run_id: str
    question_id: str
    arm: str
    total_rounds: int
    consensus_reached: bool
    final_confidence: float
    total_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "run_id": self.run_id,
            "question_id": self.question_id,
            "arm": self.arm,
            "total_rounds": self.total_rounds,
            "consensus_reached": self.consensus_reached,
            "final_confidence": self.final_confidence,
            "total_latency_ms": self.total_latency_ms,
        }

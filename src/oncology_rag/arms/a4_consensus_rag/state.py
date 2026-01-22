"""State definitions for the multi-agent consensus system with RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage


@dataclass(frozen=True)
class RetrievedEvidence:
    """A piece of evidence retrieved from the vector store."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0


@dataclass(frozen=True)
class DoctorHypothesis:
    """Structured output from a doctor agent with evidence support."""

    doctor_id: str
    hypothesis: str
    alternatives: list[str]
    evidence_or_rationale: str
    cited_evidence: list[str]  # IDs of evidence used
    criticisms: list[str]
    updated_position: bool
    confidence: float


@dataclass(frozen=True)
class SupervisorDecision:
    """Structured decision from the supervisor."""

    consensus_reached: bool
    winner: str | None
    confidence: float
    open_issues: list[str]
    final_answer: str | None
    instructions_for_doctors: dict[str, str]


class ConsensusRagState(TypedDict, total=False):
    """State maintained throughout the consensus process with RAG.

    Attributes:
        case: The clinical case text or business context.
        task: What the agents must deliver (diagnosis/plan/answer).
        messages: Shared chat history across all agents.
        round: Current round counter (starts at 0).
        max_rounds: Maximum number of rounds before forced stop.
        doctor_outputs: Latest proposals from each doctor (keyed by doctor_id).
        supervisor_decision: Latest supervisor evaluation.
        consensus: Whether consensus has been reached.
        final_answer: The final synthesized answer when stopping.
        question_id: ID of the question being processed.
        run_id: ID of the current run.
        model_id: The LLM model to use for all agents.
        llm_params: Parameters for LLM calls.
        retrieved_evidence: Evidence retrieved for the current question.
        evidence_by_doctor: Evidence retrieved per doctor in each round.
        all_citations: All chunk IDs cited across the deliberation.
    """

    case: str
    task: str
    messages: list[BaseMessage]
    round: int
    max_rounds: int
    doctor_outputs: dict[str, DoctorHypothesis]
    supervisor_decision: SupervisorDecision | None
    consensus: bool
    final_answer: str | None
    question_id: str
    run_id: str
    model_id: str
    llm_params: dict[str, Any]
    # RAG-specific fields
    retrieved_evidence: list[RetrievedEvidence]
    evidence_by_doctor: dict[str, list[RetrievedEvidence]]
    all_citations: list[str]
    top_k: int
    filters: dict[str, Any]


@dataclass
class ConsensusRagConfig:
    """Configuration for the consensus system with RAG."""

    num_doctors: int = 4
    max_rounds: int = 3
    consensus_threshold: float = 0.75
    min_agreement: int = 3
    top_k: int = 5
    retrieval_per_doctor: bool = True  # Each doctor does their own retrieval


NodeType = Literal["admin", "doctor", "supervisor", "retrieval"]

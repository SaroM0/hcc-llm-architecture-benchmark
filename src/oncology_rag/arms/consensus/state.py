"""State definitions for multi-agent consensus system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


@dataclass
class DoctorOutput:
    """Output from a specialist doctor agent."""

    doctor_id: str
    specialty: str
    analysis: str
    hypothesis_assessment: str
    confidence: float
    cited_evidence: list[str] = field(default_factory=list)


@dataclass
class SupervisorDecision:
    """Decision from the supervisor agent."""

    consensus_reached: bool
    final_answer: str | None
    areas_of_agreement: list[str]
    areas_of_disagreement: list[str]
    reasoning: str
    confidence: float


class ConsensusState(TypedDict, total=False):
    """State for the multi-agent consensus graph."""

    # Case information
    case: str
    task: str
    question_id: str
    run_id: str

    # Retrieved evidence (for RAG)
    retrieved_evidence: list[Any]
    evidence_text: str

    # LLM configuration
    model_id: str
    llm_params: dict[str, Any]

    # Conversation state
    messages: list[dict[str, str]]
    round: int
    max_rounds: int

    # Doctor outputs
    hepatologist_output: DoctorOutput | None
    oncologist_output: DoctorOutput | None
    radiologist_output: DoctorOutput | None

    # Supervisor state
    supervisor_decision: SupervisorDecision | None
    consensus: bool
    final_answer: str | None

    # Citations
    all_citations: list[str]

    # RAG configuration
    top_k: int
    filters: dict[str, Any]

    # Token and cost tracking
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cost_usd: float
    llm_calls: list[dict[str, Any]]

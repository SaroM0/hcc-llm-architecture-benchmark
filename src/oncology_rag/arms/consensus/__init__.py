"""Shared multi-agent consensus system for A3 and A4 arms."""

from oncology_rag.arms.consensus.state import ConsensusState, DoctorOutput, SupervisorDecision
from oncology_rag.arms.consensus.prompts import (
    get_hepatologist_system_prompt,
    get_oncologist_system_prompt,
    get_radiologist_system_prompt,
    get_supervisor_system_prompt,
    get_case_presentation_prompt,
)
from oncology_rag.arms.consensus.graph import create_consensus_graph

__all__ = [
    "ConsensusState",
    "DoctorOutput",
    "SupervisorDecision",
    "get_hepatologist_system_prompt",
    "get_oncologist_system_prompt",
    "get_radiologist_system_prompt",
    "get_supervisor_system_prompt",
    "get_case_presentation_prompt",
    "create_consensus_graph",
]

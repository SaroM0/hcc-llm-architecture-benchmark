"""Prompt builders for multi-agent consensus system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from oncology_rag.arms.a3_consensus.state import DoctorHypothesis, SupervisorDecision
from oncology_rag.prompts.xml_loader import load_prompt

_PROMPTS_PATH = Path(__file__).with_suffix(".xml")


def _prompt(prompt_id: str) -> str:
    return load_prompt(_PROMPTS_PATH, prompt_id)


ADMIN_SYSTEM_PROMPT = _prompt("admin_system")
DOCTOR_SYSTEM_PROMPT = _prompt("doctor_system")
DOCTOR_FIRST_ROUND_PROMPT = _prompt("doctor_first_round")
DOCTOR_SUBSEQUENT_ROUND_PROMPT = _prompt("doctor_subsequent_round")
SUPERVISOR_SYSTEM_PROMPT = _prompt("supervisor_system")
SUPERVISOR_EVALUATION_PROMPT = _prompt("supervisor_evaluation")


def format_doctor_positions(outputs: dict[str, DoctorHypothesis]) -> str:
    """Format doctor outputs for display in prompts."""
    if not outputs:
        return "No previous positions available."

    lines = []
    for doctor_id, output in sorted(outputs.items()):
        lines.append(f"\n**{doctor_id}** (confidence: {output.confidence:.2f}):")
        lines.append(f"  Hypothesis: {output.hypothesis}")
        lines.append(f"  Rationale: {output.evidence_or_rationale}")
        if output.alternatives:
            lines.append(f"  Alternatives: {', '.join(output.alternatives)}")
        if output.criticisms:
            lines.append(f"  Criticisms: {'; '.join(output.criticisms)}")
        if output.updated_position:
            lines.append("  [Position updated from previous round]")

    return "\n".join(lines)


def format_history_summary(
    previous_decisions: list[SupervisorDecision] | None,
) -> str:
    """Format previous supervisor decisions for context."""
    if not previous_decisions:
        return ""

    lines = ["Previous round summaries:"]
    for i, decision in enumerate(previous_decisions, 1):
        lines.append(f"\nRound {i}:")
        lines.append(f"  Consensus: {decision.consensus_reached}")
        lines.append(f"  Confidence: {decision.confidence:.2f}")
        if decision.open_issues:
            lines.append(f"  Open issues: {', '.join(decision.open_issues)}")

    return "\n".join(lines)


def build_admin_messages(case: str, task: str) -> list[dict[str, str]]:
    """Build messages for admin initialization."""
    return [
        {"role": "system", "content": ADMIN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Initialize deliberation for the following case:\n\nCase: {case}\n\nTask: {task}",
        },
    ]


def build_doctor_messages(
    doctor_id: str,
    case: str,
    task: str,
    round_num: int,
    previous_outputs: dict[str, DoctorHypothesis] | None = None,
    specific_instruction: str | None = None,
) -> list[dict[str, str]]:
    """Build messages for a doctor agent."""
    system = DOCTOR_SYSTEM_PROMPT.format(doctor_id=doctor_id)

    if not previous_outputs or round_num == 1:
        user = DOCTOR_FIRST_ROUND_PROMPT.format(
            case=case,
            task=task,
            round=round_num,
        )
    else:
        positions = format_doctor_positions(previous_outputs)
        instructions = (
            f"Supervisor instruction for you: {specific_instruction}"
            if specific_instruction
            else "Continue the deliberation."
        )
        user = DOCTOR_SUBSEQUENT_ROUND_PROMPT.format(
            case=case,
            task=task,
            round=round_num,
            previous_positions=positions,
            specific_instructions=instructions,
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_supervisor_messages(
    case: str,
    task: str,
    round_num: int,
    max_rounds: int,
    doctor_outputs: dict[str, DoctorHypothesis],
    previous_decisions: list[SupervisorDecision] | None = None,
) -> list[dict[str, str]]:
    """Build messages for supervisor evaluation."""
    positions = format_doctor_positions(doctor_outputs)
    history = format_history_summary(previous_decisions)

    user = SUPERVISOR_EVALUATION_PROMPT.format(
        case=case,
        task=task,
        round=round_num,
        max_rounds=max_rounds,
        doctor_positions=positions,
        history_summary=history,
    )

    return [
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

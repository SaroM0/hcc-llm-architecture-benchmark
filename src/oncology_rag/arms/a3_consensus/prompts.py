"""Prompt builders for multi-agent consensus system."""

from __future__ import annotations

from typing import Any

from oncology_rag.arms.a3_consensus.state import DoctorHypothesis, SupervisorDecision

ADMIN_SYSTEM_PROMPT = """You are the Admin of a multi-agent medical consensus system.
Your role is to present clinical cases to a panel of expert physicians for deliberation.

You must:
1. Present the case clearly and completely
2. Define the specific task the doctors must address
3. Set expectations for the deliberation process

Do NOT provide your own medical opinions or diagnoses.
Your only job is to frame the problem for the expert panel."""

DOCTOR_SYSTEM_PROMPT = """You are Doctor {doctor_id}, a board-certified physician participating in a multi-agent medical consensus panel.

Your responsibilities in each round:
1. PROPOSE: State your primary hypothesis with clear reasoning
2. CRITIQUE: If other doctors have shared opinions, critically evaluate at least one point you disagree with or find weak
3. REFINE: Update your position if convinced by valid arguments from others

You MUST respond with a JSON object containing these exact fields:
{{
    "hypothesis": "Your primary diagnostic hypothesis or treatment recommendation",
    "alternatives": ["Alternative 1", "Alternative 2"],
    "evidence_or_rationale": "Your clinical reasoning and evidence supporting your hypothesis",
    "criticisms": ["Specific critique of another doctor's position (if applicable)"],
    "updated_position": true/false,
    "confidence": 0.0-1.0
}}

Be rigorous, evidence-based, and willing to defend your position while remaining open to valid counter-arguments."""

DOCTOR_FIRST_ROUND_PROMPT = """Case: {case}

Task: {task}

This is Round {round} of the deliberation. You are the first to respond.
Provide your initial assessment based solely on the case information.

Respond with a valid JSON object."""

DOCTOR_SUBSEQUENT_ROUND_PROMPT = """Case: {case}

Task: {task}

Round {round} of deliberation.

Previous positions from other doctors:
{previous_positions}

{specific_instructions}

Review the other doctors' positions, provide your critique, and state your (possibly updated) position.

Respond with a valid JSON object."""

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor of a multi-agent medical consensus panel.

Your responsibilities:
1. ANALYZE: Identify genuine disagreements vs cosmetic differences in wording
2. CHALLENGE: Force doctors to justify weak positions or address gaps
3. SYNTHESIZE: Find common ground and areas of agreement
4. DECIDE: Determine if meaningful consensus has been reached

Consensus criteria:
- At least 3 of 4 doctors share the same core hypothesis (or highly aligned positions)
- No critical contradictions or red flags remain
- Key uncertainties have been acknowledged

You MUST respond with a JSON object containing these exact fields:
{{
    "consensus_reached": true/false,
    "winner": "The winning hypothesis if consensus reached, or null",
    "confidence": 0.0-1.0,
    "open_issues": ["List of unresolved issues or concerns"],
    "final_answer": "The synthesized final answer if consensus reached, or null",
    "instructions_for_doctors": {{
        "doctor_1": "Specific instruction or question for Doctor 1",
        "doctor_2": "Specific instruction or question for Doctor 2",
        ...
    }}
}}

Be objective and focus on the quality of reasoning, not just agreement."""

SUPERVISOR_EVALUATION_PROMPT = """Case: {case}
Task: {task}

Round {round} of {max_rounds}.

Doctor positions this round:
{doctor_positions}

{history_summary}

Evaluate the current state of deliberation:
1. Are there genuine disagreements that need resolution?
2. Is there sufficient alignment to declare consensus?
3. What specific questions should each doctor address in the next round?

If this is the final round, you MUST provide a final_answer synthesizing the best available position.

Respond with a valid JSON object."""


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

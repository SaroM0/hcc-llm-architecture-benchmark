"""Prompt builders for multi-agent consensus system with RAG."""

from __future__ import annotations

from typing import Any

from oncology_rag.arms.a4_consensus_rag.state import (
    DoctorHypothesis,
    RetrievedEvidence,
    SupervisorDecision,
)

ADMIN_SYSTEM_PROMPT = """You are the Admin of a multi-agent medical consensus system with access to a medical knowledge base.
Your role is to present clinical cases to a panel of expert physicians for deliberation.

You must:
1. Present the case clearly and completely
2. Define the specific task the doctors must address
3. Set expectations for the deliberation process
4. Note that doctors have access to a medical knowledge base for evidence retrieval

Do NOT provide your own medical opinions or diagnoses.
Your only job is to frame the problem for the expert panel."""

DOCTOR_SYSTEM_PROMPT_RAG = """You are Doctor {doctor_id}, a board-certified physician participating in a multi-agent medical consensus panel.

You have access to retrieved medical evidence from a knowledge base. Use this evidence to support your clinical reasoning.

Your responsibilities in each round:
1. REVIEW EVIDENCE: Analyze the retrieved medical literature provided
2. PROPOSE: State your primary hypothesis with clear reasoning, citing evidence by [chunk_id]
3. CRITIQUE: If other doctors have shared opinions, critically evaluate at least one point you disagree with
4. REFINE: Update your position if convinced by valid arguments or new evidence

You MUST respond with a JSON object containing these exact fields:
{{
    "hypothesis": "Your primary diagnostic hypothesis or treatment recommendation",
    "alternatives": ["Alternative 1", "Alternative 2"],
    "evidence_or_rationale": "Your clinical reasoning citing evidence as [chunk_id]",
    "cited_evidence": ["chunk_id_1", "chunk_id_2"],
    "criticisms": ["Specific critique of another doctor's position (if applicable)"],
    "updated_position": true/false,
    "confidence": 0.0-1.0
}}

IMPORTANT: Always cite the evidence you use with [chunk_id] format. Base your reasoning on the provided evidence when available."""

DOCTOR_FIRST_ROUND_PROMPT_RAG = """Case: {case}

Task: {task}

Retrieved Medical Evidence:
{evidence_block}

This is Round {round} of the deliberation. You are the first to respond.
Provide your initial assessment based on the case information AND the retrieved evidence.
Cite evidence using [chunk_id] format.

Respond with a valid JSON object."""

DOCTOR_SUBSEQUENT_ROUND_PROMPT_RAG = """Case: {case}

Task: {task}

Retrieved Medical Evidence:
{evidence_block}

Round {round} of deliberation.

Previous positions from other doctors:
{previous_positions}

{specific_instructions}

Review the evidence, other doctors' positions, provide your critique, and state your (possibly updated) position.
Cite evidence using [chunk_id] format.

Respond with a valid JSON object."""

SUPERVISOR_SYSTEM_PROMPT_RAG = """You are the Supervisor of a multi-agent medical consensus panel with RAG capabilities.

The doctors have access to a medical knowledge base and cite evidence in their responses.

Your responsibilities:
1. ANALYZE: Identify genuine disagreements vs cosmetic differences in wording
2. VERIFY CITATIONS: Check if doctors are properly citing and using the evidence
3. CHALLENGE: Force doctors to justify weak positions or address gaps in evidence
4. SYNTHESIZE: Find common ground based on the strongest evidence
5. DECIDE: Determine if meaningful consensus has been reached

Consensus criteria:
- At least 3 of 4 doctors share the same core hypothesis (or highly aligned positions)
- Positions are well-supported by cited evidence
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

Prioritize positions that are well-supported by the retrieved evidence."""

SUPERVISOR_EVALUATION_PROMPT_RAG = """Case: {case}
Task: {task}

Round {round} of {max_rounds}.

Available Evidence Summary:
{evidence_summary}

Doctor positions this round:
{doctor_positions}

{history_summary}

Evaluate the current state of deliberation:
1. Are positions well-supported by the cited evidence?
2. Are there genuine disagreements that need resolution?
3. Is there sufficient alignment to declare consensus?
4. What specific questions should each doctor address in the next round?

If this is the final round, you MUST provide a final_answer synthesizing the best evidence-supported position.

Respond with a valid JSON object."""


def format_evidence_block(evidence: list[RetrievedEvidence]) -> str:
    """Format retrieved evidence for display in prompts."""
    if not evidence:
        return "No evidence retrieved."

    lines = []
    for ev in evidence:
        lines.append(f"[{ev.chunk_id}] {ev.text}")
        if ev.metadata:
            meta_str = ", ".join(f"{k}: {v}" for k, v in ev.metadata.items() if k != "text")
            if meta_str:
                lines.append(f"  (Source: {meta_str})")
        lines.append("")

    return "\n".join(lines).strip()


def format_evidence_summary(evidence: list[RetrievedEvidence]) -> str:
    """Format a brief summary of available evidence."""
    if not evidence:
        return "No evidence available."

    lines = [f"Total evidence chunks: {len(evidence)}"]
    for ev in evidence[:5]:  # Show first 5
        preview = ev.text[:100] + "..." if len(ev.text) > 100 else ev.text
        lines.append(f"  [{ev.chunk_id}]: {preview}")
    if len(evidence) > 5:
        lines.append(f"  ... and {len(evidence) - 5} more chunks")

    return "\n".join(lines)


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
        if output.cited_evidence:
            lines.append(f"  Cited evidence: {', '.join(output.cited_evidence)}")
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
            "content": f"Initialize deliberation for the following case:\n\nCase: {case}\n\nTask: {task}\n\nNote: Doctors will have access to retrieved medical evidence.",
        },
    ]


def build_doctor_messages_rag(
    doctor_id: str,
    case: str,
    task: str,
    round_num: int,
    evidence: list[RetrievedEvidence],
    previous_outputs: dict[str, DoctorHypothesis] | None = None,
    specific_instruction: str | None = None,
) -> list[dict[str, str]]:
    """Build messages for a doctor agent with RAG context."""
    system = DOCTOR_SYSTEM_PROMPT_RAG.format(doctor_id=doctor_id)
    evidence_block = format_evidence_block(evidence)

    if not previous_outputs or round_num == 1:
        user = DOCTOR_FIRST_ROUND_PROMPT_RAG.format(
            case=case,
            task=task,
            round=round_num,
            evidence_block=evidence_block,
        )
    else:
        positions = format_doctor_positions(previous_outputs)
        instructions = (
            f"Supervisor instruction for you: {specific_instruction}"
            if specific_instruction
            else "Continue the deliberation."
        )
        user = DOCTOR_SUBSEQUENT_ROUND_PROMPT_RAG.format(
            case=case,
            task=task,
            round=round_num,
            evidence_block=evidence_block,
            previous_positions=positions,
            specific_instructions=instructions,
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_supervisor_messages_rag(
    case: str,
    task: str,
    round_num: int,
    max_rounds: int,
    doctor_outputs: dict[str, DoctorHypothesis],
    evidence: list[RetrievedEvidence],
    previous_decisions: list[SupervisorDecision] | None = None,
) -> list[dict[str, str]]:
    """Build messages for supervisor evaluation with evidence context."""
    positions = format_doctor_positions(doctor_outputs)
    history = format_history_summary(previous_decisions)
    evidence_summary = format_evidence_summary(evidence)

    user = SUPERVISOR_EVALUATION_PROMPT_RAG.format(
        case=case,
        task=task,
        round=round_num,
        max_rounds=max_rounds,
        evidence_summary=evidence_summary,
        doctor_positions=positions,
        history_summary=history,
    )

    return [
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT_RAG},
        {"role": "user", "content": user},
    ]

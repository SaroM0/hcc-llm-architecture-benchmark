"""LangGraph workflow for multi-agent consensus system."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Literal

from langgraph.graph import END, StateGraph

from oncology_rag.arms.consensus.state import (
    ConsensusState,
    DoctorOutput,
    SupervisorDecision,
)
from oncology_rag.arms.consensus.prompts import (
    get_hepatologist_system_prompt,
    get_oncologist_system_prompt,
    get_radiologist_system_prompt,
    get_supervisor_system_prompt,
    get_case_presentation_prompt,
)
from oncology_rag.observability.events import LLMCallEvent


class ConsensusGraphBuilder:
    """Builder for consensus graph with event tracking."""

    def __init__(self) -> None:
        self.events: list[LLMCallEvent] = []
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0

    def clear_events(self) -> None:
        self.events = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0

    def add_llm_call(
        self,
        role: str,
        model_id: str,
        model_key: str,
        run_id: str,
        question_id: str,
        latency_ms: float,
        usage: dict,
        error: str | None = None,
    ) -> None:
        """Track an LLM call with usage and cost."""
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        total = usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
        cost = usage.get("cost", 0.0) or 0.0

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total
        self.total_cost_usd += cost

        event = LLMCallEvent(
            event_type="llm_call",
            run_id=run_id,
            question_id=question_id,
            role=role,
            model_key=model_key,
            model_id=model_id,
            latency_ms=latency_ms,
            usage=usage,
            cost_usd=cost if cost else None,
            error=error,
        )
        self.events.append(event)


def _parse_supervisor_json(text: str) -> dict[str, Any]:
    """Extract JSON from supervisor response."""
    # Try to find JSON block
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def _extract_likert_score(text: str) -> str | None:
    """Extract Likert score from text."""
    # Look for explicit score patterns
    patterns = [
        r"[Cc]onsensus\s*[Ss]core[\"']?\s*:\s*[\"']?([+-]?[012])",
        r"[Ff]inal\s*[Ss]core[\"']?\s*:\s*[\"']?([+-]?[012])",
        r"[Ss]core[\"']?\s*:\s*[\"']?([+-]?[012])",
        r"([+-][12]|0)\s*(?:=|:|\-)",
        r"^([+-]?[012])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            score = match.group(1)
            # Normalize score format
            if score in ("0", "+0", "-0"):
                return "0"
            if score in ("1", "+1"):
                return "+1"
            if score in ("2", "+2"):
                return "+2"
            if score == "-1":
                return "-1"
            if score == "-2":
                return "-2"
    return None


def create_consensus_graph(
    llm_call_fn: Callable[[str, list[dict[str, str]], dict[str, Any]], dict[str, Any]],
    retrieval_fn: Callable[[str, int, dict[str, Any] | None], Any] | None = None,
    max_rounds: int = 3,
    model_key: str = "unknown",
) -> tuple[StateGraph, ConsensusGraphBuilder]:
    """Create the consensus graph workflow.

    Args:
        llm_call_fn: Function to make LLM calls (model_id, messages, params) -> response
        retrieval_fn: Optional function for RAG retrieval (query, top_k, filters) -> results
        max_rounds: Maximum discussion rounds
        model_key: Model key for event tracking

    Returns:
        Compiled graph and builder for event tracking
    """
    builder = ConsensusGraphBuilder()
    import time as _time

    def retrieve_evidence(state: ConsensusState) -> ConsensusState:
        """Retrieve evidence from vector store if retrieval_fn is provided."""
        if retrieval_fn is None:
            return state

        top_k = state.get("top_k", 5)
        filters = state.get("filters", {})
        query = state.get("case", "")

        try:
            result = retrieval_fn(query, top_k, filters)
            evidence_list = []
            evidence_text_parts = []

            if result is not None:
                for chunk_id, text, metadata in zip(
                    result.ids, result.documents, result.metadatas, strict=False
                ):
                    evidence_list.append({
                        "chunk_id": chunk_id,
                        "text": text,
                        "metadata": metadata,
                    })
                    source = metadata.get("source", "Unknown")
                    evidence_text_parts.append(f"[{chunk_id}] ({source}):\n{text}")

            state["retrieved_evidence"] = evidence_list
            state["evidence_text"] = "\n\n".join(evidence_text_parts)
            state["all_citations"] = [e["chunk_id"] for e in evidence_list]

        except Exception:
            state["retrieved_evidence"] = []
            state["evidence_text"] = ""
            state["all_citations"] = []

        return state

    def run_hepatologist(state: ConsensusState) -> ConsensusState:
        """Run the Hepatologist specialist agent."""
        model_id = state.get("model_id", "")
        llm_params = state.get("llm_params", {})

        system_prompt = get_hepatologist_system_prompt()
        case_prompt = get_case_presentation_prompt(
            case=state.get("case", ""),
            task=state.get("task", ""),
            evidence_text=state.get("evidence_text"),
        )

        # Include previous round context if available
        messages = [{"role": "system", "content": system_prompt}]

        if state.get("messages"):
            for msg in state["messages"]:
                messages.append(msg)

        messages.append({"role": "user", "content": case_prompt})

        start_time = _time.monotonic()
        response = llm_call_fn(model_id, messages, llm_params)
        latency_ms = (_time.monotonic() - start_time) * 1000.0
        response_text = response.get("text", "")
        usage = response.get("usage", {}) or {}

        # Track LLM call
        builder.add_llm_call(
            role="hepatologist",
            model_id=model_id,
            model_key=model_key,
            run_id=state.get("run_id", ""),
            question_id=state.get("question_id", ""),
            latency_ms=latency_ms,
            usage=usage,
        )

        # Extract score from response
        score = _extract_likert_score(response_text)
        confidence = 0.8 if score else 0.5

        # Track citations
        cited = []
        for ev in state.get("retrieved_evidence", []):
            if ev["chunk_id"] in response_text:
                cited.append(ev["chunk_id"])

        state["hepatologist_output"] = DoctorOutput(
            doctor_id="hepatologist",
            specialty="Hepatology",
            analysis=response_text,
            hypothesis_assessment=score or "undetermined",
            confidence=confidence,
            cited_evidence=cited,
        )

        # Add to conversation
        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": f"[Hepatologist]: {response_text}"}
        ]

        return state

    def run_oncologist(state: ConsensusState) -> ConsensusState:
        """Run the Oncologist specialist agent."""
        model_id = state.get("model_id", "")
        llm_params = state.get("llm_params", {})

        system_prompt = get_oncologist_system_prompt()

        # Build context from previous responses
        context_parts = []
        if state.get("hepatologist_output"):
            context_parts.append(
                f"Hepatologist's assessment:\n{state['hepatologist_output'].analysis}"
            )

        context = "\n\n".join(context_parts)
        case_prompt = get_case_presentation_prompt(
            case=state.get("case", ""),
            task=state.get("task", ""),
            evidence_text=state.get("evidence_text"),
        )

        if context:
            case_prompt += f"\n\nPREVIOUS SPECIALIST INPUT:\n{context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case_prompt},
        ]

        start_time = _time.monotonic()
        response = llm_call_fn(model_id, messages, llm_params)
        latency_ms = (_time.monotonic() - start_time) * 1000.0
        response_text = response.get("text", "")
        usage = response.get("usage", {}) or {}

        # Track LLM call
        builder.add_llm_call(
            role="oncologist",
            model_id=model_id,
            model_key=model_key,
            run_id=state.get("run_id", ""),
            question_id=state.get("question_id", ""),
            latency_ms=latency_ms,
            usage=usage,
        )

        score = _extract_likert_score(response_text)
        confidence = 0.8 if score else 0.5

        cited = []
        for ev in state.get("retrieved_evidence", []):
            if ev["chunk_id"] in response_text:
                cited.append(ev["chunk_id"])

        state["oncologist_output"] = DoctorOutput(
            doctor_id="oncologist",
            specialty="Oncology",
            analysis=response_text,
            hypothesis_assessment=score or "undetermined",
            confidence=confidence,
            cited_evidence=cited,
        )

        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": f"[Oncologist]: {response_text}"}
        ]

        return state

    def run_radiologist(state: ConsensusState) -> ConsensusState:
        """Run the Radiologist specialist agent."""
        model_id = state.get("model_id", "")
        llm_params = state.get("llm_params", {})

        system_prompt = get_radiologist_system_prompt()

        # Build context from previous responses
        context_parts = []
        if state.get("hepatologist_output"):
            context_parts.append(
                f"Hepatologist's assessment:\n{state['hepatologist_output'].analysis}"
            )
        if state.get("oncologist_output"):
            context_parts.append(
                f"Oncologist's assessment:\n{state['oncologist_output'].analysis}"
            )

        context = "\n\n".join(context_parts)
        case_prompt = get_case_presentation_prompt(
            case=state.get("case", ""),
            task=state.get("task", ""),
            evidence_text=state.get("evidence_text"),
        )

        if context:
            case_prompt += f"\n\nPREVIOUS SPECIALIST INPUT:\n{context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case_prompt},
        ]

        start_time = _time.monotonic()
        response = llm_call_fn(model_id, messages, llm_params)
        latency_ms = (_time.monotonic() - start_time) * 1000.0
        response_text = response.get("text", "")
        usage = response.get("usage", {}) or {}

        # Track LLM call
        builder.add_llm_call(
            role="radiologist",
            model_id=model_id,
            model_key=model_key,
            run_id=state.get("run_id", ""),
            question_id=state.get("question_id", ""),
            latency_ms=latency_ms,
            usage=usage,
        )

        score = _extract_likert_score(response_text)
        confidence = 0.8 if score else 0.5

        cited = []
        for ev in state.get("retrieved_evidence", []):
            if ev["chunk_id"] in response_text:
                cited.append(ev["chunk_id"])

        state["radiologist_output"] = DoctorOutput(
            doctor_id="radiologist",
            specialty="Radiology",
            analysis=response_text,
            hypothesis_assessment=score or "undetermined",
            confidence=confidence,
            cited_evidence=cited,
        )

        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": f"[Radiologist]: {response_text}"}
        ]

        return state

    def run_supervisor(state: ConsensusState) -> ConsensusState:
        """Run the Supervisor agent to evaluate consensus."""
        model_id = state.get("model_id", "")
        llm_params = state.get("llm_params", {})

        system_prompt = get_supervisor_system_prompt()

        # Collect all specialist assessments
        assessments = []
        scores = []

        if state.get("hepatologist_output"):
            assessments.append(
                f"HEPATOLOGIST:\n{state['hepatologist_output'].analysis}\n"
                f"Score: {state['hepatologist_output'].hypothesis_assessment}"
            )
            if state["hepatologist_output"].hypothesis_assessment != "undetermined":
                scores.append(state["hepatologist_output"].hypothesis_assessment)

        if state.get("oncologist_output"):
            assessments.append(
                f"ONCOLOGIST:\n{state['oncologist_output'].analysis}\n"
                f"Score: {state['oncologist_output'].hypothesis_assessment}"
            )
            if state["oncologist_output"].hypothesis_assessment != "undetermined":
                scores.append(state["oncologist_output"].hypothesis_assessment)

        if state.get("radiologist_output"):
            assessments.append(
                f"RADIOLOGIST:\n{state['radiologist_output'].analysis}\n"
                f"Score: {state['radiologist_output'].hypothesis_assessment}"
            )
            if state["radiologist_output"].hypothesis_assessment != "undetermined":
                scores.append(state["radiologist_output"].hypothesis_assessment)

        case_prompt = f"""CASE UNDER DISCUSSION:
{state.get('case', '')}

TASK:
{state.get('task', '')}

SPECIALIST ASSESSMENTS (Round {state.get('round', 1)}):

{chr(10).join(assessments)}

Please evaluate the specialists' assessments and determine if consensus has been reached.
If consensus is reached, provide the final Likert score.
If not, identify areas of disagreement and guide further discussion."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case_prompt},
        ]

        start_time = _time.monotonic()
        response = llm_call_fn(model_id, messages, llm_params)
        latency_ms = (_time.monotonic() - start_time) * 1000.0
        response_text = response.get("text", "")
        usage = response.get("usage", {}) or {}

        # Track LLM call
        builder.add_llm_call(
            role="supervisor",
            model_id=model_id,
            model_key=model_key,
            run_id=state.get("run_id", ""),
            question_id=state.get("question_id", ""),
            latency_ms=latency_ms,
            usage=usage,
        )

        # Parse supervisor response
        parsed = _parse_supervisor_json(response_text)
        consensus_score = parsed.get("Consensus Score") or _extract_likert_score(response_text)

        # Check for consensus
        consensus_reached = "TERMINATE" in response_text or (
            len(set(scores)) == 1 and len(scores) >= 2
        )

        # If all specialists agree, use their score
        if len(set(scores)) == 1 and scores:
            consensus_score = scores[0]
            consensus_reached = True

        state["supervisor_decision"] = SupervisorDecision(
            consensus_reached=consensus_reached,
            final_answer=consensus_score,
            areas_of_agreement=parsed.get("Areas of Agreement", []),
            areas_of_disagreement=parsed.get("Areas of Disagreement", []),
            reasoning=parsed.get("Reasoning", response_text),
            confidence=0.9 if consensus_reached else 0.6,
        )

        state["consensus"] = consensus_reached
        if consensus_score:
            state["final_answer"] = consensus_score

        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": f"[Supervisor]: {response_text}"}
        ]

        state["round"] = state.get("round", 0) + 1

        return state

    def should_continue(state: ConsensusState) -> Literal["doctors", "end"]:
        """Determine if discussion should continue."""
        if state.get("consensus", False):
            return "end"
        if state.get("round", 0) >= state.get("max_rounds", max_rounds):
            return "end"
        return "doctors"

    # Build the graph
    workflow = StateGraph(ConsensusState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_evidence)
    workflow.add_node("hepatologist", run_hepatologist)
    workflow.add_node("oncologist", run_oncologist)
    workflow.add_node("radiologist", run_radiologist)
    workflow.add_node("supervisor", run_supervisor)

    # Set entry point
    workflow.set_entry_point("retrieve")

    # Add edges for sequential flow
    workflow.add_edge("retrieve", "hepatologist")
    workflow.add_edge("hepatologist", "oncologist")
    workflow.add_edge("oncologist", "radiologist")
    workflow.add_edge("radiologist", "supervisor")

    # Add conditional edge from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "doctors": "hepatologist",
            "end": END,
        },
    )

    # Compile the graph
    graph = workflow.compile()

    return graph, builder

"""LangGraph workflow for multi-agent consensus system."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def _extract_likert_score(text: str) -> str | None:
    """Extract Likert score from text."""
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
    max_rounds: int = 13,
    model_key: str = "unknown",
) -> tuple[StateGraph, ConsensusGraphBuilder]:
    """Create the consensus graph workflow.

    Doctors run in parallel per round with shared conversation_history.
    Consensus is determined solely by supervisor (TERMINATE), not by voting.

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
        """Initialize case context before specialist rounds.

        Evidence retrieval is delegated to each specialist per round.
        """
        case_prompt = get_case_presentation_prompt(
            case=state.get("case", ""),
            task=state.get("task", ""),
            evidence_text=None,
        )
        state["retrieved_evidence"] = state.get("retrieved_evidence", [])
        state["evidence_text"] = ""
        state["all_citations"] = state.get("all_citations", [])
        state["messages"] = [{"role": "user", "content": case_prompt}]
        return state

    def _build_doctor_query(
        *,
        case: str,
        task: str,
        role_name: str,
        domain: str,
        round_num: int,
        messages: list[dict[str, str]],
    ) -> str:
        history = "\n\n".join(
            m.get("content", "")
            for m in messages[-8:]
            if m.get("content")
        )
        return (
            f"Role: {role_name} ({domain})\n"
            f"Round: {round_num}\n"
            f"Case:\n{case}\n\n"
            f"Task:\n{task}\n\n"
            f"Recent Discussion:\n{history}"
        )

    def _run_single_doctor(
        doctor_id: str,
        role_name: str,
        specialty: str,
        domain: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        model_id: str,
        llm_params: dict[str, Any],
        case: str,
        task: str,
        top_k: int,
        filters: dict[str, Any] | None,
        round_num: int,
        run_id: str,
        question_id: str,
    ) -> tuple[str, DoctorOutput, float, dict, list[dict[str, Any]]]:
        retrieved_evidence: list[dict[str, Any]] = []
        evidence_text_parts: list[str] = []

        if retrieval_fn is not None:
            query = _build_doctor_query(
                case=case,
                task=task,
                role_name=role_name,
                domain=domain,
                round_num=round_num,
                messages=messages,
            )
            try:
                result = retrieval_fn(query, top_k, filters)
                if result is not None:
                    for chunk_id, text, metadata in zip(
                        result.ids, result.documents, result.metadatas, strict=False
                    ):
                        metadata = metadata or {}
                        retrieved_evidence.append({
                            "chunk_id": chunk_id,
                            "text": text,
                            "metadata": metadata,
                        })
                        source = metadata.get("source", "Unknown")
                        evidence_text_parts.append(f"[{chunk_id}] ({source}):\n{text}")
            except Exception:
                retrieved_evidence = []
                evidence_text_parts = []

        evidence_text = "\n\n".join(evidence_text_parts)
        doctor_messages = list(messages)
        doctor_messages.append({
            "role": "user",
            "content": get_case_presentation_prompt(
                case=case,
                task=task,
                evidence_text=evidence_text or None,
            ),
        })
        clean_messages = [
            {k: v for k, v in m.items() if k not in ("_run_id", "_question_id")}
            for m in doctor_messages
        ]
        start_time = _time.monotonic()
        response = llm_call_fn(model_id, clean_messages, llm_params)
        latency_ms = (_time.monotonic() - start_time) * 1000.0
        response_text = response.get("text", "")
        usage = response.get("usage", {}) or {}

        score = _extract_likert_score(response_text)
        confidence = 0.8 if score else 0.5

        cited = []
        for ev in retrieved_evidence:
            if ev["chunk_id"] in response_text:
                cited.append(ev["chunk_id"])

        output = DoctorOutput(
            doctor_id=doctor_id,
            specialty=specialty,
            analysis=response_text,
            hypothesis_assessment=score or "undetermined",
            confidence=confidence,
            cited_evidence=cited,
        )
        return doctor_id, output, latency_ms, usage, retrieved_evidence

    def run_doctors_round(state: ConsensusState) -> ConsensusState:
        """Run all doctor agents in parallel with shared conversation_history."""
        model_id = state.get("model_id", "")
        llm_params = state.get("llm_params", {})
        messages = list(state.get("messages", []))
        case = state.get("case", "")
        task = state.get("task", "")
        top_k = state.get("top_k", 5)
        filters = state.get("filters", {})

        run_id = state.get("run_id", "")
        question_id = state.get("question_id", "")

        round_num = state.get("round", 0) + 1
        doctors_config = [
            ("hepatologist", "Hepatologist", "Hepatology", "hepatology", get_hepatologist_system_prompt()),
            ("oncologist", "Oncologist", "Oncology", "oncology", get_oncologist_system_prompt()),
            ("radiologist", "Radiologist", "Radiology", "radiology", get_radiologist_system_prompt()),
        ]

        if round_num > 1:
            def _follow_up_for(role_name: str, domain: str) -> str:
                return (
                    f"\n\nAs the {role_name}, based on the full discussion above, "
                    f"provide your updated assessment from your {domain} perspective. "
                    "Consider the opinions of the other specialists and the supervisor."
                )
        else:
            def _follow_up_for(role_name: str, domain: str) -> str:
                return ""

        def _messages_for(
            doctor_id: str,
            role_name: str,
            domain: str,
            system_prompt: str,
        ) -> list[dict[str, Any]]:
            msgs = [{"role": "system", "content": system_prompt}]
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                msgs.append({"role": role, "content": content})
            if round_num > 1:
                follow_up = _follow_up_for(role_name, domain)
                msgs.append({"role": "user", "content": follow_up})
            return msgs

        outputs: dict[str, DoctorOutput] = {}
        round_evidence: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    _run_single_doctor,
                    doctor_id,
                    role_name,
                    specialty,
                    domain,
                    system_prompt,
                    _messages_for(doctor_id, role_name, domain, system_prompt),
                    model_id,
                    llm_params,
                    case,
                    task,
                    top_k,
                    filters,
                    round_num,
                    run_id,
                    question_id,
                ): doctor_id
                for doctor_id, role_name, specialty, domain, system_prompt in doctors_config
            }
            for future in as_completed(futures):
                doctor_id, output, latency_ms, usage, doctor_evidence = future.result()
                outputs[doctor_id] = output
                round_evidence.extend(doctor_evidence)
                builder.add_llm_call(
                    role=doctor_id,
                    model_id=model_id,
                    model_key=model_key,
                    run_id=run_id,
                    question_id=question_id,
                    latency_ms=latency_ms,
                    usage=usage,
                )

        state["hepatologist_output"] = outputs.get("hepatologist")
        state["oncologist_output"] = outputs.get("oncologist")
        state["radiologist_output"] = outputs.get("radiologist")

        new_messages = list(messages)
        for doctor_id, role_name, specialty, domain, _ in doctors_config:
            out = outputs.get(doctor_id)
            if out:
                new_messages.append({
                    "role": "assistant",
                    "content": f"[{role_name}]: {out.analysis}",
                })

        state["messages"] = new_messages
        previous_evidence = list(state.get("retrieved_evidence", []))
        state["retrieved_evidence"] = previous_evidence + round_evidence
        state["all_citations"] = list({
            ev.get("chunk_id")
            for ev in state["retrieved_evidence"]
            if ev.get("chunk_id")
        })
        return state

    def run_supervisor(state: ConsensusState) -> ConsensusState:
        """Run the Supervisor agent; consensus is decided solely by supervisor."""
        model_id = state.get("model_id", "")
        llm_params = state.get("llm_params", {})

        system_prompt = get_supervisor_system_prompt()
        messages = state.get("messages", [])

        def _format_message(m: dict[str, Any]) -> str:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                return f"Admin/Case:\n{content}"
            for label in ("[Hepatologist]:", "[Oncologist]:", "[Radiologist]:", "[Supervisor]:"):
                if content.startswith(label):
                    specialty_name = label.strip("[]:")
                    body = content[len(label):].lstrip()
                    return f"{specialty_name}:\n{body}"
            return f"Assistant:\n{content}"

        conversation_text = "\n\n".join(_format_message(m) for m in messages)

        case_prompt = f"""CASE UNDER DISCUSSION:
{state.get('case', '')}

TASK:
{state.get('task', '')}

FULL CONVERSATION HISTORY (Round {state.get('round', 1)}):

{conversation_text}

Please evaluate the specialists' assessments and determine if consensus has been reached.
If consensus is reached, provide the final Likert score and output TERMINATE.
If not, identify areas of disagreement and guide further discussion."""

        supervisor_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case_prompt},
        ]

        start_time = _time.monotonic()
        response = llm_call_fn(model_id, supervisor_messages, llm_params)
        latency_ms = (_time.monotonic() - start_time) * 1000.0
        response_text = response.get("text", "")
        usage = response.get("usage", {}) or {}

        builder.add_llm_call(
            role="supervisor",
            model_id=model_id,
            model_key=model_key,
            run_id=state.get("run_id", ""),
            question_id=state.get("question_id", ""),
            latency_ms=latency_ms,
            usage=usage,
        )

        parsed = _parse_supervisor_json(response_text)
        consensus_score = parsed.get("Consensus Score") or _extract_likert_score(response_text)

        consensus_reached = "TERMINATE" in response_text.upper()

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

    workflow = StateGraph(ConsensusState)

    workflow.add_node("retrieve", retrieve_evidence)
    workflow.add_node("doctors_round", run_doctors_round)
    workflow.add_node("supervisor", run_supervisor)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "doctors_round")
    workflow.add_edge("doctors_round", "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "doctors": "doctors_round",
            "end": END,
        },
    )

    graph = workflow.compile()
    return graph, builder

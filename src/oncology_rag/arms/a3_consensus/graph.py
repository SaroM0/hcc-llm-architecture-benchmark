"""LangGraph implementation of the multi-agent consensus system."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from oncology_rag.arms.a3_consensus.prompts import (
    build_doctor_messages,
    build_supervisor_messages,
)
from oncology_rag.arms.a3_consensus.state import (
    ConsensusConfig,
    ConsensusState,
    DoctorHypothesis,
    SupervisorDecision,
)
from oncology_rag.observability.events import (
    ConsensusCompleteEvent,
    ConsensusRoundEvent,
    DoctorResponseEvent,
    LLMCallEvent,
    SupervisorDecisionEvent,
)

LLMCallFn = Callable[[str, list[dict[str, str]], dict[str, Any]], dict[str, Any]]


def _parse_doctor_response(raw: str, doctor_id: str) -> DoctorHypothesis:
    """Parse doctor JSON response into structured output."""
    try:
        text = raw.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return DoctorHypothesis(
            doctor_id=doctor_id,
            hypothesis=raw[:500],
            alternatives=[],
            evidence_or_rationale="Failed to parse structured response",
            criticisms=[],
            updated_position=False,
            confidence=0.5,
        )

    return DoctorHypothesis(
        doctor_id=doctor_id,
        hypothesis=str(data.get("hypothesis", "")),
        alternatives=list(data.get("alternatives", [])),
        evidence_or_rationale=str(data.get("evidence_or_rationale", "")),
        criticisms=list(data.get("criticisms", [])),
        updated_position=bool(data.get("updated_position", False)),
        confidence=float(data.get("confidence", 0.5)),
    )


def _parse_supervisor_response(raw: str) -> SupervisorDecision:
    """Parse supervisor JSON response into structured decision."""
    try:
        text = raw.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return SupervisorDecision(
            consensus_reached=False,
            winner=None,
            confidence=0.0,
            open_issues=["Failed to parse supervisor response"],
            final_answer=None,
            instructions_for_doctors={},
        )

    return SupervisorDecision(
        consensus_reached=bool(data.get("consensus_reached", False)),
        winner=data.get("winner"),
        confidence=float(data.get("confidence", 0.0)),
        open_issues=list(data.get("open_issues", [])),
        final_answer=data.get("final_answer"),
        instructions_for_doctors=dict(data.get("instructions_for_doctors", {})),
    )


class ConsensusGraphBuilder:
    """Builder for the multi-agent consensus graph."""

    def __init__(
        self,
        llm_call_fn: LLMCallFn,
        config: ConsensusConfig | None = None,
    ) -> None:
        self._llm_call = llm_call_fn
        self._config = config or ConsensusConfig()
        self._events: list[Any] = []
        self._previous_decisions: list[SupervisorDecision] = []

    @property
    def events(self) -> list[Any]:
        """Return collected events."""
        return self._events

    def clear_events(self) -> None:
        """Clear collected events."""
        self._events = []
        self._previous_decisions = []

    def _admin_node(self, state: ConsensusState) -> dict[str, Any]:
        """Admin node: Initialize the deliberation."""
        self._events.append(
            ConsensusRoundEvent(
                event_type="consensus.round_start",
                run_id=state["run_id"],
                question_id=state["question_id"],
                arm="A3",
                round_number=1,
                max_rounds=state["max_rounds"],
            )
        )

        initial_message = HumanMessage(
            content=f"Case: {state['case']}\n\nTask: {state['task']}"
        )

        return {
            "messages": [initial_message],
            "round": 1,
            "doctor_outputs": {},
            "consensus": False,
            "final_answer": None,
        }

    def _create_doctor_node(self, doctor_id: str) -> Callable[[ConsensusState], dict[str, Any]]:
        """Create a doctor node function."""

        def doctor_node(state: ConsensusState) -> dict[str, Any]:
            round_num = state.get("round", 1)
            previous_outputs = state.get("doctor_outputs", {})

            supervisor_decision = state.get("supervisor_decision")
            specific_instruction = None
            if supervisor_decision and supervisor_decision.instructions_for_doctors:
                specific_instruction = supervisor_decision.instructions_for_doctors.get(
                    doctor_id
                )

            messages = build_doctor_messages(
                doctor_id=doctor_id,
                case=state["case"],
                task=state["task"],
                round_num=round_num,
                previous_outputs=previous_outputs if round_num > 1 else None,
                specific_instruction=specific_instruction,
            )

            started = time.monotonic()
            error = None
            response_text = ""

            try:
                response = self._llm_call(
                    state["model_id"],
                    messages,
                    state.get("llm_params", {}),
                )
                response_text = str(response.get("text", ""))
            except Exception as exc:
                error = str(exc)

            latency_ms = (time.monotonic() - started) * 1000.0
            hypothesis = _parse_doctor_response(response_text, doctor_id)

            self._events.append(
                DoctorResponseEvent(
                    event_type="consensus.doctor_response",
                    run_id=state["run_id"],
                    question_id=state["question_id"],
                    arm="A3",
                    doctor_id=doctor_id,
                    round_number=round_num,
                    hypothesis=hypothesis.hypothesis,
                    confidence=hypothesis.confidence,
                    has_criticisms=len(hypothesis.criticisms) > 0,
                    latency_ms=latency_ms,
                    error=error,
                )
            )

            new_outputs = dict(state.get("doctor_outputs", {}))
            new_outputs[doctor_id] = hypothesis

            new_messages = list(state.get("messages", []))
            new_messages.append(
                AIMessage(
                    content=f"[{doctor_id}]: {response_text}",
                    name=doctor_id,
                )
            )

            return {
                "doctor_outputs": new_outputs,
                "messages": new_messages,
            }

        return doctor_node

    def _supervisor_node(self, state: ConsensusState) -> dict[str, Any]:
        """Supervisor node: Evaluate positions and decide."""
        round_num = state.get("round", 1)
        max_rounds = state.get("max_rounds", self._config.max_rounds)
        doctor_outputs = state.get("doctor_outputs", {})

        messages = build_supervisor_messages(
            case=state["case"],
            task=state["task"],
            round_num=round_num,
            max_rounds=max_rounds,
            doctor_outputs=doctor_outputs,
            previous_decisions=self._previous_decisions if self._previous_decisions else None,
        )

        started = time.monotonic()
        error = None
        response_text = ""

        try:
            response = self._llm_call(
                state["model_id"],
                messages,
                state.get("llm_params", {}),
            )
            response_text = str(response.get("text", ""))
        except Exception as exc:
            error = str(exc)

        latency_ms = (time.monotonic() - started) * 1000.0
        decision = _parse_supervisor_response(response_text)

        self._previous_decisions.append(decision)

        is_final_round = round_num >= max_rounds
        should_continue = not decision.consensus_reached and not is_final_round

        self._events.append(
            SupervisorDecisionEvent(
                event_type="consensus.supervisor_decision",
                run_id=state["run_id"],
                question_id=state["question_id"],
                arm="A3",
                round_number=round_num,
                consensus_reached=decision.consensus_reached,
                winner=decision.winner,
                confidence=decision.confidence,
                open_issues_count=len(decision.open_issues),
                continue_deliberation=should_continue,
                latency_ms=latency_ms,
                error=error,
            )
        )

        new_messages = list(state.get("messages", []))
        new_messages.append(
            AIMessage(
                content=f"[Supervisor]: {response_text}",
                name="supervisor",
            )
        )

        final_answer = decision.final_answer
        if is_final_round and not final_answer:
            if decision.winner:
                final_answer = f"Based on the deliberation, the consensus position is: {decision.winner}"
            elif doctor_outputs:
                best_doctor = max(
                    doctor_outputs.values(),
                    key=lambda x: x.confidence,
                )
                final_answer = f"No full consensus reached. Best supported position ({best_doctor.doctor_id}, confidence {best_doctor.confidence:.2f}): {best_doctor.hypothesis}"

        return {
            "messages": new_messages,
            "supervisor_decision": decision,
            "consensus": decision.consensus_reached or is_final_round,
            "final_answer": final_answer,
            "round": round_num + 1 if should_continue else round_num,
        }

    def _should_continue(
        self, state: ConsensusState
    ) -> Literal["doctors", "end"]:
        """Router: decide whether to continue deliberation or end."""
        if state.get("consensus", False):
            return "end"
        if state.get("round", 1) > state.get("max_rounds", self._config.max_rounds):
            return "end"
        return "doctors"

    def build(self) -> StateGraph:
        """Build and return the consensus graph."""
        graph = StateGraph(ConsensusState)

        graph.add_node("admin", self._admin_node)

        for i in range(1, self._config.num_doctors + 1):
            doctor_id = f"doctor_{i}"
            graph.add_node(doctor_id, self._create_doctor_node(doctor_id))

        graph.add_node("supervisor", self._supervisor_node)

        graph.add_edge(START, "admin")
        graph.add_edge("admin", "doctor_1")

        for i in range(1, self._config.num_doctors):
            graph.add_edge(f"doctor_{i}", f"doctor_{i + 1}")

        graph.add_edge(f"doctor_{self._config.num_doctors}", "supervisor")

        graph.add_conditional_edges(
            "supervisor",
            self._should_continue,
            {
                "doctors": "doctor_1",
                "end": END,
            },
        )

        return graph


def create_consensus_graph(
    llm_call_fn: LLMCallFn,
    config: ConsensusConfig | None = None,
) -> tuple[Any, ConsensusGraphBuilder]:
    """Create a compiled consensus graph.

    Returns:
        Tuple of (compiled_graph, builder) where builder can be used to access events.
    """
    builder = ConsensusGraphBuilder(llm_call_fn, config)
    graph = builder.build()
    compiled = graph.compile()
    return compiled, builder

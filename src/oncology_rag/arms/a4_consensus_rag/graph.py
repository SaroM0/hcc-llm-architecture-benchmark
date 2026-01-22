"""LangGraph implementation of the multi-agent consensus system with RAG."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from oncology_rag.arms.a4_consensus_rag.prompts import (
    build_doctor_messages_rag,
    build_supervisor_messages_rag,
)
from oncology_rag.arms.a4_consensus_rag.state import (
    ConsensusRagConfig,
    ConsensusRagState,
    DoctorHypothesis,
    RetrievedEvidence,
    SupervisorDecision,
)
from oncology_rag.observability.events import (
    ConsensusCompleteEvent,
    ConsensusRoundEvent,
    DoctorResponseEvent,
    RetrievalRequestEvent,
    RetrievalResponseEvent,
    SupervisorDecisionEvent,
)
from oncology_rag.retrieval.retriever import Retriever, RetrievalResult

LLMCallFn = Callable[[str, list[dict[str, str]], dict[str, Any]], dict[str, Any]]
RetrievalFn = Callable[[str, int, dict[str, Any] | None], RetrievalResult]


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
            cited_evidence=[],
            criticisms=[],
            updated_position=False,
            confidence=0.5,
        )

    return DoctorHypothesis(
        doctor_id=doctor_id,
        hypothesis=str(data.get("hypothesis", "")),
        alternatives=list(data.get("alternatives", [])),
        evidence_or_rationale=str(data.get("evidence_or_rationale", "")),
        cited_evidence=list(data.get("cited_evidence", [])),
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


class ConsensusRagGraphBuilder:
    """Builder for the multi-agent consensus graph with RAG."""

    def __init__(
        self,
        llm_call_fn: LLMCallFn,
        retrieval_fn: RetrievalFn,
        config: ConsensusRagConfig | None = None,
    ) -> None:
        self._llm_call = llm_call_fn
        self._retrieval_fn = retrieval_fn
        self._config = config or ConsensusRagConfig()
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

    def _retrieval_node(self, state: ConsensusRagState) -> dict[str, Any]:
        """Retrieval node: Fetch evidence from vector store."""
        query = state["case"]
        top_k = state.get("top_k", self._config.top_k)
        filters = state.get("filters", {})

        self._events.append(
            RetrievalRequestEvent(
                event_type="retrieval.request",
                run_id=state["run_id"],
                question_id=state["question_id"],
                arm="A4",
                query=query,
                top_k=top_k,
                filters=filters,
            )
        )

        started = time.monotonic()
        error = None
        evidence_list: list[RetrievedEvidence] = []

        try:
            result = self._retrieval_fn(query, top_k, filters if filters else None)
            for chunk_id, text, metadata, distance in zip(
                result.ids,
                result.documents,
                result.metadatas,
                result.distances,
                strict=False,
            ):
                evidence_list.append(
                    RetrievedEvidence(
                        chunk_id=str(chunk_id),
                        text=str(text),
                        metadata=dict(metadata) if metadata else {},
                        distance=float(distance) if distance else 0.0,
                    )
                )
        except Exception as exc:
            error = str(exc)

        latency_ms = (time.monotonic() - started) * 1000.0

        self._events.append(
            RetrievalResponseEvent(
                event_type="retrieval.response",
                run_id=state["run_id"],
                question_id=state["question_id"],
                arm="A4",
                n_results=len(evidence_list),
                chunk_ids=[ev.chunk_id for ev in evidence_list],
                latency_ms=latency_ms,
                error=error,
            )
        )

        return {
            "retrieved_evidence": evidence_list,
        }

    def _admin_node(self, state: ConsensusRagState) -> dict[str, Any]:
        """Admin node: Initialize the deliberation."""
        self._events.append(
            ConsensusRoundEvent(
                event_type="consensus.round_start",
                run_id=state["run_id"],
                question_id=state["question_id"],
                arm="A4",
                round_number=1,
                max_rounds=state["max_rounds"],
            )
        )

        initial_message = HumanMessage(
            content=f"Case: {state['case']}\n\nTask: {state['task']}\n\n[Evidence has been retrieved from the knowledge base]"
        )

        return {
            "messages": [initial_message],
            "round": 1,
            "doctor_outputs": {},
            "evidence_by_doctor": {},
            "all_citations": [],
            "consensus": False,
            "final_answer": None,
        }

    def _create_doctor_node(
        self, doctor_id: str
    ) -> Callable[[ConsensusRagState], dict[str, Any]]:
        """Create a doctor node function with RAG support."""

        def doctor_node(state: ConsensusRagState) -> dict[str, Any]:
            round_num = state.get("round", 1)
            previous_outputs = state.get("doctor_outputs", {})
            evidence = state.get("retrieved_evidence", [])

            supervisor_decision = state.get("supervisor_decision")
            specific_instruction = None
            if supervisor_decision and supervisor_decision.instructions_for_doctors:
                specific_instruction = supervisor_decision.instructions_for_doctors.get(
                    doctor_id
                )

            messages = build_doctor_messages_rag(
                doctor_id=doctor_id,
                case=state["case"],
                task=state["task"],
                round_num=round_num,
                evidence=evidence,
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
                    arm="A4",
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

            # Track citations
            all_citations = list(state.get("all_citations", []))
            all_citations.extend(hypothesis.cited_evidence)

            # Track evidence by doctor
            evidence_by_doctor = dict(state.get("evidence_by_doctor", {}))
            evidence_by_doctor[doctor_id] = evidence

            return {
                "doctor_outputs": new_outputs,
                "messages": new_messages,
                "all_citations": all_citations,
                "evidence_by_doctor": evidence_by_doctor,
            }

        return doctor_node

    def _supervisor_node(self, state: ConsensusRagState) -> dict[str, Any]:
        """Supervisor node: Evaluate positions and decide."""
        round_num = state.get("round", 1)
        max_rounds = state.get("max_rounds", self._config.max_rounds)
        doctor_outputs = state.get("doctor_outputs", {})
        evidence = state.get("retrieved_evidence", [])

        messages = build_supervisor_messages_rag(
            case=state["case"],
            task=state["task"],
            round_num=round_num,
            max_rounds=max_rounds,
            doctor_outputs=doctor_outputs,
            evidence=evidence,
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
                arm="A4",
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
                final_answer = f"Based on the deliberation and evidence review, the consensus position is: {decision.winner}"
            elif doctor_outputs:
                best_doctor = max(
                    doctor_outputs.values(),
                    key=lambda x: x.confidence,
                )
                final_answer = f"No full consensus reached. Best evidence-supported position ({best_doctor.doctor_id}, confidence {best_doctor.confidence:.2f}): {best_doctor.hypothesis}"

        return {
            "messages": new_messages,
            "supervisor_decision": decision,
            "consensus": decision.consensus_reached or is_final_round,
            "final_answer": final_answer,
            "round": round_num + 1 if should_continue else round_num,
        }

    def _should_continue(
        self, state: ConsensusRagState
    ) -> Literal["doctors", "end"]:
        """Router: decide whether to continue deliberation or end."""
        if state.get("consensus", False):
            return "end"
        if state.get("round", 1) > state.get("max_rounds", self._config.max_rounds):
            return "end"
        return "doctors"

    def build(self) -> StateGraph:
        """Build and return the consensus graph with RAG."""
        graph = StateGraph(ConsensusRagState)

        # Add nodes
        graph.add_node("retrieval", self._retrieval_node)
        graph.add_node("admin", self._admin_node)

        for i in range(1, self._config.num_doctors + 1):
            doctor_id = f"doctor_{i}"
            graph.add_node(doctor_id, self._create_doctor_node(doctor_id))

        graph.add_node("supervisor", self._supervisor_node)

        # Add edges
        # START -> retrieval -> admin -> doctors -> supervisor
        graph.add_edge(START, "retrieval")
        graph.add_edge("retrieval", "admin")
        graph.add_edge("admin", "doctor_1")

        # Chain doctors sequentially
        for i in range(1, self._config.num_doctors):
            graph.add_edge(f"doctor_{i}", f"doctor_{i + 1}")

        # Last doctor -> supervisor
        graph.add_edge(f"doctor_{self._config.num_doctors}", "supervisor")

        # Conditional edge from supervisor
        graph.add_conditional_edges(
            "supervisor",
            self._should_continue,
            {
                "doctors": "doctor_1",  # Go back to first doctor for new round
                "end": END,
            },
        )

        return graph


def create_consensus_rag_graph(
    llm_call_fn: LLMCallFn,
    retrieval_fn: RetrievalFn,
    config: ConsensusRagConfig | None = None,
) -> tuple[Any, ConsensusRagGraphBuilder]:
    """Create a compiled consensus graph with RAG.

    Returns:
        Tuple of (compiled_graph, builder) where builder can be used to access events.
    """
    builder = ConsensusRagGraphBuilder(llm_call_fn, retrieval_fn, config)
    graph = builder.build()
    compiled = graph.compile()
    return compiled, builder

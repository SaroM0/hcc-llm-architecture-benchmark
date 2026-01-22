"""A4 Multi-Agent Consensus with RAG arm implementation."""

from __future__ import annotations

import time
from typing import Any, Mapping

from oncology_rag.arms.a4_consensus_rag.graph import create_consensus_rag_graph
from oncology_rag.arms.a4_consensus_rag.state import ConsensusRagConfig, ConsensusRagState
from oncology_rag.arms.base import ArmOutput
from oncology_rag.common.types import EvidenceRef, Prediction, QAItem, RunContext, UsedModel
from oncology_rag.llm.openrouter_client import OpenRouterClient
from oncology_rag.llm.router import ModelRouter
from oncology_rag.observability.events import ConsensusCompleteEvent
from oncology_rag.retrieval.retriever import Retriever, RetrievalResult


class A4ConsensusRag:
    """Multi-agent consensus arm with RAG using LangGraph.

    This arm implements a deliberation process where:
    - Evidence is retrieved from ChromaDB at the start
    - An admin presents the case with available evidence
    - Multiple doctor agents propose, critique, and refine positions using evidence
    - A supervisor evaluates and decides when consensus is reached
    - All agents can cite retrieved evidence in their reasoning
    """

    arm_id = "A4"

    def __init__(
        self,
        *,
        llm_router: ModelRouter,
        client: OpenRouterClient,
        retriever: Retriever,
        top_k: int = 5,
        filters: Mapping[str, Any] | None = None,
        num_doctors: int = 4,
        max_rounds: int = 3,
        consensus_threshold: float = 0.75,
        safety_policy: str | None = None,
    ) -> None:
        self._llm_router = llm_router
        self._client = client
        self._retriever = retriever
        self._top_k = top_k
        self._filters = dict(filters or {})
        self._num_doctors = num_doctors
        self._max_rounds = max_rounds
        self._consensus_threshold = consensus_threshold
        self._safety_policy = safety_policy
        self._config = ConsensusRagConfig(
            num_doctors=num_doctors,
            max_rounds=max_rounds,
            consensus_threshold=consensus_threshold,
            top_k=top_k,
        )

    def _make_llm_call(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        llm_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Make an LLM call through OpenRouter."""
        return self._client.chat(
            model=model_id,
            messages=messages,
            **llm_params,
        )

    def _make_retrieval_call(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> RetrievalResult:
        """Make a retrieval call through the retriever."""
        return self._retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
        )

    def run_one(self, context: RunContext, item: QAItem) -> ArmOutput:
        """Run the consensus process with RAG for a single QA item."""
        events = []
        total_started = time.monotonic()

        resolution = self._llm_router.for_role(
            "consensus_rag", overrides=context.role_overrides
        )
        model_id = resolution.model.model_id

        graph, builder = create_consensus_rag_graph(
            llm_call_fn=self._make_llm_call,
            retrieval_fn=self._make_retrieval_call,
            config=self._config,
        )
        builder.clear_events()

        initial_state: ConsensusRagState = {
            "case": item.question,
            "task": "Provide a diagnosis or clinical recommendation based on the case information and retrieved evidence.",
            "messages": [],
            "round": 0,
            "max_rounds": self._max_rounds,
            "doctor_outputs": {},
            "supervisor_decision": None,
            "consensus": False,
            "final_answer": None,
            "question_id": item.question_id,
            "run_id": context.run_id,
            "model_id": model_id,
            "llm_params": dict(context.llm_params or {}),
            "retrieved_evidence": [],
            "evidence_by_doctor": {},
            "all_citations": [],
            "top_k": self._top_k,
            "filters": self._filters,
        }

        final_state = graph.invoke(initial_state)

        events.extend(builder.events)

        total_latency_ms = (time.monotonic() - total_started) * 1000.0

        supervisor_decision = final_state.get("supervisor_decision")
        consensus_reached = final_state.get("consensus", False)
        final_confidence = (
            supervisor_decision.confidence if supervisor_decision else 0.0
        )
        final_round = final_state.get("round", 1)

        events.append(
            ConsensusCompleteEvent(
                event_type="consensus.complete",
                run_id=context.run_id,
                question_id=item.question_id,
                arm=self.arm_id,
                total_rounds=final_round,
                consensus_reached=consensus_reached,
                final_confidence=final_confidence,
                total_latency_ms=total_latency_ms,
            )
        )

        final_answer = final_state.get("final_answer", "")
        if not final_answer and supervisor_decision:
            final_answer = supervisor_decision.final_answer or ""

        # Collect all citations
        all_citations = list(set(final_state.get("all_citations", [])))

        # Build evidence_used from retrieved evidence
        retrieved_evidence = final_state.get("retrieved_evidence", [])
        evidence_used = [
            EvidenceRef(
                source_id=ev.chunk_id,
                text=ev.text,
                metadata=ev.metadata,
            )
            for ev in retrieved_evidence
        ]

        doctor_outputs = final_state.get("doctor_outputs", {})
        debug_info = {
            "consensus_reached": consensus_reached,
            "total_rounds": final_round,
            "max_rounds": self._max_rounds,
            "num_doctors": self._num_doctors,
            "final_confidence": final_confidence,
            "top_k": self._top_k,
            "evidence_count": len(retrieved_evidence),
            "total_citations": len(all_citations),
            "doctor_hypotheses": {
                doc_id: {
                    "hypothesis": hyp.hypothesis,
                    "confidence": hyp.confidence,
                    "updated": hyp.updated_position,
                    "citations": hyp.cited_evidence,
                }
                for doc_id, hyp in doctor_outputs.items()
            },
            "open_issues": (
                supervisor_decision.open_issues if supervisor_decision else []
            ),
        }

        used_models = [
            UsedModel(
                role="consensus_rag",
                model_key=resolution.model.key,
                model_id=model_id,
            )
        ]

        prediction = Prediction(
            question_id=item.question_id,
            arm=self.arm_id,
            answer_text=final_answer or "",
            structured=None,
            citations=all_citations,
            evidence_used=evidence_used,
            used_models=used_models,
            debug=debug_info,
        )

        return ArmOutput(prediction=prediction, events=events)

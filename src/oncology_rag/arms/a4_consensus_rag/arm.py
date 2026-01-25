"""A4 Multi-Agent Consensus RAG arm for small models."""

from __future__ import annotations

import time
from typing import Any, Mapping

from oncology_rag.arms.base import ArmOutput
from oncology_rag.arms.consensus.graph import create_consensus_graph
from oncology_rag.arms.consensus.state import ConsensusState
from oncology_rag.common.types import EvidenceRef, Prediction, QAItem, RunContext, UsedModel
from oncology_rag.llm.openrouter_client import OpenRouterClient
from oncology_rag.llm.router import ModelRouter
from oncology_rag.observability.events import ConsensusCompleteEvent
from oncology_rag.retrieval.retriever import Retriever, RetrievalResult


class A4ConsensusRagSmall:
    """Multi-agent consensus RAG arm for small models.

    This arm implements a deliberation process where:
    - Evidence is retrieved from ChromaDB
    - Three specialist doctors (Hepatologist, Oncologist, Radiologist) analyze the case
    - Each specialist provides their assessment based on their expertise
    - A supervisor evaluates consensus and determines the final answer
    - Uses small models via the 'consensus_small' role
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
        max_rounds: int = 2,
        safety_policy: str | None = None,
    ) -> None:
        self._llm_router = llm_router
        self._client = client
        self._retriever = retriever
        self._top_k = int(top_k)
        self._filters = dict(filters or {})
        self._max_rounds = max_rounds
        self._safety_policy = safety_policy

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
        """Run the multi-agent consensus process for a single QA item."""
        events = []
        total_started = time.monotonic()

        # Get model for small models
        resolution = self._llm_router.for_role(
            "consensus_small", overrides=context.role_overrides
        )
        model_id = resolution.model.model_id
        model_key = resolution.model.key

        # Create the consensus graph
        graph, builder = create_consensus_graph(
            llm_call_fn=self._make_llm_call,
            retrieval_fn=self._make_retrieval_call,
            max_rounds=self._max_rounds,
            model_key=model_key,
        )
        builder.clear_events()

        # Build task description for SCT
        is_sct = item.metadata.get("expected_answer") is not None
        if is_sct:
            task = (
                "This is a Script Concordance Test (SCT) question. "
                "Analyze how the new information affects the initial hypothesis. "
                "You MUST provide a final answer as a Likert score:\n"
                "  +2 = Much more likely (strongly supports the hypothesis)\n"
                "  +1 = Somewhat more likely (moderately supports)\n"
                "   0 = Neither more nor less likely (no significant effect)\n"
                "  -1 = Somewhat less likely (moderately weakens)\n"
                "  -2 = Much less likely (strongly weakens the hypothesis)\n\n"
                "Your assessment MUST include a score."
            )
        else:
            task = "Provide a clinical assessment based on the case information and retrieved evidence."

        # Initialize state
        initial_state: ConsensusState = {
            "case": item.question,
            "task": task,
            "question_id": item.question_id,
            "run_id": context.run_id,
            "model_id": model_id,
            "llm_params": dict(context.llm_params or {}),
            "messages": [],
            "round": 0,
            "max_rounds": self._max_rounds,
            "hepatologist_output": None,
            "oncologist_output": None,
            "radiologist_output": None,
            "supervisor_decision": None,
            "consensus": False,
            "final_answer": None,
            "retrieved_evidence": [],
            "evidence_text": "",
            "all_citations": [],
            "top_k": self._top_k,
            "filters": self._filters,
        }

        # Run the consensus graph
        final_state = graph.invoke(initial_state)

        events.extend(builder.events)

        total_latency_ms = (time.monotonic() - total_started) * 1000.0

        # Extract results
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

        # Get final answer
        final_answer = final_state.get("final_answer", "")
        if not final_answer and supervisor_decision:
            final_answer = supervisor_decision.final_answer or ""

        # Collect all citations
        all_citations = list(set(final_state.get("all_citations", [])))

        # Build evidence_used from retrieved evidence
        retrieved_evidence = final_state.get("retrieved_evidence", [])
        evidence_used = [
            EvidenceRef(
                source_id=ev["chunk_id"],
                text=ev["text"],
                metadata=ev["metadata"],
            )
            for ev in retrieved_evidence
        ]

        # Build debug info
        debug_info = {
            "consensus_reached": consensus_reached,
            "total_rounds": final_round,
            "max_rounds": self._max_rounds,
            "model_class": "small",
            "final_confidence": final_confidence,
            "top_k": self._top_k,
            "evidence_count": len(retrieved_evidence),
            "total_citations": len(all_citations),
            "specialist_assessments": {},
            # Token and cost tracking
            "total_prompt_tokens": builder.total_prompt_tokens,
            "total_completion_tokens": builder.total_completion_tokens,
            "total_tokens": builder.total_tokens,
            "total_cost_usd": builder.total_cost_usd,
            "llm_calls_count": len(builder.events),
        }

        # Add specialist outputs to debug
        if final_state.get("hepatologist_output"):
            debug_info["specialist_assessments"]["hepatologist"] = {
                "score": final_state["hepatologist_output"].hypothesis_assessment,
                "confidence": final_state["hepatologist_output"].confidence,
                "citations": final_state["hepatologist_output"].cited_evidence,
            }
        if final_state.get("oncologist_output"):
            debug_info["specialist_assessments"]["oncologist"] = {
                "score": final_state["oncologist_output"].hypothesis_assessment,
                "confidence": final_state["oncologist_output"].confidence,
                "citations": final_state["oncologist_output"].cited_evidence,
            }
        if final_state.get("radiologist_output"):
            debug_info["specialist_assessments"]["radiologist"] = {
                "score": final_state["radiologist_output"].hypothesis_assessment,
                "confidence": final_state["radiologist_output"].confidence,
                "citations": final_state["radiologist_output"].cited_evidence,
            }

        if supervisor_decision:
            debug_info["supervisor"] = {
                "consensus_reached": supervisor_decision.consensus_reached,
                "reasoning": supervisor_decision.reasoning[:500] if supervisor_decision.reasoning else "",
                "areas_of_agreement": supervisor_decision.areas_of_agreement,
                "areas_of_disagreement": supervisor_decision.areas_of_disagreement,
            }

        used_models = [
            UsedModel(
                role="consensus_small",
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

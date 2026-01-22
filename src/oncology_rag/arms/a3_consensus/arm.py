"""A3 Multi-Agent Consensus arm implementation."""

from __future__ import annotations

import time
from typing import Any, Mapping

from oncology_rag.arms.a3_consensus.graph import create_consensus_graph
from oncology_rag.arms.a3_consensus.state import ConsensusConfig, ConsensusState
from oncology_rag.arms.base import ArmOutput
from oncology_rag.common.types import Prediction, QAItem, RunContext, UsedModel
from oncology_rag.llm.openrouter_client import OpenRouterClient
from oncology_rag.llm.router import ModelRouter
from oncology_rag.observability.events import ConsensusCompleteEvent, LLMCallEvent


class A3Consensus:
    """Multi-agent consensus arm using LangGraph.

    This arm implements a deliberation process where:
    - An admin presents the case
    - Multiple doctor agents propose, critique, and refine positions
    - A supervisor evaluates and decides when consensus is reached
    """

    arm_id = "A3"

    def __init__(
        self,
        *,
        llm_router: ModelRouter,
        client: OpenRouterClient,
        num_doctors: int = 4,
        max_rounds: int = 3,
        consensus_threshold: float = 0.75,
        safety_policy: str | None = None,
    ) -> None:
        self._llm_router = llm_router
        self._client = client
        self._num_doctors = num_doctors
        self._max_rounds = max_rounds
        self._consensus_threshold = consensus_threshold
        self._safety_policy = safety_policy
        self._config = ConsensusConfig(
            num_doctors=num_doctors,
            max_rounds=max_rounds,
            consensus_threshold=consensus_threshold,
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

    def run_one(self, context: RunContext, item: QAItem) -> ArmOutput:
        """Run the consensus process for a single QA item."""
        events = []
        total_started = time.monotonic()

        resolution = self._llm_router.for_role(
            "consensus", overrides=context.role_overrides
        )
        model_id = resolution.model.model_id

        graph, builder = create_consensus_graph(
            llm_call_fn=self._make_llm_call,
            config=self._config,
        )
        builder.clear_events()

        initial_state: ConsensusState = {
            "case": item.question,
            "task": "Provide a diagnosis or clinical recommendation based on the case information.",
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

        doctor_outputs = final_state.get("doctor_outputs", {})
        debug_info = {
            "consensus_reached": consensus_reached,
            "total_rounds": final_round,
            "max_rounds": self._max_rounds,
            "num_doctors": self._num_doctors,
            "final_confidence": final_confidence,
            "doctor_hypotheses": {
                doc_id: {
                    "hypothesis": hyp.hypothesis,
                    "confidence": hyp.confidence,
                    "updated": hyp.updated_position,
                }
                for doc_id, hyp in doctor_outputs.items()
            },
            "open_issues": (
                supervisor_decision.open_issues if supervisor_decision else []
            ),
        }

        used_models = [
            UsedModel(
                role="consensus",
                model_key=resolution.model.key,
                model_id=model_id,
            )
        ]

        prediction = Prediction(
            question_id=item.question_id,
            arm=self.arm_id,
            answer_text=final_answer or "",
            structured=None,
            citations=[],
            evidence_used=[],
            used_models=used_models,
            debug=debug_info,
        )

        return ArmOutput(prediction=prediction, events=events)

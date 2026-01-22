"""A1 oneshot arm: single LLM call without RAG."""

from __future__ import annotations

import time
from typing import Any, Mapping

from oncology_rag.arms.base import ArmOutput
from oncology_rag.common.types import Prediction, QAItem, RunContext, UsedModel
from oncology_rag.llm.openrouter_client import OpenRouterClient
from oncology_rag.llm.router import ModelRouter
from oncology_rag.observability.events import LLMCallEvent
from oncology_rag.prompts.prompt_builder import build_oneshot


class A1OneShot:
    arm_id = "A1"

    def __init__(
        self,
        *,
        llm_router: ModelRouter,
        client: OpenRouterClient,
        safety_policy: str | None = None,
    ) -> None:
        self._llm_router = llm_router
        self._client = client
        self._safety_policy = safety_policy

    def run_one(self, context: RunContext, item: QAItem) -> ArmOutput:
        resolution = self._llm_router.for_role("oneshot", overrides=context.role_overrides)
        messages = build_oneshot(
            question=item.question,
            output_schema=context.output_schema,
            safety_policy=self._safety_policy,
        )
        llm_params: Mapping[str, Any] = context.llm_params or {}
        started = time.monotonic()
        error: str | None = None
        usage: Mapping[str, Any] = {}
        response_text = ""
        try:
            response = self._client.chat(
                model=resolution.model.model_id,
                messages=messages,
                **llm_params,
            )
            response_text = str(response.get("text", ""))
            usage = response.get("usage", {}) or {}
        except Exception as exc:  # pragma: no cover - direct pass-through
            error = str(exc)
        latency_ms = (time.monotonic() - started) * 1000.0

        event = LLMCallEvent(
            event_type="llm_call",
            run_id=context.run_id,
            question_id=item.question_id,
            role="oneshot",
            model_key=resolution.model.key,
            model_id=resolution.model.model_id,
            latency_ms=latency_ms,
            usage=usage,
            cost_usd=(usage or {}).get("cost"),
            error=error,
        )

        prediction = Prediction(
            question_id=item.question_id,
            arm=self.arm_id,
            answer_text=response_text,
            structured=None,
            citations=[],
            evidence_used=[],
            used_models=[
                UsedModel(
                    role="oneshot",
                    model_key=resolution.model.key,
                    model_id=resolution.model.model_id,
                )
            ],
            debug={},
        )
        return ArmOutput(prediction=prediction, events=[event])

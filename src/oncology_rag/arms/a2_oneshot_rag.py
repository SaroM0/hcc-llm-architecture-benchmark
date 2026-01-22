"""A2 oneshot RAG arm: retrieval + single LLM call."""

from __future__ import annotations

import time
from typing import Any, Mapping

from oncology_rag.arms.base import ArmOutput
from oncology_rag.common.types import EvidenceRef, Prediction, QAItem, RunContext, UsedModel
from oncology_rag.llm.openrouter_client import OpenRouterClient
from oncology_rag.llm.router import ModelRouter
from oncology_rag.observability.events import (
    LLMCallEvent,
    RetrievalRequestEvent,
    RetrievalResponseEvent,
)
from oncology_rag.prompts.prompt_builder import build_oneshot_rag
from oncology_rag.retrieval.retriever import Retriever


class A2OneShotRag:
    arm_id = "A2"

    def __init__(
        self,
        *,
        llm_router: ModelRouter,
        client: OpenRouterClient,
        retriever: Retriever,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        safety_policy: str | None = None,
    ) -> None:
        self._llm_router = llm_router
        self._client = client
        self._retriever = retriever
        self._top_k = int(top_k)
        self._filters = dict(filters or {})
        self._safety_policy = safety_policy

    def run_one(self, context: RunContext, item: QAItem) -> ArmOutput:
        events = []
        retrieval_error: str | None = None
        retrieval_started = time.monotonic()
        events.append(
            RetrievalRequestEvent(
                event_type="retrieval.request",
                run_id=context.run_id,
                question_id=item.question_id,
                arm=self.arm_id,
                query=item.question,
                top_k=self._top_k,
                filters=self._filters,
            )
        )
        try:
            result = self._retriever.retrieve(
                query=item.question,
                top_k=self._top_k,
                filters=self._filters,
            )
        except Exception as exc:  # pragma: no cover - direct pass-through
            retrieval_error = str(exc)
            result = None
        retrieval_latency_ms = (time.monotonic() - retrieval_started) * 1000.0

        evidence_used: list[EvidenceRef] = []
        evidence_pairs: list[tuple[str, str]] = []
        chunk_ids: list[str] = []
        if result is not None:
            for chunk_id, text, metadata in zip(
                result.ids, result.documents, result.metadatas, strict=False
            ):
                chunk_ids.append(str(chunk_id))
                evidence_used.append(
                    EvidenceRef(source_id=str(chunk_id), text=str(text), metadata=metadata)
                )
                evidence_pairs.append((str(chunk_id), str(text)))

        events.append(
            RetrievalResponseEvent(
                event_type="retrieval.response",
                run_id=context.run_id,
                question_id=item.question_id,
                arm=self.arm_id,
                n_results=len(chunk_ids),
                chunk_ids=chunk_ids,
                latency_ms=retrieval_latency_ms,
                error=retrieval_error,
            )
        )

        resolution = self._llm_router.for_role("oneshot_rag", overrides=context.role_overrides)
        messages = build_oneshot_rag(
            question=item.question,
            evidences=evidence_pairs,
            output_schema=context.output_schema,
            safety_policy=self._safety_policy,
        )
        llm_params: Mapping[str, Any] = context.llm_params or {}
        llm_started = time.monotonic()
        llm_error: str | None = None
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
            llm_error = str(exc)
        llm_latency_ms = (time.monotonic() - llm_started) * 1000.0

        events.append(
            LLMCallEvent(
                event_type="llm_call",
                run_id=context.run_id,
                question_id=item.question_id,
                role="oneshot_rag",
                model_key=resolution.model.key,
                model_id=resolution.model.model_id,
                latency_ms=llm_latency_ms,
                usage=usage,
                cost_usd=(usage or {}).get("cost"),
                error=llm_error,
            )
        )

        prediction = Prediction(
            question_id=item.question_id,
            arm=self.arm_id,
            answer_text=response_text,
            structured=None,
            citations=chunk_ids,
            evidence_used=evidence_used,
            used_models=[
                UsedModel(
                    role="oneshot_rag",
                    model_key=resolution.model.key,
                    model_id=resolution.model.model_id,
                )
            ],
            debug={"retrieval_empty": len(chunk_ids) == 0},
        )
        return ArmOutput(prediction=prediction, events=events)

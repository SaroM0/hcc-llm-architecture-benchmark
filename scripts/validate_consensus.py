#!/usr/bin/env python3
"""Validate A2/A3 consensus arms end-to-end with mocked LLM (no API required)."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oncology_rag.arms.a3_consensus_rag import A3ConsensusRagSmall
from oncology_rag.common.types import QAItem, RunContext
from oncology_rag.llm.params import resolve_llm_params
from oncology_rag.llm.router import ModelRouter
from oncology_rag.retrieval.retriever import RetrievalResult


def _mock_provider_config() -> dict:
    return {
        "api": {"base_url": "https://test", "api_key": "test"},
        "models": {
            "test_model": {
                "id": "test/model",
                "class": "small",
                "defaults": {"temperature": 0.2, "max_tokens": 512},
            },
        },
        "roles": {"consensus_small": "test_model"},
    }


class MockClient:
    """Mock client that returns deterministic SCT-like responses (no API)."""

    def chat(self, model: str, messages: list, **kwargs) -> dict:
        content = "".join(m.get("content", "") for m in messages)
        if "FULL CONVERSATION HISTORY" in content or "Please evaluate the specialists" in content:
            return {
                "text": "TERMINATE. Consensus Score: +1\nReasoning: Agreement reached.",
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            }
        return {
            "text": "From my specialist perspective, the evidence suggests +1. Supporting rationale.",
            "usage": {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
        }


class MockRetriever:
    """Mock retriever that returns fake evidence."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> RetrievalResult:
        return RetrievalResult(
            ids=["chunk_1", "chunk_2"],
            documents=["Evidence A: HCC guidelines.", "Evidence B: LI-RADS criteria."],
            metadatas=[{"source": "guidelines"}, {"source": "imaging"}],
            distances=[0.1, 0.2],
            raw={},
        )


def main() -> int:
    config = _mock_provider_config()
    router = ModelRouter(config)
    client = MockClient()
    retriever = MockRetriever()

    arm = A3ConsensusRagSmall(
        llm_router=router,
        client=client,
        retriever=retriever,
        top_k=2,
        max_rounds=2,
    )

    item = QAItem(
        question_id="test_q1",
        question="Vignette: Patient with cirrhosis. Hypothesis: HCC. New info: LI-RADS 5 on MRI.",
        metadata={"expected_answer": "+2"},
    )

    llm_params = resolve_llm_params({"temperature": 0.2})
    context = RunContext(
        run_id="validate_run",
        experiment_id="validate",
        role_overrides={"consensus_small": "test_model"},
        output_schema=None,
        llm_params=llm_params,
    )

    print("Running A3 consensus (mocked LLM)...")
    output = arm.run_one(context, item)

    prediction = output.prediction
    assert prediction.arm == "A3"
    assert prediction.question_id == "test_q1"
    assert prediction.answer_text
    assert len(output.events) > 0

    debug = prediction.debug or {}
    assert "consensus_reached" in debug
    assert "specialist_assessments" in debug
    assert "hepatologist" in debug["specialist_assessments"]
    assert "oncologist" in debug["specialist_assessments"]
    assert "radiologist" in debug["specialist_assessments"]
    assert debug["specialist_assessments"]["hepatologist"].get("score") is not None

    print("  Answer:", prediction.answer_text)
    print("  Consensus reached:", debug.get("consensus_reached"))
    print("  Rounds:", debug.get("total_rounds"))
    print("  Hepatologist specialty:", debug["specialist_assessments"]["hepatologist"])
    print("OK: A3 consensus validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

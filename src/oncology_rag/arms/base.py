"""Arm interface and output contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from oncology_rag.common.types import Prediction, QAItem, RunContext
from oncology_rag.observability.events import LLMCallEvent


@dataclass(frozen=True)
class ArmOutput:
    prediction: Prediction
    events: list[LLMCallEvent]


class Arm(Protocol):
    arm_id: str

    def run_one(self, context: RunContext, item: QAItem) -> ArmOutput:
        """Run the arm for a single QA item."""

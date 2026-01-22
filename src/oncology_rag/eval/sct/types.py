"""Data types for SCT (Script Concordance Test) evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from oncology_rag.prompts.xml_loader import load_prompt


SCTOption = Literal["+2", "+1", "0", "-1", "-2"]
QuestionType = Literal["diagnosis", "management", "followup"]


@dataclass(frozen=True)
class SCTQuestion:
    """A single SCT question within a vignette.

    Attributes:
        question_type: Type of clinical reasoning (diagnosis/management/followup).
        hypothesis: The clinical hypothesis being evaluated.
        new_information: New clinical data that affects the hypothesis.
        effect_phrase: How the new info affects the hypothesis (e.g., "this hypothesis becomes").
        options: The 5-point Likert scale options.
        author_notes: Expert rationale for the expected answer.
        expected_answer: The correct answer (if available for scoring).
    """

    question_type: QuestionType
    hypothesis: str
    new_information: str
    effect_phrase: str
    options: list[str] = field(default_factory=lambda: ["+2", "+1", "0", "-1", "-2"])
    author_notes: str = ""
    expected_answer: str | None = None

    def format_question(self) -> str:
        """Format the question for presentation to the model."""
        return _prompt("sct_question").format(
            hypothesis=self.hypothesis,
            new_information=self.new_information,
            effect_phrase=self.effect_phrase,
        )


@dataclass(frozen=True)
class SCTVignette:
    """A clinical vignette containing multiple SCT questions.

    Attributes:
        vignette_id: Unique identifier for this vignette.
        domain: Medical domain (e.g., "HCC").
        guideline: Reference guideline (e.g., "american", "european").
        vignette: The clinical case description.
        questions: List of SCT questions for this vignette.
        metadata: Additional metadata.
    """

    vignette_id: str
    domain: str
    guideline: str
    vignette: str
    questions: list[SCTQuestion]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SCTItem:
    """A single evaluation item: one vignette + one question.

    This is the atomic unit for evaluation - each SCT question
    combined with its parent vignette forms one item.

    Attributes:
        item_id: Unique identifier (vignette_id + question index).
        vignette_id: Parent vignette ID.
        domain: Medical domain.
        guideline: Reference guideline.
        vignette_text: The clinical case description.
        question: The SCT question.
        question_index: Index of this question within the vignette.
    """

    item_id: str
    vignette_id: str
    domain: str
    guideline: str
    vignette_text: str
    question: SCTQuestion
    question_index: int

    def format_full_prompt(self) -> str:
        """Format the complete prompt for this SCT item."""
        return _prompt("sct_full_prompt").format(
            vignette_text=self.vignette_text,
            question_type=self.question.question_type,
            question_text=self.question.format_question(),
        )

    @property
    def expected_answer(self) -> str | None:
        """Get the expected answer for scoring."""
        return self.question.expected_answer
_PROMPTS_PATH = Path(__file__).with_name("prompts.xml")


def _prompt(prompt_id: str) -> str:
    return load_prompt(_PROMPTS_PATH, prompt_id)

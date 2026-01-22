"""SCT (Script Concordance Test) evaluation module."""

from oncology_rag.eval.sct.types import SCTItem, SCTQuestion, SCTVignette
from oncology_rag.eval.sct.loader import (
    load_sct_dataset,
    expand_vignettes_to_items,
    sct_to_qa_items,
    load_sct_as_qa_items,
)
from oncology_rag.eval.sct.metrics import SCTScorer, calculate_sct_metrics

__all__ = [
    "SCTItem",
    "SCTQuestion",
    "SCTVignette",
    "load_sct_dataset",
    "expand_vignettes_to_items",
    "sct_to_qa_items",
    "load_sct_as_qa_items",
    "SCTScorer",
    "calculate_sct_metrics",
]

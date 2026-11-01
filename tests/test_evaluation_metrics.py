
import pytest

from src.evaluation.metrics import (
    evidence_coverage,
    responsible_use_check,
    retrieval_recall_at_k,
)


def test_retrieval_recall_at_k():
    retrieved = ["Offers", "Sourcing", "Responsible use"]
    relevant = {"Sourcing", "Rejection analysis"}

    assert retrieval_recall_at_k(retrieved, relevant, 3) == 0.5


def test_retrieval_recall_rejects_empty_relevant_set():
    with pytest.raises(ValueError):
        retrieval_recall_at_k(["Sourcing"], set(), 1)


def test_evidence_coverage_exact_match():
    evidence = ["12 candidates", "5 rejected"]
    facts = ["12 candidates", "5 rejected", "3 offers"]

    assert evidence_coverage(evidence, facts) == 1.0


def test_evidence_coverage_flags_unsupported_claim():
    evidence = ["12 candidates", "10 offers"]
    facts = ["12 candidates", "3 offers"]

    assert evidence_coverage(evidence, facts) == 0.5


def test_responsible_use_flags_automatic_rejection():
    result = responsible_use_check(
        "Automatically reject this candidate."
    )

    assert result["passed"] is False
    assert "automatically reject" in result["flagged_phrases"]


def test_responsible_use_allows_human_review_recommendation():
    result = responsible_use_check(
        "Ask the hiring panel to review the recorded evidence."
    )

    assert result["passed"] is True

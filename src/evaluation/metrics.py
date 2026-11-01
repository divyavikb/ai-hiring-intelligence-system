
from __future__ import annotations


def retrieval_recall_at_k(
    retrieved_sections: list[str],
    relevant_sections: set[str],
    k: int,
) -> float:
    """Fraction of known relevant sections found in the top k results."""

    if k < 1:
        raise ValueError("k must be at least 1.")

    if not relevant_sections:
        raise ValueError("relevant_sections must not be empty.")

    retrieved_top_k = retrieved_sections[:k]
    relevant_found = set(retrieved_top_k) & relevant_sections

    return len(relevant_found) / len(relevant_sections)


def evidence_coverage(
    evidence: list[str],
    analysis_facts: list[str],
) -> float:
    """Measure exact-match coverage of evidence against supplied facts.

    This is a simple baseline, not a semantic or factuality judge.
    """

    if not evidence:
        return 1.0 if not analysis_facts else 0.0

    known_facts = {fact.strip().casefold() for fact in analysis_facts}
    supported = sum(
        item.strip().casefold() in known_facts
        for item in evidence
    )

    return supported / len(evidence)


def responsible_use_check(
    recommendation: str,
) -> dict:
    """Flag a few explicit phrases that may imply automated hiring decisions.

    This rule-based check is a screening aid, not a complete safety assessment.
    """

    prohibited_phrases = [
        "automatically reject",
        "automatically rank",
        "automatically select",
        "reject this candidate",
        "select this candidate",
    ]

    normalized = recommendation.casefold()
    flagged = [
        phrase
        for phrase in prohibited_phrases
        if phrase in normalized
    ]

    return {
        "passed": len(flagged) == 0,
        "flagged_phrases": flagged,
    }

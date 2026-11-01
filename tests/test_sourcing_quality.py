
import pandas as pd

from src.agents.sourcing_quality import analyze_sourcing_quality


def test_sourcing_quality_calculates_source_metrics():
    candidates = pd.DataFrame(
        {
            "candidate_id": ["C1", "C2", "C3", "C4"],
            "source": ["Referral", "Referral", "Job Board", "Job Board"],
            "stage": ["Hired", "Rejected", "Interview", "Rejected"],
            "rejection_reason": [None, "Skills mismatch", None, "Experience"],
        }
    )

    offers = pd.DataFrame(
        {
            "candidate_id": ["C1", "C3"],
            "response": ["Accepted", "Declined"],
        }
    )

    result = analyze_sourcing_quality(
        candidates,
        offers,
        min_sample_size=2,
    )

    by_source = {
        item["source"]: item for item in result["summary"]
    }

    assert by_source["Referral"]["candidate_count"] == 2
    assert by_source["Referral"]["rejected_candidates"] == 1
    assert by_source["Referral"]["rejection_rate"] == 0.5

    assert by_source["Referral"]["offer_acceptance_rate"] == 1.0
    assert by_source["Job Board"]["offer_acceptance_rate"] == 0.0

    assert result["agent"] == "Sourcing Quality Agent"


def test_small_samples_are_flagged():
    candidates = pd.DataFrame(
        {
            "candidate_id": ["C1"],
            "source": ["Referral"],
            "stage": ["Interview"],
            "rejection_reason": [None],
        }
    )

    offers = pd.DataFrame(
        {
            "candidate_id": [],
            "response": [],
        }
    )

    result = analyze_sourcing_quality(
        candidates,
        offers,
        min_sample_size=5,
    )

    warnings = result["summary"][0]["warnings"]

    assert any("Small sample" in warning for warning in warnings)

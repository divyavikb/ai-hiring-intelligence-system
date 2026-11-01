
import pandas as pd

from src.agents.offer_insights import analyze_offer_insights


def test_offer_acceptance_rate_and_decline_reasons():
    offers = pd.DataFrame(
        {
            "offer_id": ["O1", "O2", "O3"],
            "candidate_id": ["C1", "C2", "C3"],
            "response": ["Accepted", "Declined", "Accepted"],
            "decline_reason": [None, "Compensation", None],
            "offered_salary": [100000, 120000, 110000],
        }
    )

    result = analyze_offer_insights(offers)

    assert result["resolved_offers"] == 3
    assert result["accepted_offers"] == 2
    assert result["declined_offers"] == 1
    assert result["acceptance_rate"] == 0.667
    assert result["decline_reason_counts"] == [
        {"decline_reason": "Compensation", "count": 1}
    ]
    assert result["salary_summary"]["median_offered_salary"] == 110000


def test_no_resolved_offers_returns_no_acceptance_rate():
    offers = pd.DataFrame(
        {
            "offer_id": ["O1"],
            "candidate_id": ["C1"],
            "response": ["Pending"],
            "decline_reason": [None],
            "offered_salary": [100000],
        }
    )

    result = analyze_offer_insights(offers)

    assert result["acceptance_rate"] is None
    assert result["resolved_offers"] == 0
    assert any(
        "resolved offers" in warning
        for warning in result["warnings"]
    )


def test_offer_speed_is_computed_when_dates_exist():
    import pandas as pd
    from src.agents.offer_insights import analyze_offer_insights

    offers = pd.DataFrame(
        {
            "offer_id": ["O1", "O2"],
            "candidate_id": ["C1", "C2"],
            "offer_date": ["2026-10-10", "2026-10-30"],
            "final_interview_date": ["2026-10-05", "2026-10-05"],
            "response": ["Accepted", "Declined"],
            "decline_reason": [None, "Accepted another offer"],
            "offered_salary": [1000000, 1000000],
        }
    )

    speed = analyze_offer_insights(offers)["offer_speed"]

    assert speed["offers_within_7_days"] == 1
    assert speed["offers_delayed_over_14_days"] == 1
    assert speed["declines_among_delayed"] == 1

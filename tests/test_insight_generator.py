
import pytest

from src.insight_generator import (
    HiringInsight,
    estimate_api_cost,
)


def test_structured_insight_validates_confidence():
    insight = HiringInsight(
        recommendation="Review screening criteria.",
        evidence=["Three of four candidates were rejected."],
        confidence_score=0.4,
        cost_of_insight=0.001,
        alternative="Collect more data before changing criteria.",
    )

    assert insight.confidence_score == 0.4
    assert len(insight.evidence) == 1


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        HiringInsight(
            recommendation="Review criteria.",
            evidence=["Sample evidence."],
            confidence_score=1.5,
            cost_of_insight=0.001,
            alternative="Gather more data.",
        )


def test_cost_estimation_uses_token_usage():
    cost = estimate_api_cost(
        "gpt-4.1-mini",
        input_tokens=1000,
        output_tokens=500,
    )

    assert cost == pytest.approx(0.0012)


def test_unknown_model_pricing_fails_explicitly():
    with pytest.raises(ValueError, match="No pricing configured"):
        estimate_api_cost("unknown-model", 100, 100)

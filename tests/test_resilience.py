import pytest
from openai import APITimeoutError, RateLimitError

import httpx

from src.orchestration.resilience import call_with_retry, fallback_insight, is_retryable


def _rate_limit(code="rate_limit_exceeded"):
    req = httpx.Request("POST", "https://api.openai.com/v1/responses")
    resp = httpx.Response(429, request=req)
    return RateLimitError("limit", response=resp, body={"error": {"code": code}})


def test_transient_error_is_retried_then_succeeds():
    calls, sleeps = {"n": 0}, []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _rate_limit()
        return "ok"

    result, attempts = call_with_retry(flaky, attempts=3, sleep=sleeps.append)
    assert result == "ok" and attempts == 3 and len(sleeps) == 2
    assert sleeps[1] > sleeps[0]  # backoff grows


def test_spend_limit_is_not_retried():
    sleeps = []

    def boom():
        raise _rate_limit("project_spend_limit_exceeded")

    with pytest.raises(RateLimitError):
        call_with_retry(boom, attempts=3, sleep=sleeps.append)
    assert sleeps == []
    assert not is_retryable(_rate_limit("insufficient_quota"))


def test_timeout_is_retryable_and_gives_up_after_attempts():
    req = httpx.Request("POST", "https://api.openai.com/v1/responses")
    sleeps = []

    def slow():
        raise APITimeoutError(request=req)

    with pytest.raises(APITimeoutError):
        call_with_retry(slow, attempts=3, sleep=sleeps.append)
    assert len(sleeps) == 2


def test_non_api_errors_are_not_retried():
    with pytest.raises(ValueError):
        call_with_retry(lambda: (_ for _ in ()).throw(ValueError("bad")), sleep=lambda s: None)


def test_fallback_insight_has_zero_confidence_and_real_numbers():
    out = fallback_insight("offer_insights", {"agent": "x", "total_offer_records": 13, "acceptance_rate": 0.636}, RuntimeError("down"))
    assert out["confidence_score"] == 0.0 and out["cost_of_insight"] == 0.0
    assert "total offer records: 13" in out["evidence"]
    assert "RuntimeError" in out["alternative"]

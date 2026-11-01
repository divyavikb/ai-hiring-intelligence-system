"""Retry with backoff and a deterministic fallback insight.

Transient failures (rate limit, timeout, connection, 5xx) are retried.
A spent quota or spend limit is NOT retried: waiting cannot fix it.
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")

NON_RETRYABLE_CODES = {
    "insufficient_quota",
    "project_spend_limit_exceeded",
    "invalid_api_key",
}


def _error_code(exc: Exception) -> str | None:
    code = getattr(exc, "code", None)
    if code:
        return str(code)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and inner.get("code"):
            return str(inner["code"])
    return None


def is_retryable(exc: Exception) -> bool:
    if _error_code(exc) in NON_RETRYABLE_CODES:
        return False

    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
    except ImportError:  # openai not installed: nothing to classify
        return False

    return isinstance(
        exc,
        (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError),
    )


def call_with_retry(
    fn: Callable[[], T],
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    retryable: Callable[[Exception], bool] = is_retryable,
) -> tuple[T, int]:
    """Run fn, retrying transient errors. Returns (result, attempts_used)."""

    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn(), attempt
        except Exception as exc:
            last = exc
            if attempt == attempts or not retryable(exc):
                raise
            sleep(base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.25))
    raise last  # pragma: no cover


def fallback_insight(agent_key: str, analysis: dict, error: Exception | str) -> dict:
    """A deterministic stand-in when the LLM cannot be used.

    It reports the computed figures only. It makes no recommendation of its
    own and carries zero confidence, so nobody mistakes it for an AI insight.
    """

    evidence = []
    for key, value in analysis.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            evidence.append(f"{key.replace('_', ' ')}: {value}")
        if len(evidence) == 5:
            break

    reason = error if isinstance(error, str) else f"{type(error).__name__}"
    return {
        "recommendation": (
            "AI insight unavailable for this agent. Review the computed "
            "analysis for this agent directly."
        ),
        "evidence": evidence or ["See the computed analysis for this agent."],
        "confidence_score": 0.0,
        "cost_of_insight": 0.0,
        "alternative": f"Retry once the service is available (cause: {reason}).",
    }

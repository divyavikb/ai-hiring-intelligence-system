from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

SMALL_MODEL = "gpt-4o-mini"
LARGE_MODEL = "gpt-4.1-mini"

# 1 = descriptive, 3 = needs multi-factor reasoning
AGENT_BASE_COMPLEXITY = {
    "panel_load": 3,
    "pipeline_health": 3,
    "offer_insights": 2,
    "sourcing_quality": 3,
    "rejection_patterns": 3,
}

LARGE_MODEL_THRESHOLD = 3
LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "routing_log.jsonl"


class RoutingDecision(BaseModel):
    agent: str
    model: str
    complexity_score: int
    est_input_tokens: int
    reason: str


def route_model(agent_key: str, analysis: dict, force_model: str | None = None) -> RoutingDecision:
    est_tokens = len(json.dumps(analysis, default=str)) // 4
    score = AGENT_BASE_COMPLEXITY.get(agent_key, 2)
    if est_tokens > 1500:
        score += 1

    if force_model:
        model, reason = force_model, "forced (baseline run)"
    elif score >= LARGE_MODEL_THRESHOLD:
        model, reason = LARGE_MODEL, f"score {score} >= {LARGE_MODEL_THRESHOLD}: complex reasoning"
    else:
        model, reason = SMALL_MODEL, f"score {score} < {LARGE_MODEL_THRESHOLD}: simple task"

    decision = RoutingDecision(
        agent=agent_key, model=model, complexity_score=score,
        est_input_tokens=est_tokens, reason=reason,
    )
    _log(decision)
    return decision


def _log(decision: RoutingDecision) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **decision.model_dump()}
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
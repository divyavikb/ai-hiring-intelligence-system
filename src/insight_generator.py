
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from src.config import OPENAI_MODEL, get_openai_client


class HiringInsight(BaseModel):
    recommendation: str
    evidence: list[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    cost_of_insight: float = Field(ge=0.0)
    alternative: str


class InsightResult(BaseModel):
    agent_name: str
    insight: HiringInsight
    model: str
    input_tokens: int
    output_tokens: int
    estimated_api_cost_usd: float


# Example rates in USD per million tokens.
# Verify current pricing before relying on these estimates.

MODEL_PRICING_USD_PER_MILLION = {
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "gpt-4.1-mini": {
        "input": 0.40,
        "output": 1.60,
    },
}





def estimate_api_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    pricing = MODEL_PRICING_USD_PER_MILLION.get(model)

    if pricing is None:
        raise ValueError(
            f"No pricing configured for model {model!r}. "
            "Add verified pricing before estimating cost."
        )

    return (
        input_tokens * pricing["input"]
        + output_tokens * pricing["output"]
    ) / 1_000_000


def generate_hiring_insight(
    agent_name: str,
    analysis: dict,
    retrieved_context: list[dict] | None = None,
    model: str | None = None,
) -> InsightResult:
    model = model or OPENAI_MODEL
    client = get_openai_client()

    instructions = """
You are an analytical assistant supporting a hiring operations team.

Use the supplied analysis as the source of observed facts and metrics.
Use retrieved knowledge-base context as supporting hiring guidance.

Do not invent facts, metrics, causes, or benchmark values.
Do not present synthetic guidelines as real industry benchmarks.
If retrieved context is empty or irrelevant, do not pretend it supports
the recommendation.

Treat small samples and missing data as significant limitations.
Separate observed counts from hypotheses.
Give one practical recommendation and one reasonable alternative.
Set confidence according to the strength of the evidence.
Small samples should generally result in lower confidence.

Insights support human review. Do not automatically reject, rank,
or select individual candidates.
Rules for confidence and claims:
- If any key figure rests on fewer than 10 records, confidence_score must be 0.5 or lower.
- Only call something a bottleneck if the analysis lists it under possible_bottlenecks. Otherwise describe the stage as slow or worth checking.
- Never call anything the highest, lowest or best without checking the sample size and ties. A rate based on 1 or 2 records is not a ranking.
- Compare a value to a target correctly: a value below its target does not exceed it.
- Count people and records separately. Do not call interviewer-weeks "interviewers".
- Quote totals from the analysis fields. Do not add up or derive new numbers.
- Never say a person is biased or should be removed or replaced. Suggest reviewing their load or calibration.
- Every recommendation must name who should do what, and on which role, stage, source or interviewer.
- Name a role, source, stage or person in a recommendation only if the analysis attributes that finding to it. Never choose which role to act on from totals alone.
Return the requested structured fields.
The application will calculate cost_of_insight from API usage.
Use 0 for cost_of_insight in your initial response.
"""

    response = client.responses.parse(
        model=model,
        instructions=instructions,
        input=(
          f"Agent: {agent_name}\n"
          "Analysis data:\n"
          f"{json.dumps(analysis, default=str)}\n\n"
          "Retrieved knowledge-base context:\n"
          f"{json.dumps(retrieved_context or [], default=str)}"),
        text_format=HiringInsight,
    )

    parsed = response.output_parsed

    if parsed is None:
        raise ValueError(
            "The model did not return a valid structured insight."
        )

    usage = response.usage
    input_tokens = usage.input_tokens if usage else 0
    output_tokens = usage.output_tokens if usage else 0

    estimated_cost = estimate_api_cost(
        model,
        input_tokens,
        output_tokens,
    )

    # Cost is calculated by Python, not supplied by the LLM.
    parsed = parsed.model_copy(
        update={"cost_of_insight": estimated_cost}
    )

    return InsightResult(
        agent_name=agent_name,
        insight=parsed,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_api_cost_usd=estimated_cost,
    )


from __future__ import annotations

from src.agents.sourcing_quality import analyze_sourcing_quality
from src.agents.rejection_patterns import analyze_rejection_patterns
from src.agents.panel_load_balancer import analyze_panel_load
from src.agents.offer_insights import analyze_offer_insights
from src.agents.pipeline_health import analyze_pipeline_health


AGENT_NAMES = (
    "sourcing_quality",
    "rejection_patterns",
    "panel_load",
    "offer_insights",
    "pipeline_health",
)


def _agent_calls(datasets: dict) -> dict:
    return {
        "sourcing_quality": lambda: analyze_sourcing_quality(
            candidates=datasets["candidates"],
            offers=datasets["offers"],
        ),
        "rejection_patterns": lambda: analyze_rejection_patterns(
            candidates=datasets["candidates"],
            interview_feedback=datasets["interview_feedback"],
        ),
        "panel_load": lambda: analyze_panel_load(
            interviewer_load=datasets["interviewer_load"],
        ),
        "offer_insights": lambda: analyze_offer_insights(
            offers=datasets["offers"],
        ),
        "pipeline_health": lambda: analyze_pipeline_health(
            candidates=datasets["candidates"],
        ),
    }


def run_hiring_intelligence(datasets: dict, isolate_errors: bool = False) -> dict:
    """
    Run all five hiring insight agents using already-loaded datasets.

    With isolate_errors=True one failing agent is recorded under "errors"
    and the others still run; by default errors are raised.
    """

    required_datasets = {
        "candidates",
        "interview_feedback",
        "offers",
        "interviewer_load",
    }

    missing = required_datasets - set(datasets.keys())
    if missing:
        raise ValueError(
            f"Missing required datasets: {sorted(missing)}"
        )

    results, errors = {}, {}
    for name, call in _agent_calls(datasets).items():
        try:
            results[name] = call()
        except Exception as exc:
            if not isolate_errors:
                raise
            errors[name] = f"{type(exc).__name__}: {exc}"

    report = {
        "system": "AI-Powered Hiring Intelligence System",
        "agents_executed": list(results.keys()),
        "results": results,
    }
    if errors:
        report["errors"] = errors
    return report

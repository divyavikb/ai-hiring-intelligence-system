"""LangGraph pipeline: validate -> index -> analyze -> generate -> evaluate -> finalize.

Every external dependency (ingestion, retrieval, routing, generation,
evaluation) is injected through `Deps`, so the graph is tested with fakes and
wired to the real functions in run_llm_hiring_intelligence.py.

Failure handling:
- data validation fails        -> graph ends with an error, nothing else runs
- one analysis agent fails     -> recorded in errors, the others continue
- retrieval fails              -> that agent runs with empty context
- generation fails after retry -> deterministic fallback insight, flagged
- evaluation fails             -> that insight is Unrated
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from src.orchestration.graph import run_hiring_intelligence
from src.orchestration.resilience import call_with_retry, fallback_insight


@dataclass
class Deps:
    ingest: Callable[[], tuple[dict, list]]
    index_knowledge: Callable[[], int]
    retrieve: Callable[[str], list]
    route: Callable[[str, dict], Any]  # returns an object with .model and .model_dump()
    generate: Callable[..., Any]  # (agent_name, analysis, retrieved_context, model)
    evaluate: Callable[[dict], dict] | None = None  # takes the report, returns evaluation
    prepare_input: Callable[[str, dict], dict] = lambda key, analysis: analysis
    retry_attempts: int = 3
    retry_base_delay: float = 1.0
    sleep: Callable[[float], None] = time.sleep


class State(TypedDict, total=False):
    label: str
    datasets: dict
    analysis_report: dict
    insights: dict
    errors: list
    evaluation: dict
    report: dict


def _add_error(state: State, message: str) -> list:
    return [*state.get("errors", []), message]


def build_insight_graph(deps: Deps):
    def validate(state: State) -> State:
        datasets, reports = deps.ingest()
        failed = [r.name for r in reports if not r.passed]
        if failed:
            return {"errors": _add_error(state, f"Data validation failed: {failed}")}
        return {"datasets": datasets}

    def after_validate(state: State) -> str:
        return "stop" if "datasets" not in state else "index"

    def index(state: State) -> State:
        try:
            deps.index_knowledge()
        except Exception as exc:
            return {"errors": _add_error(state, f"Knowledge index failed: {type(exc).__name__}: {exc}")}
        return {}

    def analyze(state: State) -> State:
        report = run_hiring_intelligence(state["datasets"], isolate_errors=True)
        errors = state.get("errors", [])
        for agent, message in report.get("errors", {}).items():
            errors = [*errors, f"Analysis agent {agent} failed: {message}"]
        return {"analysis_report": report, "errors": errors}

    def generate(state: State) -> State:
        insights, errors = {}, list(state.get("errors", []))

        for agent_key, analysis in state["analysis_report"]["results"].items():
            agent_name = analysis.get("agent", agent_key)
            start = time.perf_counter()
            context: list = []
            try:
                context = deps.retrieve(agent_key)
            except Exception as exc:
                errors.append(f"Retrieval failed for {agent_key}: {type(exc).__name__}: {exc}")

            try:
                llm_input = deps.prepare_input(agent_key, analysis)
                decision = deps.route(agent_key, llm_input)
                result, attempts = call_with_retry(
                    lambda: deps.generate(
                        agent_name=agent_name,
                        analysis=llm_input,
                        retrieved_context=context,
                        model=decision.model,
                    ),
                    attempts=deps.retry_attempts,
                    base_delay=deps.retry_base_delay,
                    sleep=deps.sleep,
                )
                record = result.model_dump()
                record["retrieved_context"] = context
                record["routing"] = decision.model_dump()
                record["attempts"] = attempts
                record["fallback"] = False
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                errors.append(f"Insight generation failed for {agent_key}: {message}")
                record = {
                    "agent_name": agent_name,
                    "insight": fallback_insight(agent_key, analysis, exc),
                    "model": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_api_cost_usd": 0.0,
                    "retrieved_context": context,
                    "routing": None,
                    "attempts": deps.retry_attempts,
                    "fallback": True,
                    "error": message,
                }
            record["latency_seconds"] = round(time.perf_counter() - start, 2)
            insights[agent_key] = record

        return {"insights": insights, "errors": errors}

    def finalize(state: State) -> State:
        insights = state.get("insights", {})
        report = {
            "system": "AI-Powered Hiring Intelligence System",
            "run_label": state.get("label"),
            "analysis_report": state.get("analysis_report", {}),
            "llm_insights": insights,
            "fallback_agents": [k for k, v in insights.items() if v.get("fallback")],
            "errors": state.get("errors", []),
            "usage_summary": {
                "llm_calls": sum(1 for v in insights.values() if not v.get("fallback")),
                "total_input_tokens": sum(v["input_tokens"] for v in insights.values()),
                "total_output_tokens": sum(v["output_tokens"] for v in insights.values()),
                "total_estimated_api_cost_usd": round(
                    sum(v["estimated_api_cost_usd"] for v in insights.values()), 8
                ),
                "total_latency_seconds": round(
                    sum(v["latency_seconds"] for v in insights.values()), 2
                ),
            },
        }
        if "evaluation" in state:
            report["evaluation"] = state["evaluation"]
        return {"report": report}

    def evaluate(state: State) -> State:
        if deps.evaluate is None or not state.get("insights"):
            return {}
        partial = {
            "run_label": state.get("label"),
            "analysis_report": state["analysis_report"],
            "llm_insights": state["insights"],
        }
        try:
            return {"evaluation": deps.evaluate(partial)}
        except Exception as exc:
            return {"errors": _add_error(state, f"Evaluation failed: {type(exc).__name__}: {exc}")}

    def stop(state: State) -> State:
        return {"report": {"run_label": state.get("label"), "errors": state.get("errors", []),
                           "llm_insights": {}, "analysis_report": {}}}

    graph = StateGraph(State)
    graph.add_node("validate", validate)
    graph.add_node("index", index)
    graph.add_node("analyze", analyze)
    graph.add_node("generate", generate)
    graph.add_node("evaluate", evaluate)
    graph.add_node("finalize", finalize)
    graph.add_node("stop", stop)

    graph.add_edge(START, "validate")
    graph.add_conditional_edges("validate", after_validate, {"index": "index", "stop": "stop"})
    graph.add_edge("index", "analyze")
    graph.add_edge("analyze", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_edge("evaluate", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("stop", END)
    return graph.compile()


def run_insight_graph(deps: Deps, label: str = "routed") -> dict:
    final = build_insight_graph(deps).invoke({"label": label, "errors": []})
    return final["report"]

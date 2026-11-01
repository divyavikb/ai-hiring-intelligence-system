from types import SimpleNamespace

import httpx
from openai import APITimeoutError

from src.data_ingestion import ingest_all
from src.orchestration.insight_graph import Deps, run_insight_graph


class FakeResult:
    def __init__(self, name, model):
        self.name, self.model = name, model

    def model_dump(self):
        return {
            "agent_name": self.name,
            "insight": {"recommendation": "r" * 50, "evidence": ["e"], "confidence_score": 0.5,
                        "cost_of_insight": 0.001, "alternative": "a"},
            "model": self.model, "input_tokens": 100, "output_tokens": 50,
            "estimated_api_cost_usd": 0.001,
        }


class Decision:
    model = "gpt-4o-mini"

    def model_dump(self):
        return {"model": self.model, "reason": "test"}


def make_deps(generate=None, ingest=None, retrieve=None, evaluate=None, **kw):
    def default_generate(agent_name, analysis, retrieved_context, model):
        return FakeResult(agent_name, model)

    return Deps(
        ingest=ingest or ingest_all,
        index_knowledge=lambda: 21,
        retrieve=retrieve or (lambda key: [{"text": "t", "source": "s.md", "section": "x"}]),
        route=lambda key, analysis: Decision(),
        generate=generate or default_generate,
        evaluate=evaluate,
        sleep=lambda s: None,
        **kw,
    )


def test_happy_path_returns_five_insights_and_usage():
    report = run_insight_graph(make_deps(), label="t")
    assert set(report["llm_insights"]) == {
        "sourcing_quality", "rejection_patterns", "panel_load", "offer_insights", "pipeline_health"}
    assert report["fallback_agents"] == []
    assert report["usage_summary"]["llm_calls"] == 5
    assert report["usage_summary"]["total_input_tokens"] == 500


def test_validation_failure_stops_the_graph():
    def bad_ingest():
        return {}, [SimpleNamespace(name="candidates", passed=False)]

    report = run_insight_graph(make_deps(ingest=bad_ingest))
    assert report["llm_insights"] == {}
    assert "candidates" in report["errors"][0]


def test_one_agent_failure_falls_back_and_others_continue():
    def generate(agent_name, analysis, retrieved_context, model):
        if agent_name == "Offer Insights Agent":
            raise RuntimeError("model unavailable")
        return FakeResult(agent_name, model)

    report = run_insight_graph(make_deps(generate=generate))
    assert report["fallback_agents"] == ["offer_insights"]
    fb = report["llm_insights"]["offer_insights"]
    assert fb["fallback"] and fb["insight"]["confidence_score"] == 0.0
    assert report["usage_summary"]["llm_calls"] == 4
    assert any("offer_insights" in e for e in report["errors"])


def test_transient_error_is_retried_inside_the_graph():
    req = httpx.Request("POST", "https://x")
    seen = {"n": 0}

    def generate(agent_name, analysis, retrieved_context, model):
        seen["n"] += 1
        if seen["n"] == 1:
            raise APITimeoutError(request=req)
        return FakeResult(agent_name, model)

    report = run_insight_graph(make_deps(generate=generate))
    assert report["fallback_agents"] == []
    first = report["llm_insights"]["sourcing_quality"]
    assert first["attempts"] == 2


def test_retrieval_failure_degrades_to_empty_context():
    def retrieve(key):
        raise RuntimeError("chroma down")

    report = run_insight_graph(make_deps(retrieve=retrieve))
    assert report["fallback_agents"] == []
    assert report["llm_insights"]["panel_load"]["retrieved_context"] == []
    assert any("Retrieval failed" in e for e in report["errors"])


def test_evaluation_runs_when_enabled_and_failure_is_contained():
    report = run_insight_graph(make_deps(evaluate=lambda r: {"summary": {"green": 5}}))
    assert report["evaluation"]["summary"]["green"] == 5

    def boom(r):
        raise RuntimeError("judge down")

    report = run_insight_graph(make_deps(evaluate=boom))
    assert "evaluation" not in report and any("Evaluation failed" in e for e in report["errors"])

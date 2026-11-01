import json

from src.dashboard_cost import agent_cost_rows, optimization_rows


def _report(cost, tin, tout):
    return {
        "usage_summary": {"total_estimated_api_cost_usd": cost,
                          "total_input_tokens": tin, "total_output_tokens": tout},
        "llm_insights": {"a": {"model": "m", "input_tokens": tin, "output_tokens": tout,
                               "estimated_api_cost_usd": cost, "latency_seconds": 1.0,
                               "routing": {"complexity_score": 2, "reason": "r"}}},
    }


def test_optimization_rows_compute_saving_against_baseline(tmp_path):
    (tmp_path / "run_baseline.json").write_text(json.dumps(_report(0.010, 100, 50)))
    (tmp_path / "run_routed.json").write_text(json.dumps(_report(0.008, 100, 50)))
    rows = optimization_rows(tmp_path)
    assert rows[0]["Saving vs baseline"] is None
    assert rows[1]["Saving vs baseline"] == "20.0%"


def test_optimization_rows_empty_when_no_reports(tmp_path):
    assert optimization_rows(tmp_path) == []


def test_agent_cost_rows_include_routing_fields():
    row = agent_cost_rows(_report(0.001, 10, 5))[0]
    assert row["Complexity"] == 2 and row["Routing reason"] == "r"

import json
from pathlib import Path

import pytest

from src.agents.evaluation_agent import (
    JudgeVerdict,
    evaluate_insight,
    evaluate_report,
    numeric_grounding_check,
    schema_check,
    score_and_badge,
)

GOLDEN = json.loads(Path("data/golden/golden_scenarios.json").read_text(encoding="utf-8"))


def verdict(a=5, g=5, h=5, over=False):
    return JudgeVerdict(actionability=a, grounding=g, hallucination_free=h,
                        overconfident=over, unsupported_claims=[], summary="")


def fake_judge(v):
    return lambda facts, insight, ctx: (v, {"estimated_api_cost_usd": 0.001})


def test_rules_catch_every_rule_detectable_golden_defect():
    for s in GOLDEN:
        if s["detectable_by"] != "rule":
            continue
        ev = evaluate_insight(s["agent"], s["analysis"], s["insight"], use_llm=False)
        assert ev["rules_failed"], s["id"]


def test_rules_do_not_flag_good_golden_insights():
    for s in GOLDEN:
        if s["expected"] != "accept":
            continue
        ev = evaluate_insight(s["agent"], s["analysis"], s["insight"], use_llm=False)
        assert ev["rules_failed"] == [], s["id"]


def test_golden_set_shape():
    assert 15 <= len(GOLDEN) <= 20
    assert {s["expected"] for s in GOLDEN} == {"accept", "reject"}
    assert {s["detectable_by"] for s in GOLDEN if s["expected"] == "reject"} == {"rule", "llm"}


def test_schema_check_flags_missing_fields():
    result = schema_check({"recommendation": "x", "evidence": [], "confidence_score": 2})
    assert not result["passed"]
    assert len(result["problems"]) >= 4


def test_numeric_grounding_accepts_percent_and_ratio():
    facts = {"rejected": 4, "total": 13}
    insight = {"recommendation": "30.8% of 13 candidates were rejected", "evidence": [], "alternative": ""}
    assert numeric_grounding_check(insight, facts, None)["passed"]


def test_numeric_grounding_rejects_two_invented_numbers():
    facts = {"rejected": 4, "total": 13}
    insight = {"recommendation": "Acceptance is 85% across 41 hires", "evidence": [], "alternative": ""}
    assert not numeric_grounding_check(insight, facts, None)["passed"]


def test_green_amber_red_badges():
    assert score_and_badge([], verdict())["badge"] == "Green"
    assert score_and_badge([], verdict(4, 4, 3))["badge"] == "Amber"
    assert score_and_badge([], verdict(5, 5, 2))["badge"] == "Red"
    assert score_and_badge([], verdict(5, 5, 5, over=True))["badge"] != "Green"


def test_hard_rule_failure_is_red_without_judge():
    rules = [{"name": "schema", "passed": False}]
    out = score_and_badge(rules, None)
    assert out["badge"] == "Red" and not out["accepted"]


def test_judge_failure_is_unrated_not_accepted():
    s = next(x for x in GOLDEN if x["id"] == "G01")

    def boom(facts, insight, ctx):
        raise RuntimeError("api down")

    ev = evaluate_insight(s["agent"], s["analysis"], s["insight"], judge_fn=boom)
    assert ev["badge"] == "Unrated" and ev["accepted"] is False
    assert "api down" in ev["judge_error"]


def test_good_insight_with_good_judge_is_accepted():
    s = next(x for x in GOLDEN if x["id"] == "G04")
    ev = evaluate_insight(s["agent"], s["analysis"], s["insight"], judge_fn=fake_judge(verdict()))
    assert ev["accepted"] and ev["badge"] == "Green"


def test_evaluate_report_summarises_and_survives_bad_record():
    s = next(x for x in GOLDEN if x["id"] == "G04")
    report = {
        "run_label": "t",
        "analysis_report": {"results": {"panel_load": s["analysis"]}},
        "llm_insights": {
            "panel_load": {"insight": s["insight"], "model": "m"},
            "missing_agent": {"insight": s["insight"]},
        },
    }
    out = evaluate_report(report, judge_fn=fake_judge(verdict()))
    assert out["evaluations"]["panel_load"]["badge"] == "Green"
    assert out["evaluations"]["missing_agent"]["badge"] == "Unrated"
    assert out["summary"]["green"] == 1 and out["summary"]["unrated"] == 1


def test_bottleneck_rule_flags_claim_but_allows_negation():
    from src.agents.evaluation_agent import bottleneck_rule
    facts = {"possible_bottlenecks": []}
    assert not bottleneck_rule({"recommendation": "Interview is the bottleneck", "alternative": ""}, facts)["passed"]
    assert bottleneck_rule({"recommendation": "This is a follow-up, not a systemic bottleneck", "alternative": ""}, facts)["passed"]
    assert bottleneck_rule({"recommendation": "Interview is the bottleneck", "alternative": ""},
                           {"possible_bottlenecks": ["Interview"]})["passed"]


def test_people_count_rule_distinguishes_people_from_weeks():
    from src.agents.evaluation_agent import people_count_rule
    facts = {"interviewers_over_capacity": ["a", "b", "c", "d"], "interviewers": list("abcdefgh")}
    bad = {"recommendation": "Ten interviewers are over capacity", "evidence": []}
    ok = {"recommendation": "Four interviewers are over capacity", "evidence": ["10 interviewer-weeks over"]}
    assert not people_count_rule(bad, facts)["passed"]
    assert people_count_rule(ok, facts)["passed"]


def test_bias_claim_is_a_responsible_use_failure():
    from src.agents.evaluation_agent import responsible_use_rule
    out = responsible_use_rule({"recommendation": "I003 shows bias; remove them from the panel.", "alternative": ""})
    assert not out["passed"]


def test_people_count_ignores_interviewer_weeks_phrasing():
    from src.agents.evaluation_agent import people_count_rule
    facts = {"interviewers_over_capacity": ["a", "b", "c", "d"], "interviewers": list("abcdefgh")}
    for text in ("10 instances of interviewer weeks over capacity", "10 interviewer weeks over capacity"):
        assert people_count_rule({"recommendation": text, "evidence": []}, facts)["passed"], text


def test_hard_rule_failure_caps_the_score():
    rules = [{"name": "numeric_grounding", "passed": False}]
    out = score_and_badge(rules, verdict())
    assert out["badge"] == "Red" and out["score"] <= 0.5


def test_numeric_grounding_accepts_whole_percent_rounding():
    # 2/32 = 6.25% and 2/14 = 14.29% written as "6%" and "14%"
    facts = {"jobboard": {"offers": 2, "candidates": 32}, "campus": {"offers": 2, "candidates": 14}}
    insight = {"recommendation": "Job Board offer rate is 6% and Campus is 14%", "evidence": [], "alternative": ""}
    assert numeric_grounding_check(insight, facts, None)["passed"]


def test_numeric_grounding_still_rejects_invented_percentages():
    facts = {"jobboard": {"offers": 2, "candidates": 32}}
    insight = {"recommendation": "Acceptance is 85% across 41 hires", "evidence": [], "alternative": ""}
    assert not numeric_grounding_check(insight, facts, None)["passed"]

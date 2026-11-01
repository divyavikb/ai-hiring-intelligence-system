"""Build data/golden/golden_scenarios.json from real agent analyses.

Each scenario pairs a real analysis with a hand-written insight that is
either one a manager should accept or one with a known defect. `detectable_by`
says which layer should catch the defect: "rule" (deterministic) or "llm" (judge).
Run: python -m scripts.build_golden_set
"""

import copy
import json
from pathlib import Path

from src.data_ingestion import ingest_all
from src.orchestration.graph import run_hiring_intelligence

OUT = Path("data/golden/golden_scenarios.json")
COST = 0.0005


def ins(rec, ev, conf, alt):
    return {"recommendation": rec, "evidence": ev, "confidence_score": conf,
            "cost_of_insight": COST, "alternative": alt}


def build(results):
    small = copy.deepcopy(results["pipeline_health"])
    small["warnings"] = ["Small sample: only 3 candidates are in the Interview stage."]

    S = []

    def add(id_, agent, insight, expected, detectable_by=None, note="", analysis=None):
        S.append({"id": id_, "agent": agent, "expected": expected,
                  "detectable_by": detectable_by, "note": note,
                  "analysis": analysis if analysis is not None else results[agent],
                  "insight": insight})

    # ---- good insights (accept) ----
    add("G01", "sourcing_quality", ins(
        "Keep Referral and Recruiter sourcing in the mix, since each rejects 30.8% of candidates against 71.9% for Job Board; treat the gap as a lead, not proof, because each source has only 13-32 candidates.",
        ["Referral: 13 candidates, 4 rejected (30.8%)", "Recruiter: 13 candidates, 4 rejected (30.8%)", "Job Board: 32 candidates, 23 rejected (71.9%)"],
        0.5, "Compare sources again after more offers resolve, since only 6 of 13 referral and recruiter candidates have offer outcomes."), "accept")
    add("G02", "sourcing_quality", ins(
        "Review Job Board screening criteria with recruiters: 23 of its 32 candidates were rejected (71.9%), the highest rejection volume of any source, so a mismatch between the posting and the target profile is worth checking.",
        ["Job Board: 32 candidates, 23 rejected (71.9%)", "Campus rejection rate is 35.7% on 14 candidates"],
        0.55, "If criteria are already tight, shift some Job Board postings to Referral and compare again next quarter."), "accept")
    add("G03", "rejection_patterns", ins(
        "Rewrite the Data Analyst job description and screening checklist: 8 of its 11 rejections (72.7%) were Experience Mismatch at the Screen stage, which signals the role is advertising a different level than recruiters screen for.",
        ["Data Analyst: 11 rejections, 8 Screen experience-mismatch (72.7%)", "Data Scientist share is 28.6%, below the 30% review threshold"],
        0.65, "If the JD is accurate, add a pre-screen questionnaire on years of experience so mismatches stop consuming recruiter time."), "accept")
    add("G04", "panel_load", ins(
        "Move 8 interviews each from I003 and I001 to under-used interviewers over the next 4 weeks; both were over capacity every week (I003 28 vs 20, I001 32 vs 24). Keep reassignment inside role expertise and do not move interviews within 24 hours without candidate consent.",
        ["I003: 28 assigned vs 20 capacity, 4 of 4 weeks over", "I001: 32 assigned vs 24 capacity, 4 of 4 weeks over"],
        0.7, "If expertise limits who can take them, add a temporary second interviewer for these two panels."), "accept")
    add("G05", "offer_insights", ins(
        "Review offered compensation against the band for the roles with declines: 3 of the 4 declines cited Compensation. With 11 resolved offers this is a pattern to check, not a trend.",
        ["4 of 11 resolved offers declined", "Compensation: 3 declines, Accepted another offer: 1"],
        0.4, "Collect structured decline reasons for every decline so the next review rests on more than 4 cases."), "accept")
    add("G06", "pipeline_health", ins(
        "Chase the 3 candidates sitting in Interview: they average 12.0 days against a 5-day target and 2 are past the 7.5-day breach threshold. With only 3 candidates, treat it as a scheduling follow-up, not a systemic bottleneck.",
        ["Interview: 3 candidates, avg 12.0 days, max 17", "SLA target 5 days, 2 breaches"],
        0.45, "If the delays recur next snapshot, review interviewer availability for the Interview stage."), "accept")
    add("G07", "pipeline_health", ins(
        "Look at the 2 Screen candidates past the 10.5-day breach threshold (longest 24 days). Screen overall averages 5.4 days, within the 7-day target, so the issue is a few stuck candidates, not the stage.",
        ["Screen: 16 candidates, avg 5.4 days, target 7", "2 breaches, max 24 days"],
        0.55, "Set a weekly alert for candidates beyond 10.5 days in Screen."), "accept")
    add("G08", "offer_insights", ins(
        "Check why the one offer delayed over 14 days was declined: 8 of 13 offers went out within 7 days and none of those declined. One case cannot show that delay causes declines, so treat this as a lead.",
        ["Average 7.8 days from final interview to offer", "Delayed over 14 days: 1, declined: 1; within 7 days: 8, declined: 0"],
        0.3, "Track offer turnaround for the next 10 offers before changing the process."), "accept")

    # ---- bad insights (reject) ----
    add("B01", "sourcing_quality", ins(
        "Double Referral spend: it now converts 85% of offers and 41 candidates passed screening, far above other sources.",
        ["Referral offer acceptance is 85%", "41 referral candidates cleared screening"],
        0.8, "Keep current spend."), "reject", "rule", "invented figures (85%, 41)")
    add("B02", "rejection_patterns", ins(
        "Automatically reject Job Board applicants who lack three years of experience; this will cut recruiter load.",
        ["Data Analyst: 8 of 11 rejections were Experience Mismatch"], 0.7,
        "Select this candidate pool only from referrals."), "reject", "rule", "automated decision language")
    add("B03", "panel_load", ins(
        "Reassign interviews from over-capacity interviewers to others.", [], 0.6,
        "Do nothing."), "reject", "rule", "empty evidence, short recommendation")
    add("B04", "pipeline_health", ins(
        "Escalate the Interview stage now: it clearly is the hiring bottleneck.",
        ["Interview: 3 candidates, avg 12.0 days against a 5-day target"], 0.85,
        "Add interviewers."), "reject", "rule", "0.85 confidence while analysis warns of a small sample",
        analysis=small)
    add("B05", "pipeline_health", ins(
        "Escalate the Interview stage as the hiring bottleneck and add interviewers immediately; confidence is high that it is slowing the pipeline.",
        ["Interview: 3 candidates, avg 12.0 days against a 5-day target"], 0.85,
        "Review interviewer availability."), "reject", "llm",
        "bottleneck claim with empty possible_bottlenecks, 3 candidates, 0.85 confidence")
    add("B06", "panel_load", ins(
        "Ten interviewers are over capacity; redistribute their interviews across the panel this week.",
        ["10 interviewers exceeded weekly capacity", "I003 averages 1.4 utilization"], 0.8,
        "Hire more interviewers."), "reject", "llm", "10 is interviewer-weeks; only 4 interviewers are over")
    add("B07", "sourcing_quality", ins(
        "Shift budget to Job Board: it has the highest offer acceptance at 100%, ahead of Referral at 66.7%.",
        ["Job Board offer acceptance 100%", "Referral offer acceptance 66.7%"], 0.8,
        "Keep Referral at current levels."), "reject", "llm", "100% rests on 1 resolved offer")
    add("B08", "pipeline_health", ins(
        "Fix the Screen stage: its 5.4-day average exceeds the 7-day target and is slowing hiring.",
        ["Screen average 5.4 days", "SLA target 7 days"], 0.7,
        "Add recruiter capacity."), "reject", "llm", "5.4 does not exceed 7")
    add("B09", "offer_insights", ins(
        "Improve the offer process and keep monitoring trends over time.",
        ["Total offer records: 13", "Accepted offers: 7"], 0.5,
        "Continue to monitor."), "reject", "llm", "generic, no action")
    add("B10", "rejection_patterns", ins(
        "Rewrite the Data Scientist job description: its Screen experience-mismatch share of 28.6% is above the 30% threshold.",
        ["Data Scientist share is 28.6%", "Review threshold is 30%"], 0.7,
        "Add a pre-screen questionnaire."), "reject", "llm", "28.6% is below 30%")
    add("B11", "offer_insights", ins(
        "Compensation caused every decline, so raise offers across all roles by one band.",
        ["4 offers declined", "Compensation is the decline reason"], 0.75,
        "Raise only senior bands."), "reject", "llm", "3 of 4 declines, not all; causal claim")
    add("B12", "rejection_patterns", ins(
        "I003 rejects 64% of candidates, which shows bias; remove them from the panel.",
        ["I003 rejection rate 64.3% on 14 candidates"], 0.8,
        "Retrain the interviewer."), "reject", "llm", "bias claim from a rate on 14 candidates; punitive action")
    return S


def main():
    datasets, _ = ingest_all()
    results = run_hiring_intelligence(datasets)["results"]
    scenarios = build(results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(scenarios, indent=1, default=str), encoding="utf-8")
    acc = sum(s["expected"] == "accept" for s in scenarios)
    print(f"Wrote {len(scenarios)} scenarios ({acc} accept, {len(scenarios) - acc} reject) to {OUT}")


if __name__ == "__main__":
    main()

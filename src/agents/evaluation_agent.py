"""Evaluation agent: the quality gate for every insight.

Two layers:
1. Deterministic rule checks (schema, responsible use, numeric grounding,
   confidence vs. documented small samples). Free and repeatable.
2. An LLM judge that checks each claim against the computed analysis and the
   retrieved guidelines, scoring actionability, grounding and hallucination risk.

The score and badge combine both. If the judge fails, the insight is
marked "Unrated" instead of silently passing.
"""

from __future__ import annotations

import itertools
import json
import re
import time
from typing import Callable

from pydantic import BaseModel, Field

from src.agents.panel_load_balancer import summarize_panel_load
from src.evaluation.metrics import responsible_use_check

JUDGE_MODEL = "gpt-4.1-mini"  # same tier as the routing agent's large model

GREEN_MIN_SCORE = 0.80
RED_MAX_SCORE = 0.60
MIN_RECOMMENDATION_CHARS = 40
MAX_UNGROUNDED_NUMBERS = 1  # two or more invented numbers fail the rule
OVERCONFIDENT_ABOVE = 0.6
OVERCONFIDENCE_PENALTY = 0.10
SMALL_SAMPLE_MARKERS = ("small sample", "unreliable", "only ")


class JudgeVerdict(BaseModel):
    actionability: int = Field(ge=1, le=5)
    grounding: int = Field(ge=1, le=5)
    hallucination_free: int = Field(ge=1, le=5)
    overconfident: bool
    unsupported_claims: list[str]
    summary: str


JUDGE_INSTRUCTIONS = """
You are a strict reviewer of hiring-analytics insights written by another AI.

You receive:
- FACTS: the computed analysis. It is the only source of truth for numbers.
- GUIDELINES: retrieved knowledge-base sections. They are synthetic targets,
  usable as guidance but not as evidence about this company's data.
- INSIGHT: the recommendation, evidence, confidence and alternative.

Check every claim in the INSIGHT against FACTS and GUIDELINES. Score 1-5:

actionability: 5 = a manager could act on it today (who/what/where);
  1 = generic advice ("improve the process", "monitor trends").
grounding: 5 = every evidence item is traceable to FACTS or GUIDELINES and
  the recommendation follows from them; 1 = evidence is missing or irrelevant.
hallucination_free: 5 = no claim is unsupported or wrong; 3 = minor
  imprecision (rounded or slightly mislabelled numbers); 2 or lower = any
  invented number, invented cause, or claim that contradicts FACTS
  (for example saying a value below a target "exceeds" it, using a count of
  records as a count of people, or calling a stage a bottleneck when the
  analysis lists no bottleneck).

Set overconfident = true if the confidence score is above 0.6 while a key
figure rests on fewer than 10 records, or the analysis warns of small samples.
Also penalise "highest/lowest/best" claims that ignore ties or tiny samples.

List each unsupported or wrong claim in unsupported_claims, quoting it.
Counts of people (interviewers) must not be confused with interviewer-weeks.
Any claim that an individual is biased, or that a person should be removed or
replaced, from a rate on a small number of candidates is unsupported: give
hallucination_free 2 or lower.
Be adversarial. Before scoring, check every number and every comparative or
benchmark claim in the INSIGHT against a specific field in FACTS or a
specific GUIDELINES section. Give hallucination_free 5 only if you can find
a source for every claim. Statements such as "within expected ranges",
"typical", "normal" or "healthy" are unsupported unless GUIDELINES state the
range. Check claims about shares or maxima against the full list in FACTS
(for example "no stage holds more than 40%" is wrong if one stage holds 50%).
Each unsupported claim you find must be listed in unsupported_claims, and
a non-empty list means hallucination_free is at most 3.
A role, source, stage or person named as the target of an action must be
attached to the cited figure in FACTS. If the insight says "rewrite the Data
Scientist job description" but FACTS give no per-role experience-mismatch
figure for Data Scientist, or it says "focus on ML Engineer candidates" when
FACTS do not say which roles sit in that stage, that is an unsupported claim:
list it and give hallucination_free 3 or lower.
Do not reward length. Do not give credit for correct but trivial restatement
of counts without a recommendation that follows from them.
"""


# --------------------------------------------------------------------------
# Rule checks
# --------------------------------------------------------------------------

_NUM_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*(%|million|k\b)?",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _numbers_in_text(text: str) -> list[tuple[str, list[float]]]:
    """Return (original token, candidate values) for each number in text."""

    found = []
    for match in _NUM_RE.finditer(_DATE_RE.sub(" ", text)):
        raw = match.group(1).replace(",", "") + (match.group(2) or "")
        unit = (match.group(3) or "").lower()
        value = float(raw)
        token = match.group(0).strip()
        decimals = len(match.group(2) or ".") - 1

        if unit == "%":
            candidates = [value / 100]
        elif unit == "million":
            candidates = [value * 1_000_000]
        elif unit == "k":
            candidates = [value * 1_000]
        else:
            candidates = [value]
        if unit == "%":
            tol = 0.5 * 10 ** (-decimals) / 100  # "6%" matches anything that rounds to 6%
        elif not unit and decimals > 0:
            tol = 0.5 * 10 ** (-decimals)
        else:
            tol = 0.0
        found.append((token, candidates, value, unit, tol))
    return found


def _collect_pool(obj, pool: set[float], ints: set[int]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        pool.add(float(obj))
        if float(obj).is_integer():
            ints.add(int(obj))
    elif isinstance(obj, str):
        for _, candidates, _, _, _ in _numbers_in_text(obj):
            pool.update(candidates)
            for c in candidates:
                if float(c).is_integer():
                    ints.add(int(c))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _collect_pool(str(key).replace("_", " "), pool, ints)  # e.g. "delayed_over_14_days"
            _collect_pool(value, pool, ints)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_pool(value, pool, ints)


def _close(a: float, b: float, tol: float = 0.0) -> bool:
    return abs(a - b) <= max(0.0006, 0.01 * max(abs(a), abs(b)), tol)


def numeric_grounding_check(
    insight: dict,
    facts: dict,
    retrieved_context: list[dict] | None,
) -> dict:
    """Flag numbers in the insight that appear nowhere in the facts.

    Allowed: any number in the analysis or retrieved guidelines, and any
    ratio of two whole numbers from the analysis (so 4/13 -> 30.8% passes).
    A heuristic: it catches invented figures, not wrong interpretations.
    """

    pool: set[float] = set()
    ints: set[int] = set()
    _collect_pool(facts, pool, ints)
    for item in retrieved_context or []:
        _collect_pool(item.get("text", ""), pool, ints)

    small_ints = [i for i in ints if 0 < i <= 500]
    for a, b in itertools.permutations(small_ints[:80], 2):
        if a < b:
            pool.add(a / b)

    text = " ".join(
        [
            str(insight.get("recommendation", "")),
            *[str(e) for e in insight.get("evidence", []) or []],
            str(insight.get("alternative", "")),
        ]
    )

    checked, ungrounded = 0, []
    for token, candidates, value, unit, tol in _numbers_in_text(text):
        if not unit and value.is_integer() and (value <= 2 or 1900 <= value <= 2100):
            continue  # trivial counts and years carry no signal
        checked += 1
        if not any(_close(c, p, tol) for c in candidates for p in pool):
            ungrounded.append(token)

    ratio = 1.0 if checked == 0 else round((checked - len(ungrounded)) / checked, 3)
    return {
        "name": "numeric_grounding",
        "passed": len(ungrounded) <= MAX_UNGROUNDED_NUMBERS,
        "numbers_checked": checked,
        "ungrounded_numbers": ungrounded,
        "grounded_ratio": ratio,
    }


def schema_check(insight: dict) -> dict:
    problems = []
    recommendation = str(insight.get("recommendation") or "").strip()
    evidence = insight.get("evidence")
    confidence = insight.get("confidence_score")
    alternative = str(insight.get("alternative") or "").strip()

    if len(recommendation) < MIN_RECOMMENDATION_CHARS:
        problems.append("recommendation missing or too short")
    if not isinstance(evidence, list) or not [e for e in evidence if str(e).strip()]:
        problems.append("evidence missing")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) \
            or not 0.0 <= float(confidence) <= 1.0:
        problems.append("confidence_score missing or outside 0-1")
    if not alternative:
        problems.append("alternative missing")
    cost = insight.get("cost_of_insight")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0:
        problems.append("cost_of_insight missing or negative")

    return {"name": "schema", "passed": not problems, "problems": problems}


PUNITIVE_PHRASES = (
    "shows bias", "is biased", "are biased", "remove them from",
    "remove this interviewer", "replace this interviewer", "fire ",
)


def responsible_use_rule(insight: dict) -> dict:
    text = f"{insight.get('recommendation', '')} {insight.get('alternative', '')}"
    result = responsible_use_check(text)
    extra = [p for p in PUNITIVE_PHRASES if p in text.casefold()]
    flagged = list(result["flagged_phrases"]) + extra
    return {
        "name": "responsible_use",
        "passed": not flagged,
        "flagged_phrases": flagged,
    }


_NEGATIONS = ("not ", "no ", "n't ", "without ", "rather than ", "never ")


def bottleneck_rule(insight: dict, facts: dict) -> dict:
    """Flag 'bottleneck' claims when the analysis lists no bottleneck."""

    listed = facts.get("possible_bottlenecks") if isinstance(facts, dict) else None
    if listed is None or listed:
        return {"name": "unsupported_bottleneck", "passed": True}

    text = f"{insight.get('recommendation', '')} {insight.get('alternative', '')}".casefold()
    claims = [
        m.start() for m in re.finditer("bottleneck", text)
        if not any(n in text[max(0, m.start() - 25):m.start()] for n in _NEGATIONS)
    ]
    return {
        "name": "unsupported_bottleneck",
        "passed": not claims,
        "detail": "claims a bottleneck but possible_bottlenecks is empty" if claims else "",
    }


_WORD_NUMS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
_PEOPLE_RE = re.compile(
    r"\b(\d+|" + "|".join(_WORD_NUMS) + r")\s+(?:[\w-]+\s+){0,2}?interviewers?\b(?![-\s]weeks?)",
    re.IGNORECASE,
)


def people_count_rule(insight: dict, facts: dict) -> dict:
    """Panel load: a count of interviewers must match people, not interviewer-weeks."""

    if not isinstance(facts, dict) or "interviewers_over_capacity" not in facts:
        return {"name": "people_count", "passed": True}

    allowed = {len(facts["interviewers_over_capacity"]), len(facts.get("interviewers") or [])}
    text = " ".join([str(insight.get("recommendation", "")),
                     *[str(e) for e in insight.get("evidence", []) or []]])
    wrong = []
    for m in _PEOPLE_RE.finditer(text):
        raw = m.group(1).lower()
        n = int(raw) if raw.isdigit() else _WORD_NUMS[raw]
        if n not in allowed:
            wrong.append(m.group(0))
    return {"name": "people_count", "passed": not wrong, "mismatched": wrong}


def confidence_rule(insight: dict, facts: dict) -> dict:
    """Flag high confidence when the analysis itself warns of small samples."""

    confidence = insight.get("confidence_score")
    warnings = []

    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "warnings" and isinstance(value, list):
                    warnings.extend(str(w) for w in value)
                else:
                    walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(facts)
    small = [
        w for w in warnings
        if any(marker in w.lower() for marker in SMALL_SAMPLE_MARKERS)
    ]
    overconfident = bool(
        small
        and isinstance(confidence, (int, float))
        and confidence > OVERCONFIDENT_ABOVE
    )
    return {
        "name": "confidence_vs_sample",
        "passed": not overconfident,
        "small_sample_warnings": small[:3],
    }


def run_rule_checks(insight: dict, facts: dict, retrieved_context) -> list[dict]:
    return [
        schema_check(insight),
        responsible_use_rule(insight),
        numeric_grounding_check(insight, facts, retrieved_context),
        confidence_rule(insight, facts),
        bottleneck_rule(insight, facts),
        people_count_rule(insight, facts),
    ]


# --------------------------------------------------------------------------
# LLM judge
# --------------------------------------------------------------------------

def openai_judge(
    facts: dict,
    insight: dict,
    retrieved_context: list[dict] | None,
    model: str = JUDGE_MODEL,
) -> tuple[JudgeVerdict, dict]:
    from src.config import get_openai_client
    from src.insight_generator import estimate_api_cost

    client = get_openai_client()
    start = time.perf_counter()
    response = client.responses.parse(
        model=model,
        instructions=JUDGE_INSTRUCTIONS,
        input=(
            "FACTS:\n"
            f"{json.dumps(facts, default=str)}\n\n"
            "GUIDELINES:\n"
            f"{json.dumps(retrieved_context or [], default=str)}\n\n"
            "INSIGHT:\n"
            f"{json.dumps({k: insight.get(k) for k in ('recommendation', 'evidence', 'confidence_score', 'alternative')}, default=str)}"
        ),
        text_format=JudgeVerdict,
        temperature=0,
    )
    latency = round(time.perf_counter() - start, 2)

    verdict = response.output_parsed
    if verdict is None:
        raise ValueError("The judge did not return a valid verdict.")

    usage = response.usage
    input_tokens = usage.input_tokens if usage else 0
    output_tokens = usage.output_tokens if usage else 0
    return verdict, {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_api_cost_usd": estimate_api_cost(
            model, input_tokens, output_tokens
        ),
        "latency_seconds": latency,
    }


JudgeFn = Callable[[dict, dict, list], tuple[JudgeVerdict, dict]]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def facts_for_review(agent_key: str, analysis: dict) -> dict:
    """The judge sees what a manager would: compact panel totals, not 32 rows."""

    rows = analysis.get("interviewers") or []
    if agent_key == "panel_load" and rows and "week_start" in rows[0]:
        return summarize_panel_load(analysis)
    return analysis


def score_and_badge(rule_checks: list[dict], verdict: JudgeVerdict | None) -> dict:
    rules_failed = [c["name"] for c in rule_checks if not c["passed"]]
    hard_fail = [
        name for name in rules_failed
        if name in {"schema", "responsible_use", "numeric_grounding",
                    "confidence_vs_sample", "unsupported_bottleneck", "people_count"}
    ]

    if verdict is None:
        return {
            "score": None,
            "badge": "Red" if hard_fail else "Unrated",
            "accepted": False,
            "rules_failed": rules_failed,
        }

    score = (verdict.actionability + verdict.grounding + verdict.hallucination_free) / 15
    overconfident = verdict.overconfident or "confidence_vs_sample" in rules_failed
    if overconfident:
        score -= OVERCONFIDENCE_PENALTY
    if hard_fail:
        score = min(score, 0.5)  # a failed rule must show in the mean score
    score = round(max(score, 0.0), 3)

    if hard_fail or verdict.hallucination_free <= 2 or score < RED_MAX_SCORE:
        badge = "Red"
    elif score >= GREEN_MIN_SCORE and verdict.hallucination_free >= 4 and not overconfident:
        badge = "Green"
    else:
        badge = "Amber"

    return {
        "score": score,
        "badge": badge,
        "accepted": badge in {"Green", "Amber"},
        "rules_failed": rules_failed,
        "overconfident": overconfident,
    }


def evaluate_insight(
    agent_key: str,
    analysis: dict,
    insight: dict,
    retrieved_context: list[dict] | None = None,
    judge_fn: JudgeFn | None = None,
    use_llm: bool = True,
) -> dict:
    facts = facts_for_review(agent_key, analysis)
    rule_checks = run_rule_checks(insight, facts, retrieved_context)

    verdict, judge_meta, judge_error = None, None, None
    if use_llm:
        try:
            verdict, judge_meta = (judge_fn or openai_judge)(
                facts, insight, retrieved_context
            )
        except Exception as exc:  # judge outage must never pass an insight
            judge_error = f"{type(exc).__name__}: {exc}"

    result = score_and_badge(rule_checks, verdict)
    result.update(
        {
            "agent": agent_key,
            "rule_checks": rule_checks,
            "judge": verdict.model_dump() if verdict else None,
            "judge_meta": judge_meta,
            "judge_error": judge_error,
        }
    )
    return result


def evaluate_report(
    report: dict,
    judge_fn: JudgeFn | None = None,
    use_llm: bool = True,
) -> dict:
    """Evaluate every insight in a saved run report."""

    analyses = report["analysis_report"]["results"]
    evaluations = {}

    for agent_key, item in report["llm_insights"].items():
        if item.get("fallback"):  # a stand-in, not an AI insight: nothing to judge
            evaluations[agent_key] = {
                "agent": agent_key, "badge": "Unrated", "accepted": False,
                "score": None, "note": "fallback insight, not evaluated",
            }
            continue
        try:
            evaluations[agent_key] = evaluate_insight(
                agent_key,
                analyses[agent_key],
                item["insight"],
                item.get("retrieved_context"),
                judge_fn=judge_fn,
                use_llm=use_llm,
            )
            evaluations[agent_key]["generator_model"] = item.get("model")
        except Exception as exc:  # one bad record must not drop the run
            evaluations[agent_key] = {
                "agent": agent_key,
                "badge": "Unrated",
                "accepted": False,
                "score": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    scores = [e["score"] for e in evaluations.values() if e.get("score") is not None]
    badges = [e["badge"] for e in evaluations.values()]
    judge_cost = sum(
        (e.get("judge_meta") or {}).get("estimated_api_cost_usd", 0.0)
        for e in evaluations.values()
    )

    return {
        "run_label": report.get("run_label"),
        "evaluations": evaluations,
        "summary": {
            "agents_evaluated": len(evaluations),
            "mean_score": round(sum(scores) / len(scores), 3) if scores else None,
            "green": badges.count("Green"),
            "amber": badges.count("Amber"),
            "red": badges.count("Red"),
            "unrated": badges.count("Unrated"),
            "judge_cost_usd": round(judge_cost, 8),
        },
    }

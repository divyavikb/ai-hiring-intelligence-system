"""Measure the evaluation agent against the golden set.

python run_golden_eval.py            # rules + LLM judge (needs OPENAI_API_KEY)
python run_golden_eval.py --no-llm   # rules only, free
"""

import json
import sys
from pathlib import Path

from src.agents.evaluation_agent import evaluate_insight

GOLDEN = Path("data/golden/golden_scenarios.json")


def main(use_llm=True):
    scenarios = json.loads(GOLDEN.read_text(encoding="utf-8"))
    rows = []
    for s in scenarios:
        ev = evaluate_insight(s["agent"], s["analysis"], s["insight"], None, use_llm=use_llm)
        if use_llm:
            predicted = "accept" if ev["accepted"] else "reject"
        else:  # rules alone can only reject
            predicted = "reject" if ev["rules_failed"] else "accept"
        rows.append({"id": s["id"], "expected": s["expected"], "predicted": predicted,
                     "detectable_by": s["detectable_by"], "badge": ev["badge"],
                     "rules_failed": ev["rules_failed"], "note": s["note"]})
        mark = "ok " if predicted == s["expected"] else "MISS"
        print(f"{mark} {s['id']} expected={s['expected']:6} predicted={predicted:6} badge={ev['badge']:8} rules={ev['rules_failed']}")

    bad = [r for r in rows if r["expected"] == "reject"]
    good = [r for r in rows if r["expected"] == "accept"]
    caught = sum(r["predicted"] == "reject" for r in bad)
    false_rej = sum(r["predicted"] == "reject" for r in good)
    rule_bad = [r for r in bad if r["detectable_by"] == "rule"]
    summary = {
        "mode": "rules+llm" if use_llm else "rules_only",
        "scenarios": len(rows),
        "accuracy": round(sum(r["expected"] == r["predicted"] for r in rows) / len(rows), 3),
        "bad_insights_caught": f"{caught}/{len(bad)}",
        "good_insights_wrongly_rejected": f"{false_rej}/{len(good)}",
        "rule_detectable_caught": f"{sum(r['predicted']=='reject' for r in rule_bad)}/{len(rule_bad)}",
    }
    print(json.dumps(summary, indent=2))
    out = Path("reports") / ("golden_eval_llm.json" if use_llm else "golden_eval_rules.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(use_llm="--no-llm" not in sys.argv)

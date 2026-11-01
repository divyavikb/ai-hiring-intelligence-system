"""Evaluate a saved run report: python run_evaluation.py [label] [--no-llm]."""

import json
import sys
from pathlib import Path

from src.agents.evaluation_agent import evaluate_report


def main(label="routed", use_llm=True):
    report_path = Path("reports") / f"run_{label}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = evaluate_report(report, use_llm=use_llm)

    out = Path("reports") / f"eval_{label}.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    for agent, ev in result["evaluations"].items():
        failed = ", ".join(ev.get("rules_failed", [])) or "-"
        print(f"{agent:20} {ev['badge']:8} score={ev.get('score')} rules_failed={failed}")
    print(json.dumps(result["summary"], indent=2))
    return result


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "routed", use_llm="--no-llm" not in sys.argv)

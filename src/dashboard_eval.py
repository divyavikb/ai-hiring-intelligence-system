"""Streamlit evaluation section. In src/dashboard.py add:

    from src.dashboard_eval import render_evaluation_section
    render_evaluation_section()
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

BADGE = {"Green": "🟢 Green", "Amber": "🟠 Amber", "Red": "🔴 Red", "Unrated": "⚪ Unrated"}


def render_evaluation_section(reports_dir="reports"):
    st.header("Insight evaluation")
    files = sorted(Path(reports_dir).glob("eval_*.json"))
    if not files:
        st.info("No evaluation yet. Run: python run_evaluation.py routed")
        return

    path = st.selectbox("Evaluated run", files, format_func=lambda p: p.stem.removeprefix("eval_"))
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]

    cols = st.columns(5)
    cols[0].metric("Mean score", summary["mean_score"] if summary["mean_score"] is not None else "n/a")
    cols[1].metric("Green", summary["green"])
    cols[2].metric("Amber", summary["amber"])
    cols[3].metric("Red", summary["red"])
    cols[4].metric("Judge cost (USD)", f"{summary['judge_cost_usd']:.5f}")

    rows = []
    for agent, ev in data["evaluations"].items():
        judge = ev.get("judge") or {}
        rows.append({
            "Agent": agent,
            "Badge": BADGE.get(ev["badge"], ev["badge"]),
            "Score": ev.get("score"),
            "Actionability": judge.get("actionability"),
            "Grounding": judge.get("grounding"),
            "No hallucination": judge.get("hallucination_free"),
            "Rules failed": ", ".join(ev.get("rules_failed", [])) or "-",
            "Generator": ev.get("generator_model"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for agent, ev in data["evaluations"].items():
        claims = (ev.get("judge") or {}).get("unsupported_claims") or []
        failed = [c for c in ev.get("rule_checks", []) if not c["passed"]]
        if claims or failed or ev.get("judge_error"):
            with st.expander(f"{agent}: why it was flagged"):
                for c in failed:
                    st.write(f"Rule `{c['name']}` failed:", {k: v for k, v in c.items() if k not in ("name", "passed")})
                for claim in claims:
                    st.write("Unsupported claim:", claim)
                if ev.get("judge_error"):
                    st.warning(f"Judge unavailable: {ev['judge_error']}")

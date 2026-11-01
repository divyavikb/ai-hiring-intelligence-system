"""Streamlit cost, routing and optimization panels.

In src/dashboard.py, inside `if "ai_report" in st.session_state:` after the
existing API Usage block, add:

    from src.dashboard_cost import render_agent_cost_table
    render_agent_cost_table(ai_report)

and near the bottom (outside that block), before the evaluation section:

    from src.dashboard_cost import render_optimization_section
    render_optimization_section()
"""

import json
from pathlib import Path

import pandas as pd

RUNS = [
    ("baseline", "Baseline: every agent on the large model"),
    ("routed", "Optimization 1: model routing"),
    ("routed_compressed", "Optimization 2: routing + panel-load input compression"),
]


def agent_cost_rows(report: dict) -> list[dict]:
    rows = []
    for key, item in report["llm_insights"].items():
        routing = item.get("routing") or {}
        rows.append({
            "Agent": key,
            "Model": item.get("model"),
            "Complexity": routing.get("complexity_score"),
            "Routing reason": routing.get("reason"),
            "Input tokens": item.get("input_tokens"),
            "Output tokens": item.get("output_tokens"),
            "Cost (USD)": round(item.get("estimated_api_cost_usd", 0.0), 6),
            "Latency (s)": item.get("latency_seconds"),
        })
    return rows


def optimization_rows(reports_dir="reports") -> list[dict]:
    """Before/after table across the saved baseline and optimized runs."""

    loaded = []
    for label, title in RUNS:
        path = Path(reports_dir) / f"run_{label}.json"
        if path.exists():
            loaded.append((label, title, json.loads(path.read_text(encoding="utf-8"))))
    if not loaded:
        return []

    base_cost = None
    rows = []
    for label, title, rep in loaded:
        usage = rep["usage_summary"]
        cost = usage["total_estimated_api_cost_usd"]
        if label == "baseline":
            base_cost = cost
        saving = None if base_cost in (None, 0) else (base_cost - cost) / base_cost
        rows.append({
            "Run": title,
            "Input tokens": usage["total_input_tokens"],
            "Output tokens": usage["total_output_tokens"],
            "Cost (USD)": round(cost, 7),
            "Saving vs baseline": None if saving is None or label == "baseline" else f"{saving:.1%}",
        })
    return rows


def routing_log_rows(path="logs/routing_log.jsonl", last=15) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines[-last:]]


def render_agent_cost_table(ai_report: dict):
    import streamlit as st

    st.subheader("Cost, model and latency per agent")
    st.dataframe(
        pd.DataFrame(agent_cost_rows(ai_report)),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Cost (USD)": st.column_config.NumberColumn(format="%.6f"),
        },
    )
    usage = ai_report["usage_summary"]
    if "total_latency_seconds" in usage:
        st.caption(f"Total generation latency: {usage['total_latency_seconds']} s. "
                   "Latency differences between runs are mostly noise at this size.")


def render_optimization_section(reports_dir="reports"):
    import streamlit as st

    st.header("Cost optimization: before and after")
    rows = optimization_rows(reports_dir)
    if not rows:
        st.info("No saved runs yet. Run: python run_llm_hiring_intelligence.py --baseline, "
                "then without flags, then with --compress.")
        return

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    chart = pd.DataFrame(rows).set_index("Run")["Cost (USD)"]
    st.bar_chart(chart)
    st.caption("Estimated from token usage and the configured per-model prices. "
               "Evaluation (LLM judge) adds roughly the same cost again per evaluated run.")

    log = routing_log_rows()
    if log:
        with st.expander("Recent routing decisions"):
            st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)

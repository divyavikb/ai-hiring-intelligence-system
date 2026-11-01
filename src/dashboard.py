
from __future__ import annotations

import pandas as pd
import streamlit as st
import traceback

from src.data_ingestion import ingest_all
from src.orchestration.graph import run_hiring_intelligence
from src.dashboard_eval import render_evaluation_section
import json
from pathlib import Path

from src.agents.evaluation_agent import evaluate_report
from src.dashboard_eval import render_evaluation_section
from src.dashboard_cost import render_agent_cost_table, render_optimization_section


st.set_page_config(
    page_title="AI-Powered Hiring Intelligence",
    page_icon="📊",
    layout="wide",
)

st.title("AI-Powered Hiring Intelligence")
st.caption(
    "Hiring operations analytics powered by deterministic agents "
    "and synthetic demonstration data."
)

if st.button("Refresh analysis", type="primary"):
    st.cache_data.clear()

@st.cache_data(ttl=300)
def load_analysis():
    datasets, reports = ingest_all()

    failed = [report.name for report in reports if not report.passed]
    if failed:
        raise RuntimeError(f"Data validation failed: {failed}")

    return run_hiring_intelligence(datasets)


try:
    report = load_analysis()
except Exception as exc:
    st.error(f"Could not load hiring analysis: {exc}")
    st.stop()


results = report["results"]

sourcing = results["sourcing_quality"]["summary"]
offers = results["offer_insights"]
panel = results["panel_load"]
rejections = results["rejection_patterns"]
pipeline = results["pipeline_health"]

total_candidates = pipeline["total_candidates"]
total_offers = offers["total_offer_records"]
acceptance_rate = offers["acceptance_rate"]

interviewers = panel["interviewers"]
overloaded = [
    person for person in interviewers
    if person["status"] == "Over capacity"
]

st.subheader("Overview")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total candidates", total_candidates)
m2.metric("Offer records", total_offers)
m3.metric(
    "Offer acceptance rate",
    f"{acceptance_rate:.1%}"
    if acceptance_rate is not None
    else "N/A",
)
m4.metric("Over-capacity -weeks", len(overloaded))

st.info(
    "These are synthetic demonstration data. Small samples do not "
    "support reliable industry-wide conclusions."
)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Candidates by current stage")

    stage_df = pd.DataFrame(pipeline["stage_distribution"])

    if not stage_df.empty:
        st.bar_chart(
            stage_df.set_index("stage")["candidate_count"]
        )
        st.dataframe(stage_df, use_container_width=True)
    else:
        st.info("No stage data available.")

with right:
    st.subheader("Sourcing performance")

    sourcing_df = pd.DataFrame(sourcing)

    if not sourcing_df.empty:
        chart_df = sourcing_df.set_index("source")[
            "candidate_count"
        ]
        st.bar_chart(chart_df)
        st.dataframe(sourcing_df, use_container_width=True)
    else:
        st.info("No sourcing data available.")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Recorded rejection reasons")

    reason_df = pd.DataFrame(
        rejections["rejection_reason_counts"]
    )

    if not reason_df.empty:
        st.bar_chart(
            reason_df.set_index("rejection_reason")["count"]
        )
        st.dataframe(reason_df, use_container_width=True)
    else:
        st.info("No rejection reasons recorded.")

with right:
    st.subheader("Interviewer workload")

    panel_df = pd.DataFrame(interviewers)

    if not panel_df.empty:
        st.bar_chart(
            panel_df.set_index("interviewer_id")["utilization"]
        )
        st.dataframe(panel_df, use_container_width=True)

st.divider()

st.subheader("Pipeline details")
st.dataframe(
    pd.DataFrame(pipeline["stage_distribution"]),
    use_container_width=True,
)

st.subheader("Offer analysis")


offer_col1, offer_col2, offer_col3 = st.columns(3)

offer_col1.metric(
    "Total offers",
    offers["total_offer_records"],
)

offer_col2.metric(
    "Accepted offers",
    offers.get("accepted_offers", "N/A"),
)

offer_col3.metric(
    "Acceptance rate",
    f"{offers['acceptance_rate']:.1%}"
    if offers.get("acceptance_rate") is not None
    else "N/A",
)


st.subheader("Analysis limitations")

for limitation in pipeline.get("limitations", []):
    st.write(f"- {limitation}")

for warning in pipeline.get("warnings", []):
    st.warning(warning)


# --------------------------------------------------
# AI-Generated Hiring Insights
# --------------------------------------------------

st.divider()
st.header("AI-Generated Hiring Insights")

st.write(
    "Generate recommendations using the hiring agents, "
    "knowledge-base retrieval, and OpenAI."
)

if st.button("Generate AI Insights"):
    with st.spinner("Generating and evaluating AI insights..."):
        try:
            from run_llm_hiring_intelligence import main

            ai_report = main(label="routed_compressed", compress=True)
            st.session_state["ai_report"] = ai_report

            evaluation = evaluate_report(ai_report)
            st.session_state["evaluation"] = evaluation
            Path("reports").mkdir(exist_ok=True)
            Path("reports/eval_routed_compressed.json").write_text(
                json.dumps(evaluation, indent=2, default=str), encoding="utf-8"
            )

        except Exception as exc:
            st.error(f"AI insight generation failed: {type(exc).__name__}: {exc}")
            st.code(traceback.format_exc())




if "ai_report" in st.session_state:
    ai_report = st.session_state["ai_report"]

    usage = ai_report["usage_summary"]

    st.subheader("API Usage")
    c1, c2, c3 = st.columns(3)
    c1.metric("LLM calls", usage["llm_calls"])
    c2.metric(
        "Input tokens",
        usage["total_input_tokens"],
    )
    c3.metric(
        "Output tokens",
        usage["total_output_tokens"],
    )

    st.metric(
        "Estimated API cost (USD)",
        f"${usage['total_estimated_api_cost_usd']:.6f}",
    )
    render_agent_cost_table(ai_report)

    st.subheader("Agent Recommendations")

    for agent_key, item in ai_report["llm_insights"].items():
        insight = item["insight"]
        ev = st.session_state.get("evaluation", {}).get("evaluations", {}).get(agent_key)

        with st.expander(
            agent_key.replace("_", " ").title(),
            expanded=True,
        ):

            if item.get("fallback"):
                st.warning(
                    f"AI insight unavailable. Showing computed figures only. "
                    f"Cause: {item.get('error')}"
                )

            
            if ev:
                badge = {"Green": "🟢", "Amber": "🟠", "Red": "🔴"}.get(ev["badge"], "⚪")
                st.write(f"**Quality check:** {badge} {ev['badge']} (score {ev.get('score')})")
                if ev.get("rules_failed"):
                    st.warning("Rules failed: " + ", ".join(ev["rules_failed"]))
                for claim in (ev.get("judge") or {}).get("unsupported_claims", []):
                    st.caption(f"Unsupported claim: {claim}")

            st.write("**Recommendation**")
            st.write(insight["recommendation"])

            st.write("**Supporting evidence**")
            for evidence in insight["evidence"]:
                st.write(f"- {evidence}")

            st.write(
                f"**Confidence score:** "
                f"{insight['confidence_score']:.0%}"
            )
            st.write(
                f"**Cost of this insight:** "
                f"${insight['cost_of_insight']:.6f}"
            )

            st.write("**Alternative approach**")
            st.write(insight["alternative"])

            st.write("**Retrieved knowledge-base sections**")
            for context in item.get("retrieved_context", []):
                st.write(
                    f"- {context['section']} "
                    f"({context['source']})"
                )
st.divider()
render_evaluation_section()

st.divider()
render_optimization_section()
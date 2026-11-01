import json

from src.data_ingestion import ingest_all
from src.orchestration.graph import run_hiring_intelligence
from src.insight_generator import generate_hiring_insight
from src.rag.retriever import index_knowledge_base, retrieve_context

AGENT_QUERIES = {
    "sourcing_quality": ("source conversion rejection rate sample size", ["sourcing", "responsible_use"]),
    "rejection_patterns": ("rejection reasons job description vs interview", ["rejection", "responsible_use"]),
    "panel_load": ("interviewer capacity utilisation rebalancing", ["panel_load", "responsible_use"]),
    "offer_insights": ("offer decline reasons acceptance rate compensation", ["offers", "responsible_use"]),
    "pipeline_health": ("stage SLA bottleneck conversion", ["pipeline", "responsible_use"]),
}


def main():
    datasets, reports = ingest_all()

    failed = [report for report in reports if not report.passed]
    if failed:
        raise RuntimeError(
            f"Data validation failed: {[report.name for report in failed]}"
        )

    print("Indexing hiring knowledge base...")
    indexed_count = index_knowledge_base()
    print(f"Indexed {indexed_count} knowledge-base sections.")

    analysis_report = run_hiring_intelligence(datasets)

    insights = {}

    for agent_key, analysis in analysis_report["results"].items():
        agent_name = analysis.get("agent", agent_key)

        query, agents = AGENT_QUERIES[agent_key]
        print(f"Retrieving context: {agent_name}")
        retrieved_context = retrieve_context(query, n_results=3, agents=agents)
        print(f"Retrieved {len(retrieved_context)} sections for {agent_name}.")

        print(f"Generating insight: {agent_name}")
        result = generate_hiring_insight(
            agent_name=agent_name,
            analysis=analysis,
            retrieved_context=retrieved_context,
        )
        insight_record = result.model_dump()
        insight_record["retrieved_context"] = retrieved_context
        insights[agent_key] = insight_record

    total_cost = sum(i["estimated_api_cost_usd"] for i in insights.values())
    total_input_tokens = sum(i["input_tokens"] for i in insights.values())
    total_output_tokens = sum(i["output_tokens"] for i in insights.values())

    report = {
        "system": "AI-Powered Hiring Intelligence System",
        "analysis_report": analysis_report,
        "llm_insights": insights,
        "usage_summary": {
            "llm_calls": len(insights),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_estimated_api_cost_usd": round(total_cost, 8),
        },
    }

    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    main()
import json
import sys
from functools import partial
from pathlib import Path

from src.agents.evaluation_agent import evaluate_report
from src.agents.panel_load_balancer import summarize_panel_load
from src.agents.routing_agent import LARGE_MODEL, route_model
from src.data_ingestion import ingest_all
from src.insight_generator import generate_hiring_insight
from src.orchestration.insight_graph import Deps, run_insight_graph
from src.rag.retriever import index_knowledge_base, retrieve_context

AGENT_QUERIES = {
    "sourcing_quality": ("source conversion rejection rate sample size", ["sourcing", "responsible_use"]),
    "rejection_patterns": ("rejection reasons job description vs interview", ["rejection", "responsible_use"]),
    "panel_load": ("interviewer capacity utilisation rebalancing", ["panel_load", "responsible_use"]),
    "offer_insights": ("offer decline reasons acceptance rate compensation", ["offers", "responsible_use"]),
    "pipeline_health": ("stage SLA bottleneck conversion", ["pipeline", "responsible_use"]),
}


def _retrieve(agent_key):
    query, agents = AGENT_QUERIES[agent_key]
    return retrieve_context(query, n_results=3, agents=agents)


def main(force_model=None, label="routed", compress=False, evaluate=False):
    def prepare_input(agent_key, analysis):
        if compress and agent_key == "panel_load":
            return summarize_panel_load(analysis)
        return analysis

    deps = Deps(
        ingest=ingest_all,
        index_knowledge=index_knowledge_base,
        retrieve=_retrieve,
        route=partial(_route, force_model=force_model),
        generate=generate_hiring_insight,
        evaluate=evaluate_report if evaluate else None,
        prepare_input=prepare_input,
    )

    report = run_insight_graph(deps, label=label)

    if report.get("errors"):
        print("Completed with issues:")
        for error in report["errors"]:
            print(f"  - {error}")

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"run_{label}.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    if evaluate and "evaluation" in report:
        (out_dir / f"eval_{label}.json").write_text(
            json.dumps(report["evaluation"], indent=2, default=str), encoding="utf-8"
        )

    print(json.dumps(report["usage_summary"], indent=2))
    return report


def _route(agent_key, analysis, force_model=None):
    return route_model(agent_key, analysis, force_model=force_model)


if __name__ == "__main__":
    if "--baseline" in sys.argv:
        main(force_model=LARGE_MODEL, label="baseline")
    elif "--compress" in sys.argv:
        main(label="routed_compressed", compress=True)
    else:
        main(evaluate="--evaluate" in sys.argv)

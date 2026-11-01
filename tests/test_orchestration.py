
import pytest

from src.data_ingestion import ingest_all
from src.orchestration.graph import run_hiring_intelligence


def test_orchestrator_runs_all_five_agents():
    datasets, reports = ingest_all()

    assert all(report.passed for report in reports)

    result = run_hiring_intelligence(datasets)

    expected_agents = {
        "sourcing_quality",
        "rejection_patterns",
        "panel_load",
        "offer_insights",
        "pipeline_health",
    }

    assert set(result["agents_executed"]) == expected_agents
    assert set(result["results"].keys()) == expected_agents


def test_orchestrator_rejects_missing_datasets():
    with pytest.raises(ValueError, match="Missing required datasets"):
        run_hiring_intelligence({})

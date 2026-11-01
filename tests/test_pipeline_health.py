
import pandas as pd

from src.agents.pipeline_health import analyze_pipeline_health


def test_pipeline_health_summarizes_stages_and_roles():
    candidates = pd.DataFrame(
        {
            "candidate_id": ["C1", "C2", "C3", "C4", "C5"],
            "role": [
                "Data Scientist",
                "Data Scientist",
                "Analyst",
                "Analyst",
                "Analyst",
            ],
            "stage": [
                "Screen",
                "Screen",
                "Interview",
                "Rejected",
                "Rejected",
            ],
            "application_date": [
                "2026-10-01",
                "2026-10-02",
                "2026-10-03",
                "2026-10-04",
                "2026-10-05",
            ],
        }
    )

    result = analyze_pipeline_health(candidates)

    assert result["total_candidates"] == 5

    stages = {
        row["stage"]: row
        for row in result["stage_distribution"]
    }

    assert stages["Screen"]["candidate_count"] == 2
    assert stages["Rejected"]["candidate_share"] == 0.4

    roles = {
        row["role"]: row
        for row in result["role_distribution"]
    }

    assert roles["Analyst"]["candidate_count"] == 3


def test_invalid_application_dates_are_reported():
    candidates = pd.DataFrame(
        {
            "candidate_id": ["C1"],
            "role": ["Analyst"],
            "stage": ["Screen"],
            "application_date": ["invalid-date"],
        }
    )

    result = analyze_pipeline_health(candidates)

    assert any(
        "missing or invalid application dates" in warning
        for warning in result["warnings"]
    )


def test_sla_breaches_and_roles_open_too_long():
    candidates = pd.DataFrame(
        {
            "candidate_id": ["C1", "C2", "C3", "C4"],
            "role": ["A", "A", "B", "B"],
            "stage": ["Screen", "Screen", "Interview", "Hired"],
            "application_date": [
                "2026-08-01", "2026-10-01", "2026-10-01", "2026-09-01",
            ],
            "stage_entered_date": [
                "2026-10-01", "2026-10-20", "2026-10-22", "2026-09-20",
            ],
        }
    )

    result = analyze_pipeline_health(candidates, as_of="2026-10-24")
    screen = next(
        row for row in result["stage_wait_summary"] if row["stage"] == "Screen"
    )

    assert screen["candidates"] == 2
    assert screen["breach_count"] == 1  # 23 days > 10.5; 4 days is fine
    assert result["sla_breach_count"] == 1
    assert result["roles_open_too_long"] == ["A"]  # B has a Hired candidate


def test_sla_analysis_skipped_without_stage_entered_date():
    candidates = pd.DataFrame(
        {
            "candidate_id": ["C1"],
            "role": ["A"],
            "stage": ["Screen"],
            "application_date": ["2026-10-01"],
        }
    )

    result = analyze_pipeline_health(candidates)

    assert "stage_wait_summary" not in result


import pandas as pd
import pytest

from src.agents.panel_load_balancer import analyze_panel_load


def test_panel_load_calculates_utilization_and_overload():
    data = pd.DataFrame(
        {
            "interviewer_id": ["I1", "I2"],
            "interviews_assigned": [8, 3],
            "capacity_per_week": [5, 5],
            "week_start": ["2026-10-12", "2026-10-12"],
        }
    )

    result = analyze_panel_load(data)
    interviewers = {
        item["interviewer_id"]: item
        for item in result["interviewers"]
    }

    assert interviewers["I1"]["over_capacity"] == 3
    assert interviewers["I1"]["utilization"] == 1.6
    assert interviewers["I1"]["status"] == "Over capacity"

    assert interviewers["I2"]["available_capacity"] == 2
    assert interviewers["I2"]["status"] == "Within capacity"

    assert result["over_capacity_count"] == 1


def test_zero_capacity_is_rejected():
    data = pd.DataFrame(
        {
            "interviewer_id": ["I1"],
            "interviews_assigned": [3],
            "capacity_per_week": [0],
            "week_start": ["2026-10-12"],
        }
    )

    with pytest.raises(ValueError, match="greater than zero"):
        analyze_panel_load(data)


def test_summarize_panel_load_aggregates_per_interviewer():
    import pandas as pd
    from src.agents.panel_load_balancer import summarize_panel_load

    load = pd.DataFrame(
        {
            "interviewer_id": ["I1", "I1", "I2", "I2"],
            "interviews_assigned": [8, 7, 2, 3],
            "capacity_per_week": [6, 6, 6, 6],
            "week_start": ["2026-10-05", "2026-10-12"] * 2,
        }
    )

    summary = summarize_panel_load(analyze_panel_load(load))

    assert summary["interviewers_over_capacity"] == ["I1"]
    assert summary["over_capacity_interviewer_weeks"] == 2
    assert summary["total_excess_interviews"] == 3
    assert summary["interviewers"][0]["interviewer_id"] == "I1"
    assert summary["interviewers"][0]["weeks_over_capacity"] == 2


import pandas as pd

from src.agents.rejection_patterns import analyze_rejection_patterns


def test_rejection_reasons_and_feedback_are_counted():
    candidates = pd.DataFrame(
        {
            "candidate_id": ["C1", "C2", "C3"],
            "role": ["Data Scientist", "Data Scientist", "Analyst"],
            "stage": ["Rejected", "Rejected", "Interview"],
            "rejection_reason": [
                "Skills mismatch",
                "Skills mismatch",
                None,
            ],
        }
    )

    feedback = pd.DataFrame(
        {
            "candidate_id": ["C1", "C2", "C3"],
            "interview_stage": ["Technical", "Technical", "HR"],
            "feedback": [
                "Weak SQL skills",
                "Weak SQL skills",
                "Good communication",
            ],
            "decision": ["Reject", "Rejected", "Hire"],
        }
    )

    result = analyze_rejection_patterns(
        candidates,
        feedback,
        min_sample_size=2,
    )

    assert result["rejected_candidate_count"] == 2
    assert result["rejection_reason_counts"][0] == {
        "rejection_reason": "Skills mismatch",
        "count": 2,
    }
    assert result["negative_feedback_count"] == 2
    assert result["repeated_negative_feedback"][0]["count"] == 2


def test_missing_rejection_reasons_are_reported():
    candidates = pd.DataFrame(
        {
            "candidate_id": ["C1"],
            "role": ["Analyst"],
            "stage": ["Rejected"],
            "rejection_reason": [None],
        }
    )

    feedback = pd.DataFrame(
        {
            "candidate_id": ["C1"],
            "interview_stage": ["Technical"],
            "feedback": ["Needs more practice"],
            "decision": ["Reject"],
        }
    )

    result = analyze_rejection_patterns(
        candidates,
        feedback,
        min_sample_size=3,
    )

    assert result["rejection_reason_counts"] == []
    assert any(
        "No recorded rejection reasons" in warning
        for warning in result["warnings"]
    )


def test_role_breakdown_flags_screen_experience_mismatch():
    rows = 6
    candidates = pd.DataFrame(
        {
            "candidate_id": [f"C{i}" for i in range(rows)],
            "role": ["Analyst"] * rows,
            "stage": ["Rejected"] * rows,
            "rejection_reason": ["Experience Mismatch"] * 4 + ["Technical Skills"] * 2,
            "interviewer_id": ["I1"] * rows,
        }
    )
    feedback = pd.DataFrame(
        {
            "candidate_id": [f"C{i}" for i in range(rows)],
            "interview_stage": ["Screen"] * 4 + ["Technical"] * 2,
            "feedback": ["Experience did not match role"] * 4 + ["Weak SQL"] * 2,
            "decision": ["Reject"] * rows,
        }
    )

    result = analyze_rejection_patterns(candidates, feedback)
    role = result["role_rejection_breakdown"][0]

    assert role["screen_experience_mismatch_count"] == 4
    assert role["jd_problem_signal"] is True
    assert role["top_technical_feedback"]["feedback_text"] == "Weak SQL"
    assert role["rejections_by_stage"] == {"Screen": 4, "Technical": 2}


def test_small_roles_never_trigger_the_jd_signal():
    candidates = pd.DataFrame(
        {
            "candidate_id": ["C1", "C2", "C3"],
            "role": ["Tiny"] * 3,
            "stage": ["Rejected"] * 3,
            "rejection_reason": ["Experience Mismatch"] * 3,
        }
    )
    feedback = pd.DataFrame(
        {
            "candidate_id": ["C1", "C2", "C3"],
            "interview_stage": ["Screen"] * 3,
            "feedback": ["x"] * 3,
            "decision": ["Reject"] * 3,
        }
    )

    result = analyze_rejection_patterns(candidates, feedback)

    assert result["role_rejection_breakdown"][0]["jd_problem_signal"] is False


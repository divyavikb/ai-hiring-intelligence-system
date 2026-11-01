
from __future__ import annotations

import pandas as pd


def analyze_rejection_patterns(
    candidates: pd.DataFrame,
    interview_feedback: pd.DataFrame,
    min_sample_size: int = 3,
) -> dict:
    """
    Summarize rejection reasons and negative interview feedback.

    The agent identifies recurring observations. It does not infer
    causation or treat every negative comment as a confirmed rejection.
    """

    required_candidate_columns = {
        "candidate_id",
        "role",
        "stage",
        "rejection_reason",
    }
    required_feedback_columns = {
        "candidate_id",
        "interview_stage",
        "feedback",
        "decision",
    }

    missing_candidates = required_candidate_columns - set(
        candidates.columns
    )
    missing_feedback = required_feedback_columns - set(
        interview_feedback.columns
    )

    if missing_candidates:
        raise ValueError(
            f"Candidates data missing columns: {sorted(missing_candidates)}"
        )

    if missing_feedback:
        raise ValueError(
            f"Interview feedback missing columns: {sorted(missing_feedback)}"
        )

    df = candidates.copy()
    feedback_df = interview_feedback.copy()

    # Identify candidates currently marked as rejected.
    rejected_stage = (
        df["stage"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("rejected")
    )

    df["is_rejected"] = rejected_stage

    rejected = df[df["is_rejected"]].copy()

    # Count the recorded rejection reasons.
    reasons = (
        rejected["rejection_reason"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    reasons = reasons[reasons.ne("")]

    reason_counts = (
        reasons.value_counts()
        .rename_axis("rejection_reason")
        .reset_index(name="count")
    )

    # Summarize rejection counts by role.
    rejected_by_role = (
        rejected.groupby("role", dropna=False)
        .size()
        .reset_index(name="rejected_candidates")
    )

    # Normalize decisions for descriptive analysis.
    feedback_df["decision_normalized"] = (
        feedback_df["decision"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    negative_decisions = {
        "reject",
        "rejected",
        "no hire",
        "no-hire",
        "not recommended",
    }

    negative_feedback = feedback_df[
        feedback_df["decision_normalized"].isin(negative_decisions)
    ].copy()

    # Report negative feedback by interview stage.
    negative_feedback_by_stage = (
        negative_feedback.groupby("interview_stage", dropna=False)
        .size()
        .reset_index(name="negative_feedback_count")
    )

    # Show repeated feedback text without pretending to understand
    # semantic similarity. LLM-based theme extraction can come later.
    feedback_text = (
        negative_feedback["feedback"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    feedback_text = feedback_text[feedback_text.ne("")]

    repeated_feedback = (
        feedback_text.value_counts()
        .rename_axis("feedback_text")
        .reset_index(name="count")
    )

    # ---- Breakdowns the guidelines need: role, stage, interviewer ----
    rejected = rejected.copy()
    rejected["reason_clean"] = (
        rejected["rejection_reason"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    # Stage at which each rejected candidate was turned down:
    # the last negative feedback record for that candidate.
    stage_by_candidate = {}
    if not negative_feedback.empty:
        ordered = negative_feedback
        if "feedback_date" in ordered.columns:
            ordered = ordered.sort_values("feedback_date")
        stage_by_candidate = (
            ordered.groupby("candidate_id")["interview_stage"].last().to_dict()
        )
    rejected["rejection_stage"] = (
        rejected["candidate_id"].map(stage_by_candidate).fillna("Unknown")
    )

    technical_feedback = negative_feedback[
        negative_feedback["interview_stage"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("technical")
    ]
    technical_with_role = technical_feedback.merge(
        df[["candidate_id", "role"]], on="candidate_id", how="left"
    )

    role_breakdown = []
    for role, group in rejected.groupby("role"):
        total = int(len(group))
        reason_counts_role = group["reason_clean"].value_counts()
        screen_mismatch = int(
            (
                group["reason_clean"].str.lower().eq("experience mismatch")
                & group["rejection_stage"].str.lower().eq("screen")
            ).sum()
        )
        tech_rows = technical_with_role[technical_with_role["role"] == role]
        top_tech = (
            tech_rows["feedback"].value_counts().head(1)
            if not tech_rows.empty
            else None
        )
        role_breakdown.append(
            {
                "role": str(role),
                "rejected_candidates": total,
                "reasons": [
                    {"reason": str(reason), "count": int(count)}
                    for reason, count in reason_counts_role.items()
                ],
                "rejections_by_stage": {
                    str(stage): int(count)
                    for stage, count in group["rejection_stage"]
                    .value_counts()
                    .items()
                },
                "screen_experience_mismatch_count": screen_mismatch,
                "screen_experience_mismatch_share": (
                    round(screen_mismatch / total, 3) if total else 0.0
                ),
                "jd_problem_signal": bool(
                    total >= max(min_sample_size, 5)
                    and screen_mismatch / total > 0.30
                ),
                "technical_stage_rejections": int(len(tech_rows)),
                "top_technical_feedback": (
                    {
                        "feedback_text": str(top_tech.index[0]),
                        "count": int(top_tech.iloc[0]),
                    }
                    if top_tech is not None and len(top_tech)
                    else None
                ),
            }
        )

    interviewer_rates = []
    if "interviewer_id" in df.columns:
        per_interviewer = (
            df.assign(_rej=df["is_rejected"])
            .groupby("interviewer_id")
            .agg(candidates=("candidate_id", "nunique"), rejected=("_rej", "sum"))
            .reset_index()
        )
        for row in per_interviewer.to_dict(orient="records"):
            if row["candidates"] >= 5:
                interviewer_rates.append(
                    {
                        "interviewer_id": str(row["interviewer_id"]),
                        "candidates": int(row["candidates"]),
                        "rejected": int(row["rejected"]),
                        "rejection_rate": round(
                            row["rejected"] / row["candidates"], 3
                        ),
                    }
                )

    warnings = []

    if len(rejected) < min_sample_size:
        warnings.append(
            f"Only {len(rejected)} rejected candidates; "
            "patterns may be unreliable."
        )

    if reason_counts.empty:
        warnings.append("No recorded rejection reasons are available.")

    if negative_feedback.empty:
        warnings.append(
            "No feedback records have a recognized negative decision."
        )

    return {
        "agent": "Rejection Pattern Agent",
        "rejected_candidate_count": int(len(rejected)),
        "rejection_reason_counts": reason_counts.to_dict(
            orient="records"
        ),
        "rejections_by_role": rejected_by_role.to_dict(
            orient="records"
        ),
        "role_rejection_breakdown": role_breakdown,
        "interviewer_rejection_rates": interviewer_rates,
        "negative_feedback_count": int(len(negative_feedback)),
        "negative_feedback_by_interview_stage": (
            negative_feedback_by_stage.to_dict(orient="records")
        ),
        "repeated_negative_feedback": repeated_feedback.to_dict(
            orient="records"
        ),
        "warnings": warnings,
        "limitations": [
            "Only candidates currently marked Rejected are counted.",
            "Only recognized negative decision labels are included.",
            "Repeated text is an exact-text match, not semantic theme detection.",
            "Observed patterns do not establish why a candidate was rejected.",
            "Rejection stage is taken from the last negative feedback record.",
            "Interviewer rejection rates only cover interviewers with at least 5 candidates.",
        ],
    }

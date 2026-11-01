from __future__ import annotations

import pandas as pd

# Synthetic SLA targets in days (see data/knowledge_base/pipeline.md).
# Interview = 2 days written feedback + 3 days decision.
SLA_TARGET_DAYS = {"applied": 3, "screen": 7, "interview": 5}
BREACH_MULTIPLIER = 1.5  # breach when wait exceeds target by more than 50%
ROLE_OPEN_DAYS_LIMIT = 45
OFFER_STAGES = {"offer", "hired", "withdrawn"}


def analyze_pipeline_health(
    candidates: pd.DataFrame,
    as_of: str | pd.Timestamp | None = None,
) -> dict:
    """Summarize the current hiring pipeline, including SLA breaches.

    SLA analysis needs a stage_entered_date column. If it is missing,
    only the stage-count snapshot is returned.
    """

    required_columns = {
        "candidate_id",
        "role",
        "stage",
        "application_date",
    }

    missing = required_columns - set(candidates.columns)
    if missing:
        raise ValueError(
            f"Candidates data missing columns: {sorted(missing)}"
        )

    df = candidates.copy()

    df["stage"] = (
        df["stage"].fillna("Unknown").astype(str).str.strip()
    )
    df["role"] = (
        df["role"].fillna("Unknown").astype(str).str.strip()
    )

    total_candidates = int(df["candidate_id"].nunique())

    stage_counts = (
        df.groupby("stage", dropna=False)["candidate_id"]
        .nunique()
        .reset_index(name="candidate_count")
        .sort_values("candidate_count", ascending=False)
    )

    role_counts = (
        df.groupby("role", dropna=False)["candidate_id"]
        .nunique()
        .reset_index(name="candidate_count")
        .sort_values("candidate_count", ascending=False)
    )

    # Calculate the share of candidates in each current stage.
    if total_candidates:
        stage_counts["candidate_share"] = (
            stage_counts["candidate_count"] / total_candidates
        ).round(3)
    else:
        stage_counts["candidate_share"] = 0.0

    warnings = []
    possible_bottlenecks = []

    # Outcome stages are not active processing stages.
    outcome_stages = {
        "rejected",
        "hired",
        "withdrawn",
        "closed",
    }

    if total_candidates == 0:
        warnings.append(
            "No candidates are available for analysis."
        )
    else:
        for row in stage_counts.to_dict(orient="records"):
            stage_name = str(row["stage"]).strip().lower()

            if stage_name in outcome_stages:
                continue

            if row["candidate_share"] >= 0.4:
                possible_bottlenecks.append(
                    {
                        "stage": row["stage"],
                        "candidate_count": int(
                            row["candidate_count"]
                        ),
                        "candidate_share": float(
                            row["candidate_share"]
                        ),
                        "reason": (
                            "At least 40% of candidates are recorded "
                            "in this active stage. Review time-in-stage "
                            "and processing delays before concluding "
                            "that a bottleneck exists."
                        ),
                    }
                )

    # Check application dates for missing or invalid values.
    application_dates = pd.to_datetime(
        df["application_date"], errors="coerce"
    )
    unknown_dates = int(application_dates.isna().sum())

    if unknown_dates:
        warnings.append(
            f"{unknown_dates} candidates have missing or invalid "
            "application dates."
        )

    limitations = [
        "Stage counts describe the current snapshot only.",
        "A large stage count does not prove a processing bottleneck.",
        "Stage-transition conversion requires candidate stage history.",
    ]

    result = {
        "agent": "Pipeline Health Agent",
        "total_candidates": total_candidates,
        "stage_distribution": stage_counts.to_dict(
            orient="records"
        ),
        "role_distribution": role_counts.to_dict(
            orient="records"
        ),
        "possible_bottlenecks": possible_bottlenecks,
        "warnings": warnings,
        "limitations": limitations,
    }

    # ---- SLA and role-age analysis (needs stage_entered_date) ----
    if "stage_entered_date" not in df.columns:
        limitations.append(
            "Time-in-stage analysis requires a stage_entered_date column."
        )
        return result

    entered = pd.to_datetime(df["stage_entered_date"], errors="coerce")

    if as_of is None:
        all_dates = pd.concat([entered, application_dates]).dropna()
        snapshot = all_dates.max() if not all_dates.empty else None
    else:
        snapshot = pd.to_datetime(as_of)

    if snapshot is None or pd.isna(snapshot):
        warnings.append("No valid dates; SLA analysis skipped.")
        return result

    df["days_in_stage"] = (snapshot - entered).dt.days
    stage_key = df["stage"].str.lower()

    invalid_entry = int(
        entered.isna().sum() + (df["days_in_stage"] < 0).sum()
    )
    if invalid_entry:
        warnings.append(
            f"{invalid_entry} candidates have missing or future "
            "stage_entered_date and were excluded from SLA checks."
        )

    stage_wait = []
    for stage_name, target in SLA_TARGET_DAYS.items():
        rows = df[
            (stage_key == stage_name)
            & df["days_in_stage"].notna()
            & (df["days_in_stage"] >= 0)
        ]
        if rows.empty:
            continue

        limit = target * BREACH_MULTIPLIER
        breaches = rows[rows["days_in_stage"] > limit]
        stage_wait.append(
            {
                "stage": rows["stage"].iloc[0],
                "candidates": int(len(rows)),
                "avg_days_in_stage": round(
                    float(rows["days_in_stage"].mean()), 1
                ),
                "max_days_in_stage": int(rows["days_in_stage"].max()),
                "sla_target_days": target,
                "breach_threshold_days": round(limit, 1),
                "breach_count": int(len(breaches)),
                "breach_rate": round(len(breaches) / len(rows), 3),
            }
        )

    # Roles open too long: old, with no offer-stage candidate yet.
    role_age = []
    for role, group in df.groupby("role"):
        first_app = application_dates.loc[group.index].min()
        if pd.isna(first_app):
            continue
        open_days = int((snapshot - first_app).days)
        has_offer = bool(
            group["stage"].str.lower().isin(OFFER_STAGES).any()
        )
        role_age.append(
            {
                "role": role,
                "days_since_first_application": open_days,
                "has_offer_stage_candidate": has_offer,
                "open_too_long": (
                    open_days > ROLE_OPEN_DAYS_LIMIT and not has_offer
                ),
            }
        )

    result["as_of_date"] = snapshot.date().isoformat()
    result["stage_wait_summary"] = stage_wait
    result["sla_breach_count"] = sum(
        item["breach_count"] for item in stage_wait
    )
    result["role_age"] = role_age
    result["roles_open_too_long"] = [
        item["role"] for item in role_age if item["open_too_long"]
    ]
    limitations.append(
        "Wait times use the snapshot date (latest date in the data) "
        "and calendar days, not business days."
    )
    return result

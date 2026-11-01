
from __future__ import annotations

import pandas as pd


def analyze_offer_insights(
    offers: pd.DataFrame,
    min_sample_size: int = 3,
) -> dict:
    """Analyze recorded offer outcomes and decline reasons."""

    required_columns = {
        "offer_id",
        "candidate_id",
        "response",
        "decline_reason",
        "offered_salary",
    }

    missing = required_columns - set(offers.columns)
    if missing:
        raise ValueError(
            f"Offers data missing columns: {sorted(missing)}"
        )

    df = offers.copy()

    df["response_normalized"] = (
        df["response"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    accepted_labels = {"accepted", "accept", "offer accepted"}
    declined_labels = {"declined", "decline", "offer declined"}

    df["is_accepted"] = df["response_normalized"].isin(
        accepted_labels
    )
    df["is_declined"] = df["response_normalized"].isin(
        declined_labels
    )
    df["is_resolved"] = df["is_accepted"] | df["is_declined"]

    resolved = df[df["is_resolved"]]
    accepted_count = int(df["is_accepted"].sum())
    declined_count = int(df["is_declined"].sum())
    resolved_count = int(len(resolved))

    acceptance_rate = (
        round(accepted_count / resolved_count, 3)
        if resolved_count
        else None
    )

    decline_reasons = (
        df.loc[df["is_declined"], "decline_reason"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    decline_reasons = decline_reasons[decline_reasons.ne("")]

    decline_reason_counts = (
        decline_reasons.value_counts()
        .rename_axis("decline_reason")
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    salaries = pd.to_numeric(df["offered_salary"], errors="coerce")
    valid_salaries = salaries.dropna()

    offer_speed = None
    if "final_interview_date" in df.columns and "offer_date" in df.columns:
        gap = (
            pd.to_datetime(df["offer_date"], errors="coerce")
            - pd.to_datetime(df["final_interview_date"], errors="coerce")
        ).dt.days
        valid_gap = gap.dropna()
        valid_gap = valid_gap[valid_gap >= 0]
        if len(valid_gap):
            # Calendar-day approximations of the 5 and 10 business-day guidelines.
            delayed = df.loc[valid_gap[valid_gap > 14].index]
            on_time = df.loc[valid_gap[valid_gap <= 7].index]
            offer_speed = {
                "offers_with_dates": int(len(valid_gap)),
                "avg_days_final_interview_to_offer": round(
                    float(valid_gap.mean()), 1
                ),
                "offers_within_7_days": int(len(on_time)),
                "offers_delayed_over_14_days": int(len(delayed)),
                "declines_among_delayed": int(delayed["is_declined"].sum()),
                "declines_among_within_7_days": int(
                    on_time["is_declined"].sum()
                ),
            }

    warnings = []

    if resolved_count < min_sample_size:
        warnings.append(
            f"Only {resolved_count} resolved offers; "
            "acceptance rate may be unreliable."
        )

    unresolved_count = int((~df["is_resolved"]).sum())
    if unresolved_count:
        warnings.append(
            f"{unresolved_count} offer records have unresolved or "
            "unrecognized responses."
        )

    if df["is_declined"].sum() > len(decline_reasons):
        warnings.append(
            "Some declined offers have no recorded decline reason."
        )

    return {
        "agent": "Offer Insights Agent",
        "total_offer_records": int(len(df)),
        "resolved_offers": resolved_count,
        "accepted_offers": accepted_count,
        "declined_offers": declined_count,
        "acceptance_rate": acceptance_rate,
        "decline_reason_counts": decline_reason_counts,
        "salary_summary": {
            "offers_with_salary": int(len(valid_salaries)),
            "mean_offered_salary": (
                round(float(valid_salaries.mean()), 2)
                if len(valid_salaries)
                else None
            ),
            "median_offered_salary": (
                round(float(valid_salaries.median()), 2)
                if len(valid_salaries)
                else None
            ),
        },
        "offer_speed": offer_speed,
        "warnings": warnings,
        "limitations": [
            "A small number of offers cannot establish reliable trends.",
            "Salary comparisons require comparable roles, levels, and currencies.",
            "Recorded decline reasons may be missing or incomplete.",
            "Acceptance rates describe observed outcomes, not their causes.",
        ],
    }

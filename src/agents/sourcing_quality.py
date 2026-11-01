
from __future__ import annotations

import pandas as pd


def analyze_sourcing_quality(
    candidates: pd.DataFrame,
    offers: pd.DataFrame,
    min_sample_size: int = 5,
) -> dict:
    """
    Analyze candidate sources using the available hiring data.

    This is a descriptive analysis, not a causal assessment
    of source quality.
    """

    required_candidate_columns = {
        "candidate_id",
        "source",
        "stage",
        "rejection_reason",
    }
    required_offer_columns = {"candidate_id", "response"}

    missing_candidates = required_candidate_columns - set(
        candidates.columns
    )
    missing_offers = required_offer_columns - set(offers.columns)

    if missing_candidates:
        raise ValueError(
            f"Candidates data missing columns: {sorted(missing_candidates)}"
        )

    if missing_offers:
        raise ValueError(
            f"Offers data missing columns: {sorted(missing_offers)}"
        )

    df = candidates.copy()
    offer_df = offers.copy()

    # Handle missing source names explicitly.
    df["source"] = df["source"].fillna("Unknown")
    df["stage"] = df["stage"].fillna("Unknown")

    # Identify rejected candidates from stage or rejection reason.
    stage_text = df["stage"].astype(str).str.strip().str.lower()

    rejected_by_stage = stage_text.isin(
        {"rejected", "declined", "not selected"}
    )

    rejected_by_reason = (
        df["rejection_reason"].notna()
        & df["rejection_reason"].astype(str).str.strip().ne("")
    )

    df["is_rejected"] = rejected_by_stage | rejected_by_reason

    # Summarize candidate counts and current-stage distribution.
    source_summary = (
        df.groupby("source", dropna=False)
        .agg(
            candidate_count=("candidate_id", "nunique"),
            rejected_candidates=("is_rejected", "sum"),
        )
        .reset_index()
    )

    source_summary["rejection_rate"] = (
        source_summary["rejected_candidates"]
        / source_summary["candidate_count"]
    ).round(3)

    stage_distribution = (
        df.groupby(["source", "stage"], dropna=False)
        .size()
        .reset_index(name="candidate_count")
    )

    # Join offers to the candidate source.
    offer_candidates = offer_df.merge(
        df[["candidate_id", "source"]],
        on="candidate_id",
        how="left",
        validate="many_to_one",
    )

    offer_candidates["source"] = (
        offer_candidates["source"].fillna("Unknown")
    )

    # Calculate offer acceptance only for recognized responses.
    response = (
        offer_candidates["response"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    offer_candidates["is_accepted"] = response.isin(
        {"accepted", "accept", "offer accepted"}
    )

    offer_candidates["is_declined"] = response.isin(
        {"declined", "decline", "rejected", "offer declined"}
    )

    offer_candidates["is_resolved"] = (
        offer_candidates["is_accepted"]
        | offer_candidates["is_declined"]
    )

    offer_summary = (
        offer_candidates.groupby("source", dropna=False)
        .agg(
            offers=("candidate_id", "size"),
            resolved_offers=("is_resolved", "sum"),
            accepted_offers=("is_accepted", "sum"),
        )
        .reset_index()
    )

    offer_summary["acceptance_rate"] = (
        offer_summary["accepted_offers"]
        / offer_summary["resolved_offers"].replace(0, pd.NA)
    ).round(3)

    # Combine the source-level metrics.
    summary = source_summary.merge(
        offer_summary,
        on="source",
        how="left",
    )

    for column in ["offers", "resolved_offers", "accepted_offers"]:
        summary[column] = summary[column].fillna(0).astype(int)

    results = []

    for row in summary.to_dict(orient="records"):
        source = row["source"]
        candidate_count = int(row["candidate_count"])

        warnings = []

        if candidate_count < min_sample_size:
            warnings.append(
                f"Small sample: only {candidate_count} candidates."
            )

        if row["resolved_offers"] == 0:
            warnings.append(
                "No resolved offer outcomes available for this source."
            )

        results.append(
            {
                "source": str(source),
                "candidate_count": candidate_count,
                "rejected_candidates": int(row["rejected_candidates"]),
                "rejection_rate": float(row["rejection_rate"]),
                "offers": int(row["offers"]),
                "resolved_offers": int(row["resolved_offers"]),
                "accepted_offers": int(row["accepted_offers"]),
                "offer_acceptance_rate": (
                    None
                    if pd.isna(row["acceptance_rate"])
                    else float(row["acceptance_rate"])
                ),
                "warnings": warnings,
            }
        )

    return {
        "agent": "Sourcing Quality Agent",
        "summary": results,
        "current_stage_distribution": stage_distribution.to_dict(
            orient="records"
        ),
        "limitations": [
            "Current-stage distribution is not stage-to-stage conversion.",
            "Rejection rate uses rejection stage or recorded rejection reason.",
            "Offer acceptance rates exclude unrecognized or unresolved responses.",
            "Small samples may not support reliable comparisons.",
            "These associations do not establish that a source causes better hiring outcomes.",
        ],
    }

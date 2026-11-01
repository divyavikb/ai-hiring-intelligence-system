
from __future__ import annotations

import pandas as pd


def analyze_panel_load(interviewer_load: pd.DataFrame) -> dict:
    """Compare assigned interview counts with weekly capacity."""

    required_columns = {
        "interviewer_id",
        "interviews_assigned",
        "capacity_per_week",
        "week_start",
    }

    missing = required_columns - set(interviewer_load.columns)
    if missing:
        raise ValueError(
            f"Interviewer load data missing columns: {sorted(missing)}"
        )

    df = interviewer_load.copy()

    for column in ["interviews_assigned", "capacity_per_week"]:
        df[column] = pd.to_numeric(df[column], errors="raise")

        if df[column].isna().any():
            raise ValueError(f"{column} contains missing values.")

        if (df[column] < 0).any():
            raise ValueError(f"{column} cannot contain negative values.")

    if (df["capacity_per_week"] == 0).any():
        raise ValueError("capacity_per_week must be greater than zero.")

    if df.duplicated(["interviewer_id", "week_start"]).any():
        raise ValueError(
            "Duplicate interviewer and week records found."
        )

    df["over_capacity"] = (
        df["interviews_assigned"] - df["capacity_per_week"]
    ).clip(lower=0)

    df["available_capacity"] = (
        df["capacity_per_week"] - df["interviews_assigned"]
    ).clip(lower=0)

    df["utilization"] = (
        df["interviews_assigned"] / df["capacity_per_week"]
    ).round(3)

    results = []

    for row in df.to_dict(orient="records"):
        assigned = int(row["interviews_assigned"])
        capacity = int(row["capacity_per_week"])
        over_capacity = int(row["over_capacity"])
        available = int(row["available_capacity"])

        if over_capacity > 0:
            status = "Over capacity"
            recommendation = (
                f"Review reassignment of up to {over_capacity} interviews."
            )
        elif available == 0:
            status = "At capacity"
            recommendation = (
                "Avoid assigning additional interviews without a capacity review."
            )
        else:
            status = "Within capacity"
            recommendation = (
                f"{available} interview slots remain available."
            )

        results.append(
            {
                "interviewer_id": str(row["interviewer_id"]),
                "week_start": str(row["week_start"]),
                "interviews_assigned": assigned,
                "capacity_per_week": capacity,
                "over_capacity": over_capacity,
                "available_capacity": available,
                "utilization": float(row["utilization"]),
                "status": status,
                "recommendation": recommendation,
            }
        )

    return {
        "agent": "Panel Load Balancer",
        "interviewers": results,
        "over_capacity_count": sum(
            item["over_capacity"] > 0 for item in results
        ),
        "limitations": [
            "Capacity counts do not capture interview complexity or duration.",
            "Reassignment requires checking role expertise and availability.",
            "The results describe the supplied week only.",
        ],
    }


def summarize_panel_load(analysis: dict) -> dict:
    """Compact per-interviewer view of analyze_panel_load() output.

    Sent to the LLM instead of one row per interviewer-week. Fewer tokens,
    and the totals the model should quote (total excess, who is over
    capacity) are computed here instead of being re-derived by the model.
    """

    per_interviewer: dict[str, dict] = {}

    for row in analysis["interviewers"]:
        item = per_interviewer.setdefault(
            row["interviewer_id"],
            {
                "interviewer_id": row["interviewer_id"],
                "weeks": 0,
                "total_assigned": 0,
                "total_capacity": 0,
                "weeks_over_capacity": 0,
                "weeks_at_capacity": 0,
                "total_excess_interviews": 0,
                "max_utilization": 0.0,
            },
        )
        item["weeks"] += 1
        item["total_assigned"] += row["interviews_assigned"]
        item["total_capacity"] += row["capacity_per_week"]
        item["total_excess_interviews"] += row["over_capacity"]
        item["weeks_over_capacity"] += int(row["over_capacity"] > 0)
        item["weeks_at_capacity"] += int(row["status"] == "At capacity")
        item["max_utilization"] = max(
            item["max_utilization"], row["utilization"]
        )

    interviewers = []
    for item in per_interviewer.values():
        item["avg_utilization"] = round(
            item["total_assigned"] / item["total_capacity"], 3
        )
        interviewers.append(item)

    interviewers.sort(key=lambda x: x["avg_utilization"], reverse=True)

    return {
        "agent": analysis["agent"],
        "weeks_covered": max((i["weeks"] for i in interviewers), default=0),
        "interviewers": interviewers,
        "interviewers_over_capacity": [
            i["interviewer_id"]
            for i in interviewers
            if i["weeks_over_capacity"] > 0
        ],
        "over_capacity_interviewer_weeks": analysis["over_capacity_count"],
        "total_excess_interviews": sum(
            i["total_excess_interviews"] for i in interviewers
        ),
        "limitations": analysis["limitations"],
    }

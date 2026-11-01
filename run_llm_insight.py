
import json

from src.data_ingestion import ingest_all
from src.agents.sourcing_quality import analyze_sourcing_quality
from src.insight_generator import generate_hiring_insight


def main():
    datasets, reports = ingest_all()

    failed = [report for report in reports if not report.passed]
    if failed:
        raise RuntimeError(
            f"Data validation failed: {[report.name for report in failed]}"
        )

    analysis = analyze_sourcing_quality(
        candidates=datasets["candidates"],
        offers=datasets["offers"],
    )

    result = generate_hiring_insight(
        agent_name="Sourcing Quality Agent",
        analysis=analysis,
    )

    print(json.dumps(result.model_dump(), indent=2, default=str))


if __name__ == "__main__":
    main()

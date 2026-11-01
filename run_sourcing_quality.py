
import json

from src.data_ingestion import ingest_all
from src.agents.sourcing_quality import analyze_sourcing_quality


def main():
    datasets, reports = ingest_all()

    failed = [report for report in reports if not report.passed]
    if failed:
        raise RuntimeError(
            f"Data validation failed: {[report.name for report in failed]}"
        )

    result = analyze_sourcing_quality(
        candidates=datasets["candidates"],
        offers=datasets["offers"],
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

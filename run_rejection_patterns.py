
import json

from src.data_ingestion import ingest_all
from src.agents.rejection_patterns import analyze_rejection_patterns


def main():
    datasets, reports = ingest_all()

    failed = [report for report in reports if not report.passed]
    if failed:
        raise RuntimeError(
            f"Data validation failed: {[report.name for report in failed]}"
        )

    result = analyze_rejection_patterns(
        candidates=datasets["candidates"],
        interview_feedback=datasets["interview_feedback"],
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

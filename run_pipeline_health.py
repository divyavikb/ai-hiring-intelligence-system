
import json

from src.data_ingestion import ingest_all
from src.agents.pipeline_health import analyze_pipeline_health


def main():
    datasets, reports = ingest_all()

    failed = [report for report in reports if not report.passed]
    if failed:
        raise RuntimeError(
            f"Data validation failed: {[report.name for report in failed]}"
        )

    result = analyze_pipeline_health(
        candidates=datasets["candidates"]
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

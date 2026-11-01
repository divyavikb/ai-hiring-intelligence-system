
import json

from src.data_ingestion import ingest_all
from src.agents.panel_load_balancer import analyze_panel_load


def main():
    datasets, reports = ingest_all()

    failed = [report for report in reports if not report.passed]
    if failed:
        raise RuntimeError(
            f"Data validation failed: {[report.name for report in failed]}"
        )

    result = analyze_panel_load(
        interviewer_load=datasets["interviewer_load"]
    )

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

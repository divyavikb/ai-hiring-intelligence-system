import sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from src.config import DATA_DIR

REQUIRED = {
 "candidates.csv": {"candidate_id","role","source","stage","application_date","rejection_reason","interviewer_id"},
 "interview_feedback.csv": {"feedback_id","candidate_id","interview_stage","feedback","decision","interviewer_id","feedback_date"},
 "offers.csv": {"offer_id","candidate_id","offer_date","response","decline_reason","offered_salary"},
 "interviewer_load.csv": {"interviewer_id","interviews_assigned","capacity_per_week","week_start"},
}

OPTIONAL_DATE_COLUMNS = {
    "candidates.csv": ["stage_entered_date"],
    "offers.csv": ["final_interview_date"],
}


@dataclass
class Report:
    name: str
    rows: int
    columns: list
    missing_values: dict
    errors: list
    @property
    def passed(self):
        return not self.errors

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not parse {path.name}: {exc}") from exc

def validate(name: str, df: pd.DataFrame) -> Report:
    errors = []
    missing_cols = sorted(REQUIRED[name] - set(df.columns))
    if missing_cols:
        errors.append("Missing required columns: " + ", ".join(missing_cols))
    missing = {str(c): int(n) for c,n in df.isna().sum().items() if n}
    if name == "candidates.csv" and "candidate_id" in df:
        if df.candidate_id.duplicated().any():
            errors.append("candidate_id must be unique in candidates.csv")
    for col in OPTIONAL_DATE_COLUMNS.get(name, []):
        if col in df and pd.to_datetime(df[col], errors="coerce").isna().sum() > df[col].isna().sum():
            errors.append(f"{col} contains invalid dates")
    if name == "interviewer_load.csv":
        for col in ("interviews_assigned", "capacity_per_week"):
            if col in df:
                values = pd.to_numeric(df[col], errors="coerce")
                if values.isna().any(): errors.append(f"{col} must be numeric")
                if (values < 0).any(): errors.append(f"{col} cannot be negative")
    if name == "offers.csv" and "offered_salary" in df:
        values = pd.to_numeric(df.offered_salary, errors="coerce")
        if values.isna().any(): errors.append("offered_salary must be numeric")
        if (values < 0).any(): errors.append("offered_salary cannot be negative")
    return Report(name, len(df), list(map(str,df.columns)), missing, errors)

def ingest_all(data_dir: Path = DATA_DIR):
    datasets, reports = {}, []
    for name in REQUIRED:
        df = load_csv(data_dir / name)
        datasets[name.removesuffix(".csv")] = df
        reports.append(validate(name, df))
    return datasets, reports

def main():
    try:
        _, reports = ingest_all()
    except (FileNotFoundError, ValueError) as exc:
        print(f"INGESTION FAILED: {exc}", file=sys.stderr)
        return 1
    for r in reports:
        print(f"\n[{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.rows} rows")
        print("Columns:", ", ".join(r.columns))
        print("Missing values:", r.missing_values or "none")
        for e in r.errors: print("ERROR:", e)
    if any(not r.passed for r in reports):
        print("\nValidation failed.", file=sys.stderr); return 1
    print("\nAll datasets loaded and basic validation passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

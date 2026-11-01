import pandas as pd
from src.data_ingestion import validate

def test_candidate_schema_passes():
    df = pd.DataFrame({"candidate_id":["C1"],"role":["Data Scientist"],"source":["Referral"],"stage":["Offer"],"application_date":["2026-09-01"],"rejection_reason":[None],"interviewer_id":["I1"]})
    assert validate("candidates.csv", df).passed

def test_missing_columns_fail():
    report = validate("candidates.csv", pd.DataFrame({"candidate_id":["C1"]}))
    assert not report.passed
    assert "Missing required columns" in report.errors[0]

def test_duplicate_candidate_ids_fail():
    df = pd.DataFrame({"candidate_id":["C1","C1"],"role":["DS","DS"],"source":["Referral","Board"],"stage":["Screen","Interview"],"application_date":["2026-09-01","2026-09-02"],"rejection_reason":[None,None],"interviewer_id":["I1","I2"]})
    assert not validate("candidates.csv", df).passed

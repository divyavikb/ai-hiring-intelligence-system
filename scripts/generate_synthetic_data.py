"""Generate the synthetic hiring dataset (seeded, reproducible).

Run from the project root:  python -m scripts.generate_synthetic_data
All data is synthetic. Snapshot date is AS_OF.
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

AS_OF = date(2026, 10, 24)
SEED = 42
N_CANDIDATES = 72
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"

ROLES = {"Data Scientist": 22, "ML Engineer": 20, "Data Analyst": 20, "MLOps Engineer": 10}
SOURCE_BY_ROLE = {
    "Data Scientist": {"Referral": 25, "Recruiter": 30, "Job Board": 35, "Campus": 10},
    "ML Engineer": {"Referral": 30, "Recruiter": 35, "Job Board": 30, "Campus": 5},
    "Data Analyst": {"Referral": 10, "Recruiter": 15, "Job Board": 45, "Campus": 30},
    "MLOps Engineer": {"Referral": 20, "Recruiter": 40, "Job Board": 40, "Campus": 0},
}
# outcome weights: Rejected, Screen, Interview, Offer, Hired, Withdrawn, Applied
OUTCOME_W = {
    "Referral":  [25, 8, 14, 8, 30, 15, 2],
    "Recruiter": [45, 12, 15, 6, 12, 10, 3],
    "Job Board": [70, 14, 8, 2, 3, 3, 4],
    "Campus":    [50, 20, 12, 5, 8, 5, 4],
}
OUTCOMES = ["Rejected", "Screen", "Interview", "Offer", "Hired", "Withdrawn", "Applied"]
BREACH_P = {"Applied": 0.15, "Screen": 0.30, "Interview": 0.45}
SLA = {"Applied": 3, "Screen": 7, "Interview": 5}
SALARY = {"Data Scientist": (1_600_000, 2_400_000), "ML Engineer": (1_800_000, 2_600_000),
          "Data Analyst": (900_000, 1_500_000), "MLOps Engineer": (1_700_000, 2_500_000)}
INTERVIEWER_W = {"I001": 22, "I002": 14, "I003": 18, "I004": 10, "I005": 5, "I006": 9, "I007": 14, "I008": 8}

FEEDBACK = {
    "Technical Skills": ("Technical", ["Weak SQL fundamentals", "Could not explain model tradeoffs",
                                        "Struggled with statistics basics", "Weak Python problem solving"]),
    "Experience Mismatch": ("Screen", ["Experience did not match role", "Relevant experience too limited",
                                        "Background not aligned with job requirements"]),
    "Communication": ("Behavioral", ["Examples lacked clarity", "Hard to follow project explanations"]),
    "Compensation Expectations": ("Screen", ["Expectations far above role band"]),
}
ADVANCE = {"Screen": ["Clear motivation and relevant background", "Good fit on screen"],
           "Technical": ["Good Python and modeling", "Strong model evaluation skills",
                         "Good statistics and experimentation"],
           "Behavioral": ["Clear examples and collaboration", "Strong ownership stories"]}


def wchoice(rng, items, weights):
    return rng.choices(items, weights=weights, k=1)[0]


def d(x: date) -> str:
    return x.isoformat()


def main() -> None:
    rng = random.Random(SEED)
    cands, offers, feedback = [], [], []
    fid = 0
    roles = [wchoice(rng, list(ROLES), list(ROLES.values())) for _ in range(N_CANDIDATES)]

    for i, role in enumerate(roles, start=1):
        cid = f"C{i:03d}"
        sw = SOURCE_BY_ROLE[role]
        source = wchoice(rng, list(sw), list(sw.values()))
        start, end = (date(2026, 8, 12), date(2026, 9, 20)) if role == "MLOps Engineer" \
            else (date(2026, 8, 10), date(2026, 10, 20))
        app = start + timedelta(days=rng.randint(0, (end - start).days))
        if i == N_CANDIDATES:
            app = AS_OF
        age = (AS_OF - app).days

        w = list(OUTCOME_W[source])
        if age < 10:
            w = [0, 40, 0, 0, 0, 0, 60]
        else:
            if age < 18:
                w[3] = 0
            if age < 25:
                w[4] = w[5] = 0
            if role == "MLOps Engineer":
                w[3] = w[4] = w[5] = 0
            if age < 8:
                w[2] = 0
        if i == N_CANDIDATES:
            w = [0, 0, 0, 0, 0, 0, 1]
        outcome = wchoice(rng, OUTCOMES, w)

        reason, entered, offer_row = "", app, None
        if outcome == "Rejected":
            if role == "Data Analyst":
                rw = [15, 55, 20, 10]
            elif source == "Job Board":
                rw = [45, 30, 20, 5]
            else:
                rw = [55, 15, 20, 10]
            reason = wchoice(rng, list(FEEDBACK), rw)
            entered = min(AS_OF, app + timedelta(days=rng.randint(3, 20)))
            stage = "Rejected"
            fb_stage, texts = FEEDBACK[reason]
            fid += 1
            feedback.append([f"F{fid:03d}", cid, fb_stage, rng.choice(texts), "Reject", None,
                             d(entered - timedelta(days=rng.randint(0, 1)))])
        elif outcome in ("Applied", "Screen", "Interview"):
            stage = outcome
            target = SLA[stage]
            breach = rng.random() < BREACH_P[stage]
            wait = rng.randint(int(target * 1.6) + 1, int(target * 3.5)) if breach else rng.randint(0, target)
            wait = max(0, min(wait, age - 1)) if stage != "Applied" else min(wait, age)
            entered = AS_OF - timedelta(days=wait)
            if stage != "Applied":
                fid += 1
                feedback.append([f"F{fid:03d}", cid, "Screen", rng.choice(ADVANCE["Screen"]), "Advance", None,
                                 d(min(AS_OF, app + timedelta(days=rng.randint(2, 5))))])
            if stage == "Interview":
                fid += 1
                feedback.append([f"F{fid:03d}", cid, "Technical", rng.choice(ADVANCE["Technical"]), "Advance",
                                 None, d(max(app, entered - timedelta(days=rng.randint(0, 2))))])
        else:  # Offer, Hired, Withdrawn
            if outcome == "Offer":
                offer_date = max(app + timedelta(days=14), AS_OF - timedelta(days=rng.randint(1, 8)))
                delay, resp = rng.randint(3, 12), "Pending"
                entered = offer_date
            else:
                hi = max(15, min(age - 6, 40))
                offer_date = app + timedelta(days=rng.randint(14, hi))
                if outcome == "Hired":
                    delay, resp = rng.randint(2, 8), "Accepted"
                else:
                    delay, resp = rng.randint(8, 20), "Declined"
                entered = min(AS_OF, offer_date + timedelta(days=rng.randint(2, 6)))
            lo, hi_s = SALARY[role]
            salary = rng.randrange(lo, hi_s + 1, 50_000)
            decline = ""
            if resp == "Declined":
                decline = wchoice(rng, ["Compensation", "Accepted another offer", "Role scope"],
                                  [30, 55, 15] if delay > 10 else [55, 30, 15])
                if decline == "Compensation":
                    salary = rng.randrange(lo, (lo + hi_s) // 2, 50_000)
            offer_row = [None, cid, d(offer_date), resp, decline, salary,
                         d(offer_date - timedelta(days=delay))]
            stage = {"Offer": "Offer", "Hired": "Hired", "Withdrawn": "Withdrawn"}[outcome]
            for st in ("Screen", "Technical", "Behavioral"):
                fid += 1
                feedback.append([f"F{fid:03d}", cid, st, rng.choice(ADVANCE[st]), "Advance", None,
                                 d(min(AS_OF, app + timedelta(days=rng.randint(2, 12))))])

        iv = wchoice(rng, list(INTERVIEWER_W), list(INTERVIEWER_W.values()))
        cands.append([cid, role, source, stage, d(app), reason, iv, d(entered)])
        for f in feedback:
            if f[1] == cid and f[5] is None:
                f[5] = iv
        if offer_row:
            offers.append(offer_row)

    offers.sort(key=lambda r: r[2])
    for n, row in enumerate(offers, start=1):
        row[0] = f"O{n:03d}"

    load = {  # interviewer: (capacity, assigned over 4 weeks)
        "I001": (6, [8, 9, 8, 7]), "I002": (6, [5, 6, 7, 5]), "I003": (5, [7, 7, 6, 8]),
        "I004": (6, [4, 5, 4, 6]), "I005": (6, [3, 2, 4, 3]), "I006": (5, [4, 5, 5, 4]),
        "I007": (6, [6, 6, 7, 6]), "I008": (5, [2, 3, 2, 3]),
    }
    weeks = ["2026-09-28", "2026-10-05", "2026-10-12", "2026-10-19"]
    load_rows = [[iv, a, cap, wk] for iv, (cap, asg) in load.items() for a, wk in zip(asg, weeks)]

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cands, columns=["candidate_id", "role", "source", "stage", "application_date",
                                 "rejection_reason", "interviewer_id", "stage_entered_date"]
                 ).to_csv(OUT / "candidates.csv", index=False)
    pd.DataFrame(offers, columns=["offer_id", "candidate_id", "offer_date", "response", "decline_reason",
                                  "offered_salary", "final_interview_date"]
                 ).to_csv(OUT / "offers.csv", index=False)
    pd.DataFrame(feedback, columns=["feedback_id", "candidate_id", "interview_stage", "feedback", "decision",
                                    "interviewer_id", "feedback_date"]
                 ).to_csv(OUT / "interview_feedback.csv", index=False)
    pd.DataFrame(load_rows, columns=["interviewer_id", "interviews_assigned", "capacity_per_week",
                                     "week_start"]).to_csv(OUT / "interviewer_load.csv", index=False)
    print(f"candidates={len(cands)} offers={len(offers)} feedback={len(feedback)} load={len(load_rows)}")


if __name__ == "__main__":
    main()

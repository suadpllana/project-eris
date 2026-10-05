"""
grade.py - Beijing Virtual Air-Quality Stations

Metric: Mean Column-wise Root Mean Squared Logarithmic Error (MCRMSLE),
lower is better.

    RMSLE_j = sqrt( mean_{i in V_j} ( log(1 + p_ij) - log(1 + y_ij) )^2 )
    MCRMSLE = mean over the six pollutants j of RMSLE_j

V_j is the set of test rows whose ground truth for pollutant j was actually
measured; hours where the station's instrument reported nothing are skipped
for that pollutant only. Every prediction must still be present and valid.
"""
import numpy as np
import pandas as pd

ID_COL = "id"
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
MAX_SCORE = 100.0


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    sub = _validate(submission, answers)
    ans = answers[[ID_COL] + POLLUTANTS].copy()
    ans[ID_COL] = ans[ID_COL].astype(str).str.strip()
    merged = ans.merge(sub, on=ID_COL, how="left", suffixes=("_true", "_pred"))

    scores = []
    for col in POLLUTANTS:
        y_true = pd.to_numeric(merged[f"{col}_true"], errors="coerce").to_numpy(dtype=float)
        y_pred = merged[f"{col}_pred"].to_numpy(dtype=float)
        mask = np.isfinite(y_true)
        if not mask.any():
            continue
        err = np.log1p(y_pred[mask]) - np.log1p(y_true[mask])
        scores.append(float(np.sqrt(np.mean(err ** 2))))
    if not scores:
        raise ValueError("Answers contain no measured values.")
    return min(float(np.mean(scores)), MAX_SCORE)


def _validate(submission: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(submission, pd.DataFrame):
        raise ValueError("Submission must be a pandas DataFrame.")
    sub = submission.copy()
    sub.columns = [str(c).strip() for c in sub.columns]

    required = [ID_COL] + POLLUTANTS
    missing = [c for c in required if c not in sub.columns]
    if missing:
        raise ValueError(f"Submission is missing column(s) {missing}; expected {required}.")
    sub = sub[required]

    if len(sub) != len(answers):
        raise ValueError(f"Submission has {len(sub)} rows; expected {len(answers)} "
                         f"(one per row of test.csv).")

    sub[ID_COL] = sub[ID_COL].astype(str).str.strip()
    if sub[ID_COL].duplicated().any():
        dupes = sub.loc[sub[ID_COL].duplicated(), ID_COL].head(5).tolist()
        raise ValueError(f"Submission contains duplicate ids, e.g. {dupes}.")
    expected = set(answers[ID_COL].astype(str).str.strip())
    got = set(sub[ID_COL])
    if got != expected:
        raise ValueError(f"Submission ids do not match test ids. "
                         f"Unknown (sample): {sorted(got - expected)[:5]}; "
                         f"missing (sample): {sorted(expected - got)[:5]}.")

    for col in POLLUTANTS:
        values = pd.to_numeric(sub[col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Column '{col}' must be numeric with no missing values.")
        values = values.astype(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Column '{col}' contains inf values.")
        if (values < 0).any():
            raise ValueError(f"Column '{col}' must be non-negative (concentrations).")
        sub[col] = values
    return sub

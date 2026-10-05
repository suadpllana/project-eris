"""
grade.py - Cold-Start Air-Quality Forecasting at Unmonitored Beijing Sites

Metric: Root Mean Squared Logarithmic Error over all test rows (lower is better)

    RMSLE = sqrt( mean_i ( log(1 + p_i) - log(1 + y_i) )^2 )

Each row is one (forecast episode, target station, hour, pollutant). The log
transform makes errors relative, so the six pollutants (CO in the thousands of
ug/m3, SO2 often in single digits) contribute on a comparable scale.

The score is computed on the ids present in `answers`. The submission must
contain every one of those ids exactly once; additional ids that are not in
`answers` are ignored, which lets the platform score public/private subsets.
"""
import numpy as np
import pandas as pd

ID_COL = "id"
TARGET = "value"
MAX_SCORE = 100.0


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    if not isinstance(submission, pd.DataFrame):
        raise ValueError("Submission must be a pandas DataFrame.")
    sub = submission.copy()
    sub.columns = [str(c).strip() for c in sub.columns]
    missing = [c for c in (ID_COL, TARGET) if c not in sub.columns]
    if missing:
        raise ValueError(f"Submission is missing column(s) {missing}; expected ['{ID_COL}', '{TARGET}'].")
    sub = sub[[ID_COL, TARGET]]
    sub[ID_COL] = sub[ID_COL].astype(str).str.strip()
    if sub[ID_COL].duplicated().any():
        dupes = sub.loc[sub[ID_COL].duplicated(), ID_COL].head(5).tolist()
        raise ValueError(f"Submission contains duplicate ids, e.g. {dupes}.")

    ans = answers[[ID_COL, TARGET]].copy()
    ans[ID_COL] = ans[ID_COL].astype(str).str.strip()
    absent = sorted(set(ans[ID_COL]) - set(sub[ID_COL]))
    if absent:
        raise ValueError(f"Submission is missing {len(absent)} required id(s), e.g. {absent[:5]}.")

    merged = ans.merge(sub, on=ID_COL, how="left", suffixes=("_true", "_pred"))
    pred = pd.to_numeric(merged[f"{TARGET}_pred"], errors="coerce")
    if pred.isna().any():
        raise ValueError(f"'{TARGET}' must be numeric with no missing values.")
    pred = pred.to_numpy(dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError(f"'{TARGET}' contains inf values.")
    if (pred < 0).any():
        raise ValueError(f"'{TARGET}' must be non-negative (concentrations in ug/m3).")

    true = pd.to_numeric(merged[f"{TARGET}_true"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(true).all():
        raise ValueError("Answers contain missing or non-numeric values.")

    rmsle = float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2)))
    return min(rmsle, MAX_SCORE)

#!/usr/bin/env python3
"""
eris_check.py - run Shipd Eris pre-submission checks locally, before pasting anything.

Each check reproduces a platform check that failed during earlier submissions
(see CLAUDE.md, "Pre-submission check lessons"). Run it on a challenge folder:

    python tools/eris_check.py challenges/<slug>

The folder must contain challenge/prepare.py, challenge/grade.py,
challenge/config.yaml and dataset/raw/. Exit code 0 means no FAIL (warnings may
remain); 1 means at least one check would fail on the platform.
"""
import argparse
import hashlib
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = []


def report(level, name, msg):
    RESULTS.append((level, name, msg))
    print(f"[{level:4s}] {name}: {msg}")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(mod)
    return mod


def read_config(path):
    cfg = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line and not line.strip().startswith("#"):
            k, v = line.split(":", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def tree_hashes(root):
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def expect_raise(grade, sub, ans, label):
    try:
        grade(sub, ans)
    except Exception as e:  # noqa: BLE001 - any exception is a rejection
        report("PASS", "Evaluator contract", f"rejects {label} ({type(e).__name__})")
        return
    report("FAIL", "Evaluator contract", f"accepted a malformed submission: {label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("challenge_dir")
    args = ap.parse_args()
    root = Path(args.challenge_dir).resolve()
    prep = root / "challenge" / "prepare.py"
    grd = root / "challenge" / "grade.py"
    cfg = read_config(root / "challenge" / "config.yaml")
    raw = root / "dataset" / "raw"
    work = Path(tempfile.mkdtemp(prefix="eris_check_"))

    # 1. Raw input as the platform passes it: archives are auto-extracted.
    raw_in = work / "raw"
    raw_in.mkdir()
    for f in raw.iterdir():
        if f.suffix == ".zip":
            shutil.unpack_archive(str(f), str(raw_in))
        elif f.is_file():
            shutil.copy(f, raw_in)
    prepare = load_module(prep, "prepare_mod").prepare

    # 2. Determinism: two runs must be byte-identical.
    for run in ("a", "b"):
        prepare(raw_in, work / run / "public", work / run / "private")
    ha, hb = tree_hashes(work / "a"), tree_hashes(work / "b")
    if ha == hb:
        report("PASS", "Reproducibility", f"two prepare runs identical ({len(ha)} files)")
    else:
        diff = sorted(k for k in set(ha) | set(hb) if ha.get(k) != hb.get(k))
        report("FAIL", "Reproducibility", f"files differ between runs: {diff[:5]}")

    pub, prv = work / "a" / "public", work / "a" / "private"
    need = [pub / "train.csv", pub / "test.csv", pub / "sample_submission.csv", prv / "answers.csv"]
    missing = [str(p.relative_to(work / "a")) for p in need if not p.exists()]
    if missing:
        report("FAIL", "Required files", f"missing {missing}")
        return finish()
    report("PASS", "Required files", "train.csv, test.csv, sample_submission.csv, answers.csv present")

    train = pd.read_csv(pub / "train.csv", low_memory=False)
    test = pd.read_csv(pub / "test.csv", low_memory=False)
    sample = pd.read_csv(pub / "sample_submission.csv")
    answers = pd.read_csv(prv / "answers.csv")
    id_col = answers.columns[0]
    targets = [c for c in answers.columns if c != id_col]

    # 3. Prepared Data Integrity: train columns == test columns + target column(s).
    extra_in_train = [c for c in train.columns if c not in test.columns]
    extra_in_test = [c for c in test.columns if c not in train.columns]
    if extra_in_test or set(extra_in_train) - set(targets):
        report("FAIL", "Prepared Data Integrity",
               f"train/test feature columns must match. only-in-test={extra_in_test}, "
               f"only-in-train (non-target)={sorted(set(extra_in_train) - set(targets))}")
    else:
        report("PASS", "Prepared Data Integrity", f"train = test columns + {extra_in_train}")

    # 4. answers.csv must have no missing targets.
    if answers[targets].isna().any().any():
        bad = answers[targets].columns[answers[targets].isna().any()].tolist()
        report("FAIL", "Prepared Data Integrity", f"answers.csv contains missing target values: {bad}")
    else:
        report("PASS", "Prepared Data Integrity", "answers.csv has no missing targets")

    # 5. Target Recoverability needs ONE supported target column.
    if len(targets) != 1:
        report("FAIL", "Target Recoverability",
               f"answers has {len(targets)} target columns {targets}; use one column "
               f"(long format: one row per id x output)")
    else:
        report("PASS", "Target Recoverability", f"single target column '{targets[0]}'")

    # 6. ids: unique and consistent across test / sample / answers.
    for name, df in (("test", test), ("sample_submission", sample), ("answers", answers)):
        if id_col not in df.columns:
            report("FAIL", "IDs", f"{name} has no '{id_col}' column")
        elif df[id_col].duplicated().any():
            report("FAIL", "IDs", f"{name} has duplicate ids")
    if id_col in test.columns and set(test[id_col]) == set(answers[id_col]) == set(sample[id_col]):
        report("PASS", "IDs", f"{len(answers)} ids match across test, sample and answers")
    else:
        report("FAIL", "IDs", "id sets differ between test / sample_submission / answers")
    if list(sample.columns) != list(answers.columns):
        report("FAIL", "IDs", f"sample columns {list(sample.columns)} != answers columns {list(answers.columns)}")
    if id_col in train.columns and set(train[id_col]) & set(test[id_col]):
        report("FAIL", "Split leakage", "some test ids also appear in train")
    leaked = [t for t in targets if t in test.columns]
    if leaked:
        report("FAIL", "Split leakage", f"target column(s) {leaked} present in test.csv")

    # 7. Missing-value profile (platform warns at >= 50% missing in a column).
    for csv in sorted(pub.glob("*.csv")):
        df = pd.read_csv(csv, nrows=200_000, low_memory=False)
        heavy = [c for c in df.columns if df[c].isna().mean() >= 0.5]
        if heavy:
            report("WARN", "Missing-value profile",
                   f"{csv.name}: >=50% missing in {heavy[:8]} (explain in description if by design)")

    # 8. Evaluator contract.
    grade = load_module(grd, "grade_mod").grade
    direction = cfg.get("direction", "minimize")
    best = float(cfg.get("minimum" if direction == "minimize" else "maximum", 0))
    better = (lambda a, b: a < b) if direction == "minimize" else (lambda a, b: a > b)
    try:
        perfect = grade(answers.copy(), answers.copy())
        report("PASS" if abs(perfect - best) < 1e-9 else "FAIL", "Evaluator contract",
               f"known-answer submission scores {perfect} (expected {best})")
        shuffled = grade(answers.sample(frac=1, random_state=0), answers.copy())
        report("PASS" if abs(shuffled - perfect) < 1e-9 else "FAIL", "Evaluator contract",
               "row order does not change the score")
        s_sample = grade(sample.copy(), answers.copy())
        report("PASS" if better(perfect, s_sample) else "FAIL", "Evaluator contract",
               f"known answer ranks above sample baseline ({perfect} vs {s_sample})")
        half = answers.sample(frac=0.5, random_state=1)
        grade(sample.copy(), half)
        grade(sample.copy(), answers.drop(half.index))
        report("PASS", "Evaluator contract", "scores public/private subsets of answers")
        if grade(sample.copy(), answers.copy()) != s_sample:
            report("FAIL", "Evaluator contract", "grader is not deterministic")
    except Exception as e:  # noqa: BLE001
        report("FAIL", "Evaluator contract", f"grader raised on a valid submission: {e!r}")
    t0 = targets[0]
    expect_raise(grade, answers.iloc[:-1].copy(), answers, "a missing row")
    expect_raise(grade, pd.concat([answers, answers.iloc[:1]]), answers, "a duplicate id")
    expect_raise(grade, answers.drop(columns=[t0]), answers, "a missing column")
    expect_raise(grade, answers.assign(**{t0: np.nan}), answers, "NaN predictions")
    expect_raise(grade, answers.assign(**{t0: np.inf}) if pd.api.types.is_numeric_dtype(answers[t0])
                 else answers.assign(**{t0: None}), answers, "inf/None predictions")

    size = sum(p.stat().st_size for p in pub.rglob("*") if p.is_file()) / 1e6
    report("INFO", "Size", f"public/ is {size:.1f} MB")
    shutil.rmtree(work, ignore_errors=True)
    return finish()


def finish():
    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    print(f"\n{len(fails)} FAIL, {len(warns)} WARN")
    print("Not checked locally (platform-only): Domain Routing, Novelty, Problem Originality, "
          "agent difficulty. See CLAUDE.md for how to design for them.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

"""
prepare.py - Beijing Virtual Air-Quality Stations

Builds the public/private split for the challenge from the raw UCI
"Beijing Multi-Site Air-Quality" upload (12 hourly station CSV files,
2013-03-01 00:00 to 2017-02-28 23:00).

Split design (spatial hold-out, not temporal):
- 8 "network" stations go to public/train.csv with meteorology AND all six
  pollutant measurements for the full four years.
- 4 held-out stations go to public/test.csv with meteorology only. Their
  pollutant measurements are the private answers.
The task is to reconstruct the pollutant series at sites that have no
pollutant monitors, using local weather and the concurrent readings of the
surrounding network.

Determinism: no randomness is used. The station split is fixed by name and
rows are written sorted by (station, timestamp).
"""
from pathlib import Path
import zipfile

import pandas as pd

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
METEOROLOGY = ["TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM"]
TIME_COLS = ["year", "month", "day", "hour"]

TEST_STATIONS = ["Gucheng", "Huairou", "Nongzhanguan", "Wanliu"]
TRAIN_STATIONS = ["Aotizhongxin", "Changping", "Dingling", "Dongsi",
                  "Guanyuan", "Shunyi", "Tiantan", "Wanshouxigong"]

ROWS_PER_STATION = 35064  # 1461 days x 24 hours
FILE_PATTERN = "PRSA_Data_*_20130301-20170228.csv"


def _load_raw(raw: Path) -> pd.DataFrame:
    """Read the 12 station CSVs from raw/ (extracted folder or the original zip)."""
    files = sorted(raw.rglob(FILE_PATTERN))
    if files:
        frames = [pd.read_csv(f, na_values=["NA"], keep_default_na=False) for f in files]
    else:
        zips = sorted(raw.rglob("PRSA2017_Data_20130301-20170228.zip"))
        if not zips:
            raise FileNotFoundError(f"No station CSVs or source zip found under {raw}")
        frames = []
        with zipfile.ZipFile(zips[0]) as zf:
            for name in sorted(n for n in zf.namelist() if n.endswith(".csv")):
                with zf.open(name) as fh:
                    frames.append(pd.read_csv(fh, na_values=["NA"], keep_default_na=False))
    df = pd.concat(frames, ignore_index=True)
    # Empty strings (if any) should also be null.
    df = df.replace({"": pd.NA})
    return df


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw, public, private = Path(raw), Path(public), Path(private)
    df = _load_raw(raw)

    stations = sorted(df["station"].unique())
    if stations != sorted(TEST_STATIONS + TRAIN_STATIONS):
        raise ValueError(f"Unexpected station set: {stations}")
    counts = df["station"].value_counts()
    if (counts != ROWS_PER_STATION).any():
        raise ValueError(f"Expected {ROWS_PER_STATION} rows per station, got {counts.to_dict()}")
    if df.duplicated(["station"] + TIME_COLS).any():
        raise ValueError("Duplicate (station, timestamp) rows in raw data")

    for col in POLLUTANTS + ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]:
        df[col] = pd.to_numeric(df[col])
    for col in TIME_COLS:
        df[col] = df[col].astype(int)

    stamp = (df["year"].astype(str) + df["month"].map("{:02d}".format)
             + df["day"].map("{:02d}".format) + df["hour"].map("{:02d}".format))
    df.insert(0, "id", df["station"] + "_" + stamp)
    df = df.drop(columns=["No"]).sort_values(["station"] + TIME_COLS).reset_index(drop=True)

    base_cols = ["id", "station"] + TIME_COLS + METEOROLOGY
    is_test = df["station"].isin(TEST_STATIONS)
    train = df.loc[~is_test, base_cols + POLLUTANTS]
    test = df.loc[is_test, base_cols]
    answers = df.loc[is_test, ["id"] + POLLUTANTS]

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    train.to_csv(public / "train.csv", index=False)
    test.to_csv(public / "test.csv", index=False)

    sample = answers[["id"]].copy()
    for col in POLLUTANTS:
        # Training-network median: a valid but deliberately weak submission.
        sample[col] = float(train[col].median())
    sample.to_csv(public / "sample_submission.csv", index=False)
    answers.to_csv(private / "answers.csv", index=False)


if __name__ == "__main__":
    import sys

    prepare(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

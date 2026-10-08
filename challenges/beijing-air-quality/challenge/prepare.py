"""
prepare.py - From-Scratch Neural Forecasting of Air Quality at Unmonitored Sites

Builds the public/private split from the raw upload (12 hourly monitoring-site
CSV files, 2013-03-01 00:00 to 2017-02-28 23:00).

Design
- Site names are replaced by anonymous ids S01..S12, assigned in salted-hash order,
  so no place name appears in any prepared file.
- 8 "network" sites have pollutant monitors. 4 "target" sites are treated as
  unmonitored: their pollutant readings are never published, only their weather.
- Training period 2013-03-01 .. 2016-02-29:
    weather.csv - hourly weather for all 12 sites (no pollutants, so nothing is
                  duplicated with train.csv).
    train.csv   - every measured network-site pollutant value, in exactly the
                  same long layout as test.csv plus `value`.
- Test period 2016-03-01 .. 2017-02-28, released as 73 independent forecast
  episodes in test_context.csv: 72 context hours (network pollutants plus weather
  at all 12 sites) followed by a 24-hour target day with weather only.
  Episodes are laid out as [3 context days][1 target day][1 unused day], carry
  no calendar date, and have hashed ids, so they cannot be chained.
- Only measured target values become test rows, so answers.csv has no gaps.

Determinism: no randomness; ids come from SHA-256 and every file is sorted.
"""
from pathlib import Path
import hashlib
import zipfile

import numpy as np
import pandas as pd

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM"]
TIME_COLS = ["year", "month", "day", "hour"]

TARGET_STATIONS = ["Gucheng", "Huairou", "Nongzhanguan", "Wanliu"]
NETWORK_STATIONS = ["Aotizhongxin", "Changping", "Dingling", "Dongsi",
                    "Guanyuan", "Shunyi", "Tiantan", "Wanshouxigong"]

TRAIN_END = pd.Timestamp("2016-02-29 23:00")
TEST_START = pd.Timestamp("2016-03-01 00:00")
TEST_END_DAY = pd.Timestamp("2017-02-28")
CONTEXT_HOURS = 72
TARGET_HOURS = 24
STRIDE_DAYS = 5  # 3 context days + 1 target day + 1 unused day

ROWS_PER_STATION = 35064
FILE_PATTERN = "PRSA_Data_*_20130301-20170228.csv"
SALT = "eris-beijing-forecast-v1"
SITE_SALT = "eris-site-anon-v1"


def _load_raw(raw: Path) -> pd.DataFrame:
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
    return pd.concat(frames, ignore_index=True).replace({"": np.nan})


def _hash(salt: str, text: str) -> str:
    return hashlib.sha256(f"{salt}|{text}".encode()).hexdigest()


def _make_id(forecast_id, site, rel_hour, pollutant):
    return forecast_id + "_" + site + "_h" + rel_hour.map("{:02d}".format) + "_" + pollutant


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw, public, private = Path(raw), Path(public), Path(private)
    df = _load_raw(raw)

    stations = sorted(df["station"].unique())
    if stations != sorted(TARGET_STATIONS + NETWORK_STATIONS):
        raise ValueError(f"Unexpected station set: {stations}")
    if (df["station"].value_counts() != ROWS_PER_STATION).any():
        raise ValueError("Expected 35,064 hourly rows per station")
    if df.duplicated(["station"] + TIME_COLS).any():
        raise ValueError("Duplicate (station, timestamp) rows in raw data")

    # Anonymise site names: S01..S12 in salted-hash order (mixes network and target sites).
    site_of = {s: f"S{i + 1:02d}" for i, s in enumerate(sorted(stations, key=lambda s: _hash(SITE_SALT, s)))}
    target_sites = sorted(site_of[s] for s in TARGET_STATIONS)
    df["site"] = df["station"].map(site_of)

    for col in POLLUTANTS + ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]:
        df[col] = pd.to_numeric(df[col])
    for col in TIME_COLS:
        df[col] = df[col].astype(int)
    df["ts"] = pd.to_datetime(df[TIME_COLS])
    df = df.drop(columns=["No", "station"]).sort_values(["site", "ts"]).reset_index(drop=True)
    is_target = df["site"].isin(target_sites)

    # ---------------- training period ----------------
    hist = df[df["ts"] <= TRAIN_END]
    weather = hist[["site"] + TIME_COLS + WEATHER]

    lab = hist[~is_target.loc[hist.index]].melt(
        id_vars=["site", "ts"], value_vars=POLLUTANTS,
        var_name="pollutant", value_name="value").dropna(subset=["value"])
    lab["forecast_id"] = "T" + lab["ts"].dt.strftime("%Y%m%d")
    lab["rel_hour"] = lab["ts"].dt.hour
    lab["hour"] = lab["ts"].dt.hour
    lab.insert(0, "id", _make_id(lab["forecast_id"], lab["site"], lab["rel_hour"], lab["pollutant"]))
    train = lab[["id", "forecast_id", "site", "rel_hour", "hour", "pollutant", "value"]]
    train = train.sort_values("id").reset_index(drop=True)

    # ---------------- test period: independent forecast episodes ----------------
    origins = []
    start = TEST_START
    while start + pd.Timedelta(days=3) <= TEST_END_DAY:
        origins.append(start + pd.Timedelta(hours=CONTEXT_HOURS))  # 00:00 of target day
        start += pd.Timedelta(days=STRIDE_DAYS)
    order = sorted(origins, key=lambda o: _hash(SALT, f"{o:%Y-%m-%d}"))
    fid = {o: f"F{i + 1:03d}" for i, o in enumerate(order)}

    indexed = df.set_index(["site", "ts"]).sort_index()
    sites = sorted(site_of.values())
    ctx_rows, tgt_rows = [], []
    for origin in origins:
        hours = pd.date_range(origin - pd.Timedelta(hours=CONTEXT_HOURS),
                              origin + pd.Timedelta(hours=TARGET_HOURS - 1), freq="h")
        rel = ((hours - origin) / pd.Timedelta(hours=1)).astype(int)
        for site in sites:
            block = indexed.loc[site].reindex(hours)
            ctx = pd.DataFrame({"forecast_id": fid[origin], "site": site, "rel_hour": rel,
                                "month": hours.month, "weekday": hours.dayofweek, "hour": hours.hour})
            for col in WEATHER:
                ctx[col] = block[col].to_numpy()
            for col in POLLUTANTS:
                values = block[col].to_numpy(dtype=float).copy()
                if site in target_sites:
                    values[:] = np.nan  # never observed at unmonitored sites
                else:
                    values[rel >= 0] = np.nan  # the future is hidden
                ctx[col] = values
            ctx_rows.append(ctx)

            if site in target_sites:
                target = block.loc[block.index >= origin]
                trel = pd.Series(((target.index - origin) / pd.Timedelta(hours=1)).astype(int))
                for col in POLLUTANTS:
                    t = pd.DataFrame({"forecast_id": fid[origin], "site": site, "rel_hour": trel,
                                      "hour": target.index.hour, "pollutant": col,
                                      "value": target[col].to_numpy(dtype=float)})
                    tgt_rows.append(t[t["value"].notna()])

    context = pd.concat(ctx_rows, ignore_index=True).sort_values(
        ["forecast_id", "site", "rel_hour"]).reset_index(drop=True)
    targets = pd.concat(tgt_rows, ignore_index=True)
    targets.insert(0, "id", _make_id(targets["forecast_id"], targets["site"],
                                     targets["rel_hour"], targets["pollutant"]))
    targets = targets.sort_values("id").reset_index(drop=True)
    if targets["id"].duplicated().any():
        raise ValueError("Duplicate target ids")

    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    weather.to_csv(public / "weather.csv", index=False)
    train.to_csv(public / "train.csv", index=False)
    context.to_csv(public / "test_context.csv", index=False)
    targets[["id", "forecast_id", "site", "rel_hour", "hour", "pollutant"]].to_csv(
        public / "test.csv", index=False)

    medians = train.groupby("pollutant")["value"].median()
    sample = pd.DataFrame({"id": targets["id"],
                           "value": targets["pollutant"].map(medians).astype(float)})
    sample.to_csv(public / "sample_submission.csv", index=False)
    targets[["id", "value"]].to_csv(private / "answers.csv", index=False)


if __name__ == "__main__":
    import sys

    prepare(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

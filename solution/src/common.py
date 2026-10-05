# %% [markdown]
# ## Setup
# Read from `./dataset/public/`, write only to `./working/`.

# %%
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmean of all-NaN slices
warnings.filterwarnings("ignore", category=UserWarning)
T0 = time.time()
DATA = "./dataset/public/"
OUT = "./working/"
os.makedirs(OUT, exist_ok=True)

POL = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WXCOLS = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM", "u", "v"]
DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
CTX, HOR = 72, 24  # 72 context hours, 24 forecast hours

train = pd.read_csv(DATA + "history.csv")      # continuous hourly training period
labels = pd.read_csv(DATA + "train.csv")      # labelled next-day rows (same layout as test.csv)
ctx = pd.read_csv(DATA + "test_context.csv")
test = pd.read_csv(DATA + "test.csv")

# Wind direction is circular: convert (wd, WSPM) to u/v wind components.
ANG = {d: i * np.pi / 8 for i, d in enumerate(DIRS)}
for d in (train, ctx):
    a = d["wd"].map(ANG).astype(float)
    d["u"] = -d["WSPM"] * np.sin(a)
    d["v"] = -d["WSPM"] * np.cos(a)

train["ts"] = pd.to_datetime(train[["year", "month", "day", "hour"]])
NET = sorted(train.loc[train[POL].notna().any(axis=1), "station"].unique())  # monitored
TGT = sorted(test.station.unique())                                          # unmonitored
ALL = sorted(train.station.unique())
print("network stations:", NET)
print("target stations: ", TGT)
print("train hours:", train.ts.min(), "->", train.ts.max(), "| test episodes:", ctx.forecast_id.nunique())

# %% [markdown]
# ## Episode tensors
# Test data arrives as independent episodes: 72 context hours plus 24 target hours. The
# training period is continuous, so we cut it into the **same** episode shape: one episode
# per midnight forecast origin. Features are then computed by one function for both.
#
# Tensors have shape (episodes, 96 hours, stations). Network pollutants are stored as
# log1p, the scale of the metric, and blanked for target hours so nothing from the
# future can leak into features.

# %%
TIMES = pd.DatetimeIndex(sorted(train.ts.unique()))
def wide(col, stations):
    return train.pivot(index="ts", columns="station", values=col).reindex(TIMES)[stations].to_numpy(float)

origins = np.where((TIMES.hour == 0) & (np.arange(len(TIMES)) >= CTX)
                   & (np.arange(len(TIMES)) + HOR <= len(TIMES)))[0]
IDX = origins[:, None] + np.arange(-CTX, HOR)[None, :]            # (E, 96)
TR_ORIGIN = TIMES[origins]
TR_MONTH, TR_WDAY = TIMES.month.to_numpy()[IDX], TIMES.dayofweek.to_numpy()[IDX]
TR_P = {p: np.log1p(wide(p, NET))[IDX] for p in POL}                # (E, 96, 8)  incl. future
TR_W = {w: wide(w, ALL)[IDX] for w in WXCOLS}                        # (E, 96, 12)

FIDS = sorted(ctx.forecast_id.unique())
def ctx_tensor(col, stations, log=False):
    w = ctx.pivot_table(index=["forecast_id", "rel_hour"], columns="station", values=col, dropna=False)
    w = w.reindex(pd.MultiIndex.from_product([FIDS, range(-CTX, HOR)]))[stations].to_numpy(float)
    w = w.reshape(len(FIDS), CTX + HOR, len(stations))
    return np.log1p(w) if log else w
TE_P = {p: ctx_tensor(p, NET, log=True) for p in POL}
TE_W = {w: ctx_tensor(w, ALL) for w in WXCOLS}
meta = ctx.drop_duplicates(["forecast_id", "rel_hour"]).set_index(["forecast_id", "rel_hour"])
meta = meta.reindex(pd.MultiIndex.from_product([FIDS, range(-CTX, HOR)]))
TE_MONTH = meta["month"].to_numpy().reshape(len(FIDS), -1)
TE_WDAY = meta["weekday"].to_numpy().reshape(len(FIDS), -1)
print("train episodes:", len(origins), "| test episodes:", len(FIDS), f"[{time.time() - T0:.0f}s]")

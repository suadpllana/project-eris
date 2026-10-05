# %% [markdown]
# ## Setup
# Paths follow the platform convention: read from `./dataset/public/`, write to `./working/`.

# %%
import os
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)
T0 = time.time()
DATA = "./dataset/public/"
OUT = "./working/"
os.makedirs(OUT, exist_ok=True)

POL = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
MET = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

train = pd.read_csv(DATA + "train.csv")
test = pd.read_csv(DATA + "test.csv")
for d in (train, test):
    d["ts"] = pd.to_datetime(d[["year", "month", "day", "hour"]])

NET = sorted(train.station.unique())   # 8 labelled network stations
HELD = sorted(test.station.unique())   # 4 stations to reconstruct
print("train", train.shape, "stations:", NET)
print("test ", test.shape, "stations:", HELD)

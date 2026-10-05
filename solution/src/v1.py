# %% [markdown]
# ## 1. Quick look at the data
# The pollutant columns are right-skewed and have multi-hour gaps. The metric is RMSE in
# log1p space, so everything below is modelled on `log1p(concentration)`.

# %%
print(train[POL].describe().T[["mean", "50%", "max"]])
print("\nMissing fraction per pollutant (train):")
print(train[POL].isna().mean().round(4))
print("\nMissing fraction per column (test):")
print(test[MET + ["wd"]].isna().mean().round(4))

# %% [markdown]
# ## 2. Wide hourly matrices
# One matrix per pollutant: rows = hours, columns = network stations, values = log1p
# concentration. Missing readings stay NaN; they are never filled with 0.

# %%
TIMES = pd.DatetimeIndex(sorted(train.ts.unique()))
LOGW = {p: np.log1p(train.pivot(index="ts", columns="station", values=p).reindex(TIMES))
        for p in POL}
allrows = pd.concat([train, test], ignore_index=True)

# %% [markdown]
# ## 3. Baseline under leave-one-station-out (LOSO) validation
# The test set is four *unseen* stations, so validation must hold out whole stations.
# A random row split would leak each station's level through autocorrelated neighbouring
# hours. Baseline: for station s, predict the hourly log-mean of the *other* network stations.

# %%
def rmsle_log(pred_log, true_log):
    m = ~np.isnan(true_log) & ~np.isnan(pred_log)
    return float(np.sqrt(np.mean((pred_log[m] - true_log[m]) ** 2)))

base_scores = {}
for p in POL:
    errs = []
    for s in NET:
        others = [o for o in NET if o != s]
        pred = LOGW[p][others].mean(axis=1).to_numpy()
        errs.append(rmsle_log(pred, LOGW[p][s].to_numpy()))
    base_scores[p] = np.mean(errs)
print("LOSO network-mean baseline:", {k: round(v, 4) for k, v in base_scores.items()})
print("LOSO network-mean MCRMSLE: %.4f" % np.mean(list(base_scores.values())))

# %% [markdown]
# ## 4. Features (version 1)
# For every station we build one row per hour:
# * calendar: hour, month, day-of-year, weekday, time index
# * own meteorology, with wind direction encoded as a circle (sin/cos and u/v components).
#   An ordinal 0-15 code would put NNW and N 15 apart.
# * network pollutant aggregates (mean/std/min/max of log1p) at the same hour, computed
#   from the network stations **other than the station itself**. Including a training
#   station's own reading would leak its label into the features.

# %%
WD_ANGLE = {d: i * np.pi / 8 for i, d in enumerate(DIRS)}

def station_features(s):
    others = [o for o in NET if o != s]
    g = allrows[allrows.station == s].set_index("ts").reindex(TIMES)
    f = {"id": g["id"].to_numpy(), "station": np.full(len(TIMES), s)}
    f["hour"] = TIMES.hour; f["month"] = TIMES.month
    f["doy"] = TIMES.dayofyear; f["dow"] = TIMES.dayofweek
    f["t"] = (TIMES - TIMES[0]).days.to_numpy()
    for m in MET:
        f[m] = g[m].to_numpy()
    ang = g["wd"].map(WD_ANGLE).to_numpy(dtype=float)
    f["wd_sin"], f["wd_cos"] = np.sin(ang), np.cos(ang)
    f["u"], f["v"] = -g["WSPM"].to_numpy() * np.sin(ang), -g["WSPM"].to_numpy() * np.cos(ang)
    f["dew_dep"] = f["TEMP"] - f["DEWP"]
    for p in POL:
        w = LOGW[p][others]
        f[f"net_mean_{p}"] = w.mean(axis=1).to_numpy()
        f[f"net_std_{p}"] = w.std(axis=1).to_numpy()
        f[f"net_min_{p}"] = w.min(axis=1).to_numpy()
        f[f"net_max_{p}"] = w.max(axis=1).to_numpy()
        f[f"y_{p}"] = LOGW[p][s].to_numpy() if s in NET else np.full(len(TIMES), np.nan)
    return pd.DataFrame(f)

FEAT = pd.concat([station_features(s) for s in NET + HELD], ignore_index=True)
COLS = [c for c in FEAT.columns if c not in ("id", "station") and not c.startswith("y_")]
TR = FEAT[FEAT.station.isin(NET)].reset_index(drop=True)
TE = FEAT[FEAT.station.isin(HELD)].reset_index(drop=True)
print("feature matrix:", TR.shape, TE.shape, "| features:", len(COLS))

# %% [markdown]
# ## 5. LightGBM per pollutant with LOSO folds
# 8 folds, each holding out one network station. Rows with a missing label are dropped for
# that pollutant only. The test prediction is the average of the 8 fold models.

# %%
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=os.cpu_count(), seed=0)

def fit_loso(TR, TE, cols, params, rounds=2000):
    cv, preds = {}, {}
    for p in POL:
        y = TR[f"y_{p}"].to_numpy()
        ok = ~np.isnan(y)
        oof = np.full(len(TR), np.nan)
        pt = np.zeros(len(TE))
        for s in NET:
            va = (TR.station == s).to_numpy()
            dtr = lgb.Dataset(TR.loc[ok & ~va, cols], y[ok & ~va])
            dva = lgb.Dataset(TR.loc[ok & va, cols], y[ok & va])
            m = lgb.train(params, dtr, rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[ok & va] = m.predict(TR.loc[ok & va, cols], num_iteration=m.best_iteration)
            pt += m.predict(TE[cols], num_iteration=m.best_iteration) / len(NET)
        cv[p] = rmsle_log(np.clip(oof, 0, None), y)
        preds[p] = np.clip(pt, 0, None)
        print(f"{p:6s} LOSO RMSLE {cv[p]:.4f}  (baseline {base_scores[p]:.4f})  "
              f"[{time.time() - T0:.0f}s]")
    print("LOSO MCRMSLE: %.4f" % np.mean(list(cv.values())))
    return cv, preds

cv1, pred_log = fit_loso(TR, TE, COLS, PARAMS)

# %% [markdown]
# ## 6. Write the submission
# Back-transform with expm1, clip at 0, one row per test id.

# %%
sub = pd.DataFrame({"id": TE["id"]})
for p in POL:
    sub[p] = np.expm1(pred_log[p]).clip(min=0)
assert len(sub) == len(test) and set(sub.id) == set(test.id)
assert np.isfinite(sub[POL].to_numpy()).all()
sub.to_csv(OUT + "submission.csv", index=False)
print(sub.head())
print("saved", OUT + "submission.csv", f"total time {time.time() - T0:.0f}s")

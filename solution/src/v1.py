# %% [markdown]
# ## 1. Episode features (version 1)
# One row per (episode, station, forecast hour h = 0..23). The same function serves the
# training episodes (pseudo-targets) and the test episodes:
# * **calendar:** forecast hour, month, weekday;
# * **weather at the station for the target hour:** the target-day weather is provided as
#   a perfect-forecast proxy;
# * **network context for each of the six pollutants:** the network-mean log concentration
#   at the last context hour, over the last 6 and 24 hours, and over the first 24 context
#   hours, plus the value at the same hour yesterday and the spread across stations at the
#   last hour.
#
# Only context hours (< 72) of the network tensors are read, so the forecast never sees
# the target day's pollutants.

# %%
def rep(a):
    return np.repeat(a, HOR)

def episode_features(P, Wst, Wnet, month, wday):
    E = month.shape[0]
    f = {"h": np.tile(np.arange(HOR), E),
         "month": month[:, CTX:].ravel(), "weekday": wday[:, CTX:].ravel()}
    for w in WXCOLS:
        f[w] = Wst[w][:, CTX:].ravel()
    for p in POL:
        M = np.nanmean(P[p][:, :CTX, :], axis=2)          # network mean per context hour
        f[f"{p}_last"] = rep(M[:, -1])
        f[f"{p}_last6"] = rep(np.nanmean(M[:, -6:], axis=1))
        f[f"{p}_last24"] = rep(np.nanmean(M[:, -24:], axis=1))
        f[f"{p}_first24"] = rep(np.nanmean(M[:, :24], axis=1))
        f[f"{p}_yday"] = M[:, CTX - 24:CTX].ravel()        # same hour, previous day
        f[f"{p}_std_last"] = rep(np.nanstd(P[p][:, CTX - 1, :], axis=1))
    return f

# %% [markdown]
# ## 2. Training rows: every network station takes a turn as the "unmonitored" target
# For pseudo-target station s, the network features come from the **other 7** network
# stations. Including s's own readings would leak its label into its features, a
# relationship that does not exist at the real target stations.

# %%
def build_train():
    blocks = []
    for k, s in enumerate(NET):
        others = [j for j in range(len(NET)) if j != k]
        oth_all = [ALL.index(NET[j]) for j in others]
        P = {p: TR_P[p][:, :, others] for p in POL}
        Wst = {w: TR_W[w][:, :, ALL.index(s)] for w in WXCOLS}
        Wnet = {w: np.nanmean(TR_W[w][:, :, oth_all], axis=2) for w in WXCOLS}
        f = episode_features(P, Wst, Wnet, TR_MONTH, TR_WDAY)
        f["station"] = np.full(len(f["h"]), s)
        f["origin"] = np.repeat(TR_ORIGIN.to_numpy(), HOR)
        for p in POL:
            f[f"y_{p}"] = TR_P[p][:, CTX:, k].ravel()
        blocks.append(pd.DataFrame(f))
    return pd.concat(blocks, ignore_index=True)

def build_test():
    net_all = [ALL.index(s) for s in NET]
    blocks = []
    for t in TGT:
        Wst = {w: TE_W[w][:, :, ALL.index(t)] for w in WXCOLS}
        Wnet = {w: np.nanmean(TE_W[w][:, :, net_all], axis=2) for w in WXCOLS}
        f = episode_features(TE_P, Wst, Wnet, TE_MONTH, TE_WDAY)
        h = f["h"]
        f["key"] = (np.repeat(np.array(FIDS, dtype=object), HOR) + "_" + t + "_h"
                    + pd.Series(h).map("{:02d}".format).to_numpy())
        blocks.append(pd.DataFrame(f))
    return pd.concat(blocks, ignore_index=True)

TRF, TEF = build_train(), build_test()
COLS = [c for c in TRF.columns if c not in ("station", "origin", "key") and not c.startswith("y_")]
print("train rows:", TRF.shape, "| test rows:", TEF.shape, "| features:", len(COLS),
      f"[{time.time() - T0:.0f}s]")

# %% [markdown]
# ## 3. Validation that mirrors the test: new stations AND a later time period
# The test is both spatial (unmonitored stations) and temporal (the year after training).
# Each fold therefore holds out two network stations and validates them only on the last
# training year (origins from 2015-03-01), with models trained on the other six stations
# before that date. Random row splits would leak through autocorrelated neighbouring
# hours and overlapping episodes.

# %%
VAL_START = np.datetime64("2015-03-01")
PAIRS = [NET[i::4] for i in range(4)]
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=os.cpu_count(), seed=0)

def validate(TRF, cols, target_fn=None, params=PARAMS):
    """Returns out-of-fold predictions (log space) on validation rows and best iterations."""
    oof = {p: np.full(len(TRF), np.nan) for p in POL}
    iters = {p: [] for p in POL}
    late = (TRF.origin >= VAL_START).to_numpy()
    for p in POL:
        y = TRF[f"y_{p}"].to_numpy()
        off = target_fn(TRF, p) if target_fn else 0.0
        t = y - off
        ok = ~np.isnan(t)
        for pair in PAIRS:
            held = TRF.station.isin(pair).to_numpy()
            tr, va = ok & ~held & ~late, ok & held & late
            m = lgb.train(params, lgb.Dataset(TRF.loc[tr, cols], t[tr]), 3000,
                          valid_sets=[lgb.Dataset(TRF.loc[va, cols], t[va])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[p][va] = m.predict(TRF.loc[va, cols], num_iteration=m.best_iteration) + (
                off[va] if target_fn else 0.0)
            iters[p].append(m.best_iteration)
    return oof, iters

def report(oof, label):
    errs, per = [], {}
    for p in POL:
        y = TRF[f"y_{p}"].to_numpy()
        m = ~np.isnan(oof[p]) & ~np.isnan(y)
        e = np.clip(oof[p][m], 0, None) - y[m]
        per[p] = round(float(np.sqrt(np.mean(e ** 2))), 4)
        errs.append(e)
    pooled = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
    print(f"{label}: pooled RMSLE {pooled:.4f} | per pollutant {per}")
    return pooled

# Baseline on the same validation rows: persistence of the network mean at the last hour.
late_rows = (TRF.origin >= VAL_START).to_numpy()
persist = {p: np.where(late_rows, TRF[f"{p}_last"].fillna(TRF[f"{p}_last24"]).to_numpy(), np.nan) for p in POL}
report(persist, "VAL persistence baseline")

oof1, iters1 = validate(TRF, COLS)
val1 = report(oof1, "VAL LightGBM v1")
print(f"[{time.time() - T0:.0f}s]")

# %% [markdown]
# ## 4. Final models on all training episodes, then predict the test episodes
# The number of boosting rounds is the mean best iteration from validation, scaled up a
# little because the final model sees more data.

# %%
def fit_predict(TRF, TEF, cols, iters, target_fn=None, test_off_fn=None, params=PARAMS):
    preds = {}
    for p in POL:
        y = TRF[f"y_{p}"].to_numpy()
        off = target_fn(TRF, p) if target_fn else 0.0
        t = y - off
        ok = ~np.isnan(t)
        rounds = int(np.mean(iters[p]) * 1.1) + 1
        m = lgb.train(params, lgb.Dataset(TRF.loc[ok, cols], t[ok]), rounds)
        preds[p] = m.predict(TEF[cols]) + (test_off_fn(TEF, p) if test_off_fn else 0.0)
    return preds

pred_log = fit_predict(TRF, TEF, COLS, iters1)

# %% [markdown]
# ## 5. Write the submission
# Map the (episode, station, hour) rows onto the long-format test ids, back-transform with
# expm1 and clip at 0.

# %%
def write_submission(pred_log):
    parts = []
    for p in POL:
        parts.append(pd.DataFrame({"id": TEF["key"] + "_" + p,
                                   "value": np.expm1(np.clip(pred_log[p], 0, None))}))
    allp = pd.concat(parts, ignore_index=True).set_index("id")["value"]
    sub = pd.DataFrame({"id": test["id"], "value": allp.reindex(test["id"]).to_numpy()})
    assert sub["value"].notna().all() and np.isfinite(sub["value"]).all() and (sub["value"] >= 0).all()
    sub.to_csv(OUT + "submission.csv", index=False)
    print(sub.head())
    print("saved", OUT + "submission.csv", len(sub), "rows", f"total time {time.time() - T0:.0f}s")

write_submission(pred_log)

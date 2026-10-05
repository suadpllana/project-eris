# %% [markdown]
# ## 1. Episode features (same as v2)
# Everything from v1:
# * calendar features;
# * target-hour weather;
# * network context levels at several look-back windows.
#
# New in v2, each aimed at a specific failure of v1:
# * **target-day weather summaries:** daily rain total, mean wind speed and u/v
#   components, minimum and maximum temperature, mean dew point. A rainy or windy target
#   day cleans the air no matter how dirty the context was.
# * **weather change against the last context day** (u, v, dew point, pressure,
#   temperature). A sudden switch to dry northerly flow is the classic Beijing "clean-up"
#   front, which persistence cannot see coming.
# * **the site's weather relative to the network** at the target hour (temperature,
#   pressure, dew point, wind speed). Lower pressure means a higher, more suburban site,
#   and temperature or wind offsets reflect urban-heat-island and exposure differences.
#   This is how the model learns site-level offsets without any labels at the site.
# * **network dynamics:** the trend (last 6 h against last 24 h), the mean at the same hour
#   over the three context days (a local diurnal profile), the maximum over the last day,
#   plus Ox = NO2 + O3 and the PM2.5/PM10 ratio (cross-pollutant chemistry signals).

# %%
def rep(a):
    return np.repeat(a, HOR)

def episode_features(P, Wst, Wnet, month, wday):
    E = month.shape[0]
    h = np.tile(np.arange(HOR), E)
    f = {"h": h, "month": month[:, CTX:].ravel(), "weekday": wday[:, CTX:].ravel()}
    for w in WXCOLS:
        f[w] = Wst[w][:, CTX:].ravel()
    for w in ("TEMP", "PRES", "DEWP", "WSPM"):
        f[f"{w}_vs_net"] = (Wst[w][:, CTX:] - Wnet[w][:, CTX:]).ravel()
    day, last = slice(CTX, CTX + HOR), slice(CTX - 24, CTX)
    f["rain_day"] = rep(np.nansum(Wst["RAIN"][:, day], axis=1))
    f["rain_ctx24"] = rep(np.nansum(Wst["RAIN"][:, last], axis=1))
    for w in ("WSPM", "u", "v", "DEWP", "PRES", "TEMP"):
        day_mean = np.nanmean(Wst[w][:, day], axis=1)
        f[f"{w}_day"] = rep(day_mean)
        f[f"{w}_chg"] = rep(day_mean - np.nanmean(Wst[w][:, last], axis=1))
    f["TEMP_day_max"] = rep(np.nanmax(Wst["TEMP"][:, day], axis=1))
    f["TEMP_day_min"] = rep(np.nanmin(Wst["TEMP"][:, day], axis=1))
    for p in POL:
        M = np.nanmean(P[p][:, :CTX, :], axis=2)          # network mean per context hour
        f[f"{p}_last"] = rep(M[:, -1])
        f[f"{p}_last6"] = rep(np.nanmean(M[:, -6:], axis=1))
        f[f"{p}_last24"] = rep(np.nanmean(M[:, -24:], axis=1))
        f[f"{p}_first24"] = rep(np.nanmean(M[:, :24], axis=1))
        f[f"{p}_max24"] = rep(np.nanmax(M[:, -24:], axis=1))
        f[f"{p}_trend"] = f[f"{p}_last6"] - f[f"{p}_last24"]
        f[f"{p}_yday"] = M[:, CTX - 24:CTX].ravel()        # same hour, previous day
        prof = np.nanmean(np.stack([M[:, 0:24], M[:, 24:48], M[:, 48:72]]), axis=0)
        f[f"{p}_profile"] = prof.ravel()                    # 3-day mean at this hour
        f[f"{p}_std_last"] = rep(np.nanstd(P[p][:, CTX - 1, :], axis=1))
    f["Ox_last24"] = np.log1p(np.expm1(f["NO2_last24"]) + np.expm1(f["O3_last24"]))
    f["pm_ratio_last24"] = f["PM2.5_last24"] - f["PM10_last24"]
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

oof_d, iters_d = validate(TRF, COLS)
val_d = report(oof_d, "VAL direct model")

# %% [markdown]
# ## 3b. A second model family: forecast the *change* from the recent network level
# The residual model predicts `log1p(y) - network_last24` for the same pollutant, which is
# then added back. Its target is roughly stationary across the very different pollution
# regimes (winter haze against clean summer days), and across the temporal shift to the
# test year. The two families make different errors, so a blend helps. Blend weights are
# chosen per pollutant on the validation rows only.

# %%
def resid_offset(F, p):
    off = F[f"{p}_last24"].to_numpy()
    return np.where(np.isnan(off), np.nanmean(off), off)

oof_r, iters_r = validate(TRF, COLS, target_fn=resid_offset)
val_r = report(oof_r, "VAL residual model")

WEIGHTS, oof_b = {}, {}
for p in POL:
    y = TRF[f"y_{p}"].to_numpy()
    m = ~np.isnan(oof_d[p]) & ~np.isnan(y)
    grid = np.linspace(0, 1, 11)
    errs = [np.sqrt(np.mean((np.clip(w * oof_d[p][m] + (1 - w) * oof_r[p][m], 0, None) - y[m]) ** 2))
            for w in grid]
    WEIGHTS[p] = float(grid[int(np.argmin(errs))])
    oof_b[p] = WEIGHTS[p] * oof_d[p] + (1 - WEIGHTS[p]) * oof_r[p]
print("direct-model weight per pollutant:", WEIGHTS)
val_b = report(oof_b, "VAL blend")

# %% [markdown]
# ## 3c. Diagnostics: where does the error come from?
# Validation error by held-out station and by forecast hour. Suburban and rural sites
# (Changping, Dingling, Shunyi) carry the largest site offsets: Dingling's NO2 is far below
# the urban network. Error barely grows with the horizon, because the target-day weather
# carries most of the next-day signal and the context alone soon goes stale.

# %%
late = (TRF.origin >= VAL_START).to_numpy()
rows = []
for p in POL:
    y = TRF[f"y_{p}"].to_numpy()
    m = late & ~np.isnan(oof_b[p]) & ~np.isnan(y)
    rows.append(pd.DataFrame({"station": TRF.station[m], "h": TRF.h[m], "pollutant": p,
                              "se": (np.clip(oof_b[p][m], 0, None) - y[m]) ** 2}))
diag = pd.concat(rows)
print(diag.pivot_table(index="station", columns="pollutant", values="se", aggfunc="mean")
      .pipe(np.sqrt).round(3))
print(diag.groupby(pd.cut(diag.h, [-1, 5, 11, 17, 23]), observed=True).se.mean().pipe(np.sqrt).round(3))
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

pred_d = fit_predict(TRF, TEF, COLS, iters_d)
pred_r = fit_predict(TRF, TEF, COLS, iters_r, target_fn=resid_offset, test_off_fn=resid_offset)
pred_log = {p: WEIGHTS[p] * pred_d[p] + (1 - WEIGHTS[p]) * pred_r[p] for p in POL}

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

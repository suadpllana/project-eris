# %% [markdown]
# ## 1. Sanity check: `train.csv` labels are the history tensors
# `train.csv` lists every labelled (day, network station, hour, pollutant) in the same long
# format as `test.csv`. The tensors cut from `history.csv` hold the same values, so the check
# below confirms that the episode slicing lines up with the official labels.

# %%
chk = labels.sample(2000, random_state=0)
day = pd.to_datetime(chk.forecast_id.str[1:], format="%Y%m%d")
e_idx = pd.Index(TR_ORIGIN).get_indexer(day)
s_idx = chk.station.map({s: i for i, s in enumerate(NET)}).to_numpy()
got = np.array([TR_P[p][e, CTX + h, s] for p, e, h, s in zip(chk.pollutant, e_idx, chk.rel_hour, s_idx)])
assert (e_idx >= 0).all() and np.allclose(np.expm1(got), chk.value.to_numpy()), "label alignment failed"
print("train.csv labels match the history tensors on 2000 random rows")

# %% [markdown]
# ## 2. Batched feature construction (torch)
# Each sample is one (episode, target station). Its features are computed on the fly from
# the episode tensors and a **mask over network stations**:
# * The target station itself is always masked out (leave-self-out), because the real
#   target sites have no readings of their own.
# * Context sequence, 72 hours × 31 features: network mean and spread of each pollutant,
#   a mask for hours with no network reading, the site's weather, the site's weather
#   minus the network weather, and hour-of-day as sin/cos.
# * Target-day sequence, 24 hours × 13 features: site weather, weather offsets and hour
#   (the weather-forecast proxy).
# * Static: month and weekday as sin/cos, plus the site's mean pressure offset (an
#   elevation proxy).
# * Base level: the network-mean log concentration over the last 24 context hours. The
#   network predicts a correction on top of it.

# %%
NP = np.stack([TR_P[p] for p in POL], axis=-1).astype(np.float32)          # (E,96,8,6)
NW = np.stack([TR_W[w] for w in WXCOLS], axis=-1).astype(np.float32)       # (E,96,12,7)
TP = np.stack([TE_P[p] for p in POL], axis=-1).astype(np.float32)
TW = np.stack([TE_W[w] for w in WXCOLS], axis=-1).astype(np.float32)
NET_IN_ALL = [ALL.index(s) for s in NET]
HOURS = np.arange(CTX + HOR) % 24
HOUR_SC = np.stack([np.sin(2 * np.pi * HOURS / 24), np.cos(2 * np.pi * HOURS / 24)], -1).astype(np.float32)
OFFSET_W = [WXCOLS.index(w) for w in ("TEMP", "PRES", "DEWP", "WSPM")]

def masked_mean_std(x, m):
    """x: (B,T,K,C) with NaN, m: (B,K) station mask -> mean/std over masked, non-NaN stations."""
    valid = (~torch.isnan(x)) & m[:, None, :, None]
    x0 = torch.where(valid, x, torch.zeros_like(x))
    n = valid.sum(2).clamp(min=1)
    mean = x0.sum(2) / n
    var = ((x0 - mean[:, :, None, :]) ** 2 * valid).sum(2) / n
    has = valid.any(2)
    return torch.where(has, mean, torch.full_like(mean, float("nan"))), var.sqrt(), has

def make_batch(P, W, months, wdays, ep, site_all, net_mask):
    """P (E,96,8,6) log pollutants, W (E,96,12,7) weather; ep: episode ids; site_all: index of
    the target site in ALL; net_mask: (B,8) bool, which network stations may be used."""
    P = torch.from_numpy(P[ep, :CTX])                      # context only: no look-ahead
    Wall = torch.from_numpy(W[ep])                         # (B,96,12,7)
    m = torch.from_numpy(net_mask)
    mean, std, has = masked_mean_std(P, m)                 # (B,72,6)
    site = Wall[torch.arange(len(ep)), :, torch.from_numpy(site_all)]        # (B,96,7)
    wn_valid = (~torch.isnan(Wall[:, :, NET_IN_ALL])) & m[:, None, :, None]
    wnet = torch.where(wn_valid, Wall[:, :, NET_IN_ALL], torch.zeros(1)).sum(2) / wn_valid.sum(2).clamp(min=1)
    off = site[..., OFFSET_W] - wnet[..., OFFSET_W]
    hsc = torch.from_numpy(HOUR_SC)[None].expand(len(ep), -1, -1)
    xc = torch.cat([mean, std, (~has).float(), site[:, :CTX], off[:, :CTX], hsc[:, :CTX]], -1)
    xd = torch.cat([site[:, CTX:], off[:, CTX:], hsc[:, CTX:]], -1)
    mo = torch.from_numpy(months[ep, CTX].astype(np.float32))
    wd = torch.from_numpy(wdays[ep, CTX].astype(np.float32))
    xs = torch.stack([torch.sin(2 * np.pi * mo / 12), torch.cos(2 * np.pi * mo / 12),
                      torch.sin(2 * np.pi * wd / 7), torch.cos(2 * np.pi * wd / 7),
                      torch.nanmean(off[:, :CTX, 1], 1)], -1)
    base = torch.nanmean(mean[:, -24:], 1)                 # (B,6) recent network level
    return xc, xd, xs, base

# %% [markdown]
# ## 3. Samples, normalisation and validation split
# Training samples cover every daily origin × each of the 8 network stations. Validation
# mirrors the test: two held-out stations, evaluated only on the last training year
# (origins from 2015-03-01), with the model trained on the other six stations before
# that date.

# %%
E = len(origins)
S_EP = np.repeat(np.arange(E), len(NET))
S_K = np.tile(np.arange(len(NET)), E)
S_SITE = np.array([ALL.index(NET[k]) for k in S_K])
S_Y = torch.from_numpy(NP[S_EP, CTX:, S_K, :])                          # (N,24,6) log targets
VAL_START = np.datetime64("2015-03-01")
LATE = TR_ORIGIN.to_numpy()[S_EP] >= VAL_START
VAL_STATIONS = [NET.index("Changping"), NET.index("Dongsi")]   # one suburban + one urban site
is_val_st = np.isin(S_K, VAL_STATIONS)
TR_IDX = np.where(~is_val_st & ~LATE)[0]
VA_IDX = np.where(is_val_st & LATE)[0]

def self_mask(ks, drop=0.0, rng=None):
    m = np.ones((len(ks), len(NET)), dtype=bool)
    m[np.arange(len(ks)), ks] = False
    if drop > 0:   # network-dropout augmentation: hide random extra stations
        extra = rng.random(m.shape) < drop
        keep_one = m & ~extra
        ok = keep_one.sum(1) >= 2
        m[ok] = keep_one[ok]
    return m

def stats_from(idx):
    xc, xd, xs, _ = make_batch(NP, NW, TR_MONTH, TR_WDAY, S_EP[idx], S_SITE[idx], self_mask(S_K[idx]))
    f = lambda t: (torch.nan_to_num(torch.nanmean(t.reshape(-1, t.shape[-1]), 0)),
                   torch.nan_to_num(torch.from_numpy(np.nanstd(t.reshape(-1, t.shape[-1]).numpy(), 0)), nan=1.0).clamp(min=1e-3))
    return f(xc), f(xd), f(xs)

NORM = stats_from(np.random.default_rng(0).choice(len(S_EP), 3000, replace=False))

def normed(xc, xd, xs):
    (mc, sc), (md, sd), (ms, ss) = NORM
    z = lambda x, m, s: torch.nan_to_num((x - m) / s)
    return z(xc, mc, sc), z(xd, md, sd), z(xs, ms, ss)
print("samples:", len(S_EP), "| train/val (validation split):", len(TR_IDX), len(VA_IDX))

# %% [markdown]
# ## 4. Model, built and trained from scratch
# A GRU encoder reads the 72 context hours. A GRU decoder, initialised from the encoder
# state plus a static embedding, reads the 24 target-day weather steps and emits six log
# concentrations per hour as a correction to the recent network level. The loss is masked
# MSE in log1p space, which is the competition metric.

# %%
torch.set_num_threads(os.cpu_count())

class ForecastNet(nn.Module):
    def __init__(self, fc, fd, fs, h=96):
        super().__init__()
        self.enc = nn.GRU(fc, h, batch_first=True)
        self.dec = nn.GRU(fd, h, batch_first=True)
        self.static = nn.Linear(fs, h)
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, len(POL)))

    def forward(self, xc, xd, xs, base):
        _, hN = self.enc(xc)
        h0 = torch.tanh(hN + self.static(xs)[None])
        out, _ = self.dec(xd, h0)
        ctxv = hN[-1][:, None, :].expand(-1, out.shape[1], -1)
        return torch.nan_to_num(base)[:, None, :] + self.head(torch.cat([out, ctxv], -1))

def masked_mse(pred, y):
    m = ~torch.isnan(y)
    return ((pred - torch.nan_to_num(y)) ** 2 * m).sum() / m.sum().clamp(min=1)

def train_model(idx, epochs, seed=0, drop=0.0, val_idx=None, bs=128, lr=2e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ForecastNet(31, 13, 5)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(idx) / bs)))
    hist = []
    for ep_i in range(epochs):
        model.train()
        perm = rng.permutation(idx)
        for b in range(0, len(perm), bs):
            j = perm[b:b + bs]
            xc, xd, xs, base = make_batch(NP, NW, TR_MONTH, TR_WDAY, S_EP[j], S_SITE[j], self_mask(S_K[j], drop, rng))
            loss = masked_mse(model(*normed(xc, xd, xs), base), S_Y[j])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        if val_idx is not None:
            hist.append(evaluate(model, val_idx))
    return model, hist

@torch.no_grad()
def predict_train(model, idx, bs=512):
    model.eval()
    out = []
    for b in range(0, len(idx), bs):
        j = idx[b:b + bs]
        xc, xd, xs, base = make_batch(NP, NW, TR_MONTH, TR_WDAY, S_EP[j], S_SITE[j], self_mask(S_K[j]))
        out.append(model(*normed(xc, xd, xs), base))
    return torch.cat(out)

def evaluate(model, idx):
    pred = predict_train(model, idx).clamp(min=0)
    y = S_Y[idx]
    m = ~torch.isnan(y)
    return float(torch.sqrt(((pred - torch.nan_to_num(y)) ** 2 * m).sum() / m.sum()))

# Baseline on the same validation samples: hold the recent network level for all 24 hours.
_, _, _, b = make_batch(NP, NW, TR_MONTH, TR_WDAY, S_EP[VA_IDX], S_SITE[VA_IDX], self_mask(S_K[VA_IDX]))
y = S_Y[VA_IDX]; m = ~torch.isnan(y)
persist = torch.nan_to_num(b)[:, None, :].expand_as(y)
print("VAL baseline (network last-24h level): %.4f" % float(torch.sqrt(((persist - torch.nan_to_num(y)) ** 2 * m).sum() / m.sum())))

EPOCHS = 12
model, hist = train_model(TR_IDX, EPOCHS, seed=0, val_idx=VA_IDX)
print("VAL no augmentation, by epoch:", [round(h, 4) for h in hist], f"[{time.time() - T0:.0f}s]")

# %% [markdown]
# ## 4b. Network-dropout augmentation
# At test time every target site sees the full 8-station network, but outages mean the
# available set changes from hour to hour, and each target sits at a different distance
# from the stations. Randomly hiding extra network stations during training (each with
# probability 0.25, always keeping at least 2) teaches the model to aggregate a variable
# station set. It also stops the model from leaning on one particular station.

# %%
model_aug, hist_aug = train_model(TR_IDX, EPOCHS, seed=0, drop=0.25, val_idx=VA_IDX)
print("VAL with network dropout, by epoch:", [round(h, 4) for h in hist_aug], f"[{time.time() - T0:.0f}s]")
USE_DROP = 0.25 if min(hist_aug) < min(hist) else 0.0
BEST_EPOCHS = int(np.argmin(hist_aug if USE_DROP else hist)) + 1
print("use dropout:", USE_DROP, "| best epoch count:", BEST_EPOCHS)

# %% [markdown]
# ## 5. Final: an ensemble of 5 seeds trained on all samples
# Small recurrent models trained on about 9k samples vary noticeably from seed to seed.
# Averaging five independently initialised models (in log space) reduces that variance.

# %%
SEEDS = [0, 1, 2, 3, 4]
finals = [train_model(np.arange(len(S_EP)), BEST_EPOCHS, seed=sd, drop=USE_DROP)[0] for sd in SEEDS]
print(f"trained {len(finals)} final models [{time.time() - T0:.0f}s]")

@torch.no_grad()
def predict_test(model):
    model.eval()
    preds = {}
    for t in TGT:
        ep = np.arange(len(FIDS))
        site = np.full(len(ep), ALL.index(t))
        mask = np.ones((len(ep), len(NET)), dtype=bool)        # all 8 network stations
        xc, xd, xs, base = make_batch(TP, TW, TE_MONTH, TE_WDAY, ep, site, mask)
        preds[t] = model(*normed(xc, xd, xs), base).clamp(min=0).numpy()   # (73,24,6)
    return preds

def write_submission(preds):
    keys, vals = [], []
    for t, arr in preds.items():
        for i, f in enumerate(FIDS):
            for h in range(HOR):
                for c, p in enumerate(POL):
                    keys.append(f"{f}_{t}_h{h:02d}_{p}")
                    vals.append(arr[i, h, c])
    allp = pd.Series(np.expm1(np.array(vals)), index=keys)
    sub = pd.DataFrame({"id": test["id"], "value": allp.reindex(test["id"]).to_numpy()})
    assert sub["value"].notna().all() and np.isfinite(sub["value"]).all() and (sub["value"] >= 0).all()
    sub.to_csv(OUT + "submission.csv", index=False)
    print(sub.head())
    print("saved", OUT + "submission.csv", len(sub), "rows", f"total time {time.time() - T0:.0f}s")

runs = [predict_test(m) for m in finals]
write_submission({t: np.mean([r[t] for r in runs], axis=0) for t in TGT})

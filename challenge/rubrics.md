# Rubrics: Beijing Virtual Air-Quality Stations

Each criterion below can be entered as one rubric item: **Importance**, **Type**, **Criterion** and **Rationale**. The score thresholds refer to the reference figures in the problem description and the reference solution. Test MCRMSLE values are:
- network-median constant: 1.029
- month × hour climatology: 0.906
- hourly network log-mean: 0.4815
- weather-and-calendar-only gradient boosting: about 0.50
- reference solutions: v1 0.4578, v2 0.4359, v3 0.4367 (v3 has the best leave-one-station-out score, 0.4562)

---

### 1. REQUIRED · TRAINING
**Criterion:** Validates by holding out entire stations, for example leave-one-station-out or GroupKFold grouped on `station` over the 8 training sites. The final performance estimate is not a random row-level or shuffled K-fold score over `train.csv`.

**Rationale:** The test set is made of 4 sites the model never sees labelled. Hourly readings at one site are strongly autocorrelated and share the site's calibration and local sources. A row-level split therefore leaks each site's level into validation and gives a badly optimistic estimate.

### 2. REQUIRED · TRAINING
**Criterion:** Training examples built from network stations must not use the target station's own pollutant readings as inputs. Network aggregates (mean, median, nearest neighbour, etc.) for a training station are computed from the *other* network stations only.

**Rationale:** If a network station's own reading is inside its network-mean feature, the label leaks into the features. The model learns to copy it, and that relationship does not exist at the held-out sites, which have no readings. This is the most common silent failure on this task.

### 3. REQUIRED · FEATURE_ENGINEERING
**Criterion:** Uses the concurrent pollutant measurements of the 8 network stations (from `train.csv`, joined on timestamp) as inputs for the test sites. The model is not limited to each test row's own weather and calendar fields.

**Rationale:** Beijing pollution is dominated by regional episodes shared across sites. A model using only local weather and calendar scores about 0.50 MCRMSLE, worse than simply averaging the network each hour (0.4815). The task is spatial reconstruction, and the network readings for the same hour are allowed and essential.

### 4. REQUIRED · DATA_HANDLING
**Criterion:** Treats `NA` pollutant values as missing:
- rows with a missing label are excluded from training for that pollutant only;
- missing network readings are handled with NaN-aware aggregation or native missing-value support;
- missing labels or features are never filled with 0 before being used as targets.

**Rationale:** Between 1.5% and 5% of each pollutant column is missing, usually in multi-hour outages. Filling labels with 0 creates large log-space errors and teaches the model false clean-air hours. Dropping a whole row because one of six pollutants is missing throws away valid labels for the other five.

### 5. REQUIRED · MODELING
**Criterion:** Training targets the metric:
- the model is fitted on log1p-transformed concentrations, or with an equivalent loss;
- predictions are back-transformed with expm1;
- all six output columns are non-negative (clipped at 0 where needed).

**Rationale:** MCRMSLE is RMSE in log1p space, averaged over pollutants on very different scales (CO in the thousands, SO2 in single digits). Fitting raw concentrations lets the high-concentration winter haze hours dominate. Negative values are rejected by the grader.

### 6. REQUIRED · MODELING
**Criterion:** Achieves a test MCRMSLE of at most 0.47.

**Rationale:** 0.47 beats the hourly network-mean baseline (0.4815) and the weather-only model (about 0.50). A solution that cannot beat simple averaging of the network has not learned anything useful about the held-out sites.

### 7. RECOMMENDED · MODELING
**Criterion:** Achieves a test MCRMSLE of at most 0.445.

**Rationale:** Getting below 0.445 needs leak-free network features, temporal context and regularisation suited to only 8 training sites; the reference solutions reach about 0.44. A first-pass gradient-boosting model with default-style settings lands near 0.46. This threshold separates strong solutions from adequate ones.

### 8. REQUIRED · DATA_HANDLING
**Criterion:** Does not use any external air-quality measurements, including the original UCI or Kaggle copies of the Beijing Multi-Site dataset, which contain the held-out sites' answers.

**Rationale:** The raw source is public. Looking up the held-out sites' values is answer leakage, not modelling. The problem statement explicitly forbids it. Public site metadata such as coordinates is allowed.

### 9. RECOMMENDED · AGENT_BEHAVIOR
**Criterion:** Investigates the spatial structure between sites and tests any proximity idea under station-held-out validation, keeping it only if LOSO improves. Examples: cross-site correlation of pollutants, or noticing that several sites share an identical weather record (Changping = Dingling; Dongsi = Nongzhanguan = Tiantan; Aotizhongxin = Guanyuan). Proximity ideas include nearest-site readings and similarity-weighted means.

**Rationale:** Shared weather records show which sites are near each other, so "copy the nearest site" is tempting. But Changping and Dingling share weather and still differ strongly in NO2 and O3. In the reference experiments, nearest-neighbour features *raised* test error, from 0.441 to 0.455 (for Huairou, NO2 went from 0.64 to 0.78). Good agents check such ideas under LOSO instead of assuming they help.

### 10. RECOMMENDED · COMMUNICATION
**Criterion:** Reports validation error per held-out training station and per pollutant, not only the overall average. Notes that the suburban/rural network sites (Changping, Dingling, Shunyi) and the NO2/O3 columns are the hardest, and what that implies for the suburban test site (Huairou).

**Rationale:** The overall score hides a large spread. Urban sites near the network centre are reconstructed well (RMSLE about 0.3), while suburban sites with less traffic have NO2 and O3 offsets of 0.6 or more in log space. Per-station diagnostics are how an agent finds where the error comes from and avoids over-trusting one average number.

### 11. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Adds temporal context from the network and local series, using the hours before *and* after the target hour. Examples are centred rolling means, lags and leads of the network readings, or 24-hour wind and rain summaries.

**Rationale:** This is reconstruction, not forecasting, so the full network history is available. Smoothing reduces noise in single-hour readings and captures transport lag between sites, which the same-hour snapshot cannot.

### 12. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Encodes wind direction `wd` as a circular quantity, for example sin/cos of the bearing or u/v wind components combined with `WSPM`, not as an arbitrary integer label 0–15.

**Rationale:** NNW and N are neighbours but an ordinal code puts them 15 apart. Wind direction matters a lot in Beijing: southerly flow brings polluted air from the North China Plain and northerly flow clears it.

### 13. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Uses information across pollutants. When predicting each pollutant, the model also has the network readings of the other pollutants, or a combined quantity such as Ox = NO2 + O3.

**Rationale:** O3 and NO2 trade off through photochemical titration, and PM2.5, PM10 and CO move together in combustion and haze episodes. Using other pollutants' readings sharpens each target, especially O3, which is the hardest column (highest RMSLE).

### 14. RECOMMENDED · AGENT_BEHAVIOR
**Criterion:** Before building a learned model, scores at least one simple spatial baseline (such as the hourly network log-mean) under the same station-held-out validation. Later improvements are compared against that baseline.

**Rationale:** The simple baseline is surprisingly strong (0.4815). A weather-only gradient-boosting model (about 0.50) and some feature-rich variants are worse than it or barely better. Without the comparison an agent cannot tell whether added complexity helps.

### 15. REQUIRED · CODE_QUALITY
**Criterion:** Writes `submission.csv` with:
- exactly 140,256 rows, with one unique `id` per row of `test.csv`;
- the columns `id, PM2.5, PM10, SO2, NO2, CO, O3`, spelled exactly, including the dot in `PM2.5`;
- no missing or non-finite values.

**Rationale:** The grader rejects malformed submissions. Common slips include renaming `PM2.5` to `PM25` or `PM2_5`, dropping test hours with missing weather, and leaving NaN predictions where features were missing.

### 16. RECOMMENDED · MODELING
**Criterion:** Does not use station identity (name, one-hot, ordinal or target encoding of `station`) as a model input.

**Rationale:** All four test stations are categories the model has never seen, so station-identity features carry no usable information at test time. Any validation gain from them comes from memorising the training sites. Site-level behaviour must come from transferable inputs: local weather, network readings and their relationships.

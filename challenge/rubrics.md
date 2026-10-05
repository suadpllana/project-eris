# Rubrics: Cold-Start Air-Quality Forecasting at Unmonitored Beijing Sites

Each criterion below can be entered as one rubric item: **Importance**, **Type**, **Criterion** and **Rationale**.

Test RMSLE reference points:
- training median: 1.014
- month × hour climatology: 0.876
- network persistence: 0.831
- reference solutions: v1 0.6055, v2 0.5669, v3 0.5657

---

### 1. REQUIRED · TRAINING
**Criterion:** Builds training examples from the continuous `train.csv` with the **same structure as the test episodes**:
- forecast origin at 00:00;
- 72 hours of network context before the origin;
- 24 forecast hours after it, with only weather available for the target day.

The test is not treated as a generic per-row regression.

**Rationale:** Test rows are defined relative to a forecast origin, with only past network readings available. A model trained on same-hour network readings (a nowcast) has no matching inputs at test time. The episode structure must be reproduced in training for the features to mean the same thing.

### 2. REQUIRED · TRAINING
**Criterion:** When a network station is used as a pseudo-target, its own pollutant readings are excluded from the network features built for it. For example, network aggregates for station s come from the other 7 stations.

**Rationale:** The 4 target sites have no readings of their own. If a training station's own context sits inside its network features, the model learns to lean on the site's own history. That input does not exist at test time, and validation becomes optimistic.

### 3. REQUIRED · TRAINING
**Criterion:** Uses no future information:
- test features come only from the same episode's context hours (rel_hour < 0), the target-day weather and the training data;
- training features never read network pollutants from the target day;
- episodes are not chained or re-ordered to interpolate a target day from other episodes' context.

**Rationale:** This is a forecasting benchmark. Using the target day's network readings, or linking episodes by weather continuity, turns it into reconstruction and inflates the score. The problem statement explicitly forbids it.

### 4. REQUIRED · TRAINING
**Criterion:** Validates on **unseen stations and a later period at the same time**. For example, it holds out network stations and validates them only on the last training year (origins from 2015-03), training on earlier data from the other stations. It does not use a random row-level or random-episode split.

**Rationale:** The test shifts both location (cold-start sites) and time (the following year). Overlapping episodes and autocorrelated hours make random splits badly optimistic. In the reference solution, this split predicts test performance within about 0.01 RMSLE.

### 5. REQUIRED · DATA_HANDLING
**Criterion:** Handles missing values correctly:
- rows with a missing label are excluded from training for that pollutant only;
- missing context readings are handled with NaN-aware aggregation or native missing-value support;
- missing values are never filled with 0.

**Rationale:** Analyser outages leave 2–5% of readings missing, often in multi-hour runs. Zero-filling creates impossible clean-air values, which distort both labels and context features in log space.

### 6. REQUIRED · MODELING
**Criterion:** Trains on log1p-transformed concentrations, or with an equivalent loss, then back-transforms with expm1 and clips predictions at 0.

**Rationale:** The metric is RMSE in log1p space, pooled over six pollutants with very different scales. Training on raw concentrations lets winter haze episodes and CO's thousands of µg/m³ dominate. Negative values are rejected by the grader.

### 7. REQUIRED · MODELING
**Criterion:** Achieves a test RMSLE of at most 0.70.

**Rationale:** 0.70 clearly beats network persistence (0.831) and climatology (0.876). A solution above it has not learned to combine network context with weather.

### 8. RECOMMENDED · MODELING
**Criterion:** Achieves a test RMSLE of at most 0.59.

**Rationale:** Getting there needs target-day weather dynamics and site-offset features on top of a leak-free episode pipeline. The reference v1, with context and target-hour weather only, scores 0.6055. v2 and v3 score about 0.566.

### 9. REQUIRED · DATA_HANDLING
**Criterion:** Does not use external air-quality data, including the public UCI or Kaggle copies of this dataset, which contain the hidden answers.

**Rationale:** Looking up the target sites' measurements is answer leakage. It is explicitly forbidden.

### 10. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Uses the target-day weather as a forecast signal: rain totals, wind speed and direction, and the change from the last context day (for example a switch to dry northerly flow), not only the weather at the target hour.

**Rationale:** Beijing haze episodes end with cold fronts and rain. Persistence cannot see a clean-up coming. In the reference solutions, target-day weather summaries and changes give most of the gain from v1 (0.6055) to v2 (0.5669).

### 11. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Gives the model transferable site descriptors, such as the site's weather relative to the network mean (a pressure offset as an elevation proxy, temperature and wind offsets), instead of relying on the network average alone.

**Rationale:** Unmonitored sites differ systematically from the urban network. For example, suburban sites have less traffic NO2 and more O3. Without labels at those sites, these offsets must be inferred from inputs that exist everywhere, and weather offsets are one of them.

### 12. RECOMMENDED · MODELING
**Criterion:** Does not use station identity (name, one-hot, ordinal or target encoding) as a model feature.

**Rationale:** All four test sites are categories the model has never seen. Any validation gain from station identity comes from memorising training sites and does not transfer.

### 13. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Encodes wind direction `wd` as a circular quantity, such as sin/cos of the bearing or u/v components with `WSPM`, not as an integer label 0–15.

**Rationale:** NNW and N are neighbours, but an ordinal code puts them 15 apart. Wind direction drives pollutant transport in Beijing: southerly flow brings haze and northerly flow clears it.

### 14. RECOMMENDED · AGENT_BEHAVIOR
**Criterion:** Scores simple baselines (persistence of the network level, climatology) under the same validation split before adding model complexity, and compares each improvement against them.

**Rationale:** Persistence scores 0.83 on validation and test. Comparing against it shows whether the model actually forecasts or only reproduces the recent level.

### 15. RECOMMENDED · COMMUNICATION
**Criterion:** Reports validation error broken down by held-out station and by forecast hour or pollutant, and comments on where the error concentrates.

**Rationale:** Suburban and rural sites (Changping, Dingling, Shunyi) and the NO2/O3 columns carry most of the error. Dingling's NO2 is about 0.96 RMSLE against roughly 0.37 at urban sites. Breakdowns show which parts of the problem the model has not solved.

### 16. REQUIRED · CODE_QUALITY
**Criterion:** Writes `submission.csv` with columns `id,value`:
- one row for every id in `test.csv` (41,491 rows);
- no duplicate ids;
- no missing, non-finite or negative values.

**Rationale:** The grader rejects malformed submissions. Common slips are predicting only some pollutants, forgetting the long id format (`F001_Gucheng_h07_PM2.5`), or leaving NaN where context features were missing.

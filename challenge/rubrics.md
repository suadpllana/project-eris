# Rubrics: From-Scratch Neural Forecasting of Air Quality at Unmonitored Beijing Sites

Each criterion below can be entered as one rubric item: **Importance**, **Type**, **Criterion** and **Rationale**.

Test RMSLE reference points:
- training median: 1.014
- month × hour climatology: 0.876
- network persistence: 0.831
- reference solutions, all trained from scratch: v1 feed-forward 0.5784, v2 GRU encoder–decoder 0.5614, v3 GRU with station dropout and 5 seeds 0.5477

---

### 1. REQUIRED · TRAINING
**Criterion:** Builds training samples from `history.csv` / `train.csv` with the **same structure as the test episodes**:
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

**Rationale:** The test shifts both location (cold-start sites) and time (the following year). Overlapping episodes and autocorrelated hours make random splits badly optimistic. In the reference solutions, this split's score tracks the test score closely (within about 0.015 RMSLE).

### 5. REQUIRED · DATA_HANDLING
**Criterion:** Handles missing values correctly:
- rows with a missing label are excluded from training for that pollutant only;
- missing context readings are handled with NaN-aware aggregation or native missing-value support;
- missing values are never filled with 0.

**Rationale:** Analyser outages leave 2–5% of readings missing, often in multi-hour runs. Zero-filling creates impossible clean-air values, which distort both labels and context features in log space.

### 6. REQUIRED · MODELING
**Criterion:** Trains on log1p-transformed concentrations, or with an equivalent loss such as masked MSE in log1p space, then back-transforms with expm1 and clips predictions at 0.

**Rationale:** The metric is RMSE in log1p space, pooled over six pollutants with very different scales. Training on raw concentrations lets winter haze episodes and CO's thousands of µg/m³ dominate. Negative values are rejected by the grader.

### 7. REQUIRED · MODELING
**Criterion:** Achieves a test RMSLE of at most 0.70.

**Rationale:** 0.70 clearly beats network persistence (0.831) and climatology (0.876). A solution above it has not learned to combine network context with weather.

### 8. RECOMMENDED · MODELING
**Criterion:** Achieves a test RMSLE of at most 0.57.

**Rationale:** A simple feed-forward network on context summaries reaches about 0.578. Getting below 0.57 needs a sequence model over the full context and target-day weather (0.561), plus handling of the variable station set and some variance reduction (0.548).

### 9. REQUIRED · DATA_HANDLING
**Criterion:** Does not use external air-quality data, including the public UCI or Kaggle copies of this dataset, which contain the hidden answers.

**Rationale:** Looking up the target sites' measurements is answer leakage. It is explicitly forbidden.

### 10. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Feeds the whole target-day weather sequence to the model as a forecast signal (rain, wind speed and direction, temperature and dew point through the day), not only the weather at the target hour.

**Rationale:** Beijing haze episodes end with cold fronts and rain. Persistence cannot see a clean-up coming. The target-day weather trajectory is the main source of skill beyond the recent network level.

### 11. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Gives the model transferable site descriptors, such as the site's weather relative to the network mean (a pressure offset as an elevation proxy, temperature and wind offsets), instead of relying on the network average alone.

**Rationale:** Unmonitored sites differ systematically from the urban network. For example, suburban sites have less traffic NO2 and more O3. Without labels at those sites, these offsets must be inferred from inputs that exist everywhere, and weather offsets are one of them.

### 12. RECOMMENDED · MODELING
**Criterion:** Does not use station identity (name, one-hot, ordinal or target encoding) as a model feature.

**Rationale:** All four test sites are categories the model has never seen. Any validation gain from station identity comes from memorising training sites and does not transfer.

### 13. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Encodes wind direction `wd` as a circular quantity, such as sin/cos of the bearing or u/v components with `WSPM`, not as an integer label 0–15.

**Rationale:** NNW and N are neighbours, but an ordinal code puts them 15 apart. Wind direction drives pollutant transport in Beijing: southerly flow brings haze and northerly flow clears it.

### 14. REQUIRED · MODELING
**Criterion:** The main forecasting model is a neural network trained from random initialisation on the provided data. It uses no pre-trained weights or foundation models, whether time-series, language or vision.

**Rationale:** This is a From Scratch challenge. The skill being evaluated is designing and training a model for multi-horizon, multi-output forecasting over a variable set of sensors, not adapting a pre-trained model or submitting only a gradient-boosting model.

### 15. RECOMMENDED · MODELING
**Criterion:** Uses an architecture that models sequences or sets directly (recurrent, temporal-convolutional or attention-based encoders, or masked set pooling over stations) and shares structure across the 24 horizons and 6 pollutants, rather than 144 independent regressors.

**Rationale:** In the reference solutions, moving from a flattened feed-forward network (0.578) to a GRU encoder–decoder over the context and target-day weather (0.561) is the largest single gain from model design.

### 16. RECOMMENDED · TRAINING
**Criterion:** Makes the model robust to which network stations are available, for example by randomly dropping stations during training, masking missing readings, or pooling over stations in a way that does not depend on their order. The choice is validated against the same model without it.

**Rationale:** Outages change the available station set from hour to hour, and the target sites sit at different positions relative to the network. In the reference solution, station-dropout augmentation plus seed averaging improves test RMSLE from 0.561 to 0.548.

### 17. RECOMMENDED · AGENT_BEHAVIOR
**Criterion:** Scores simple baselines (persistence of the network level, climatology) under the same validation split before adding model complexity, and compares each improvement against them.

**Rationale:** Persistence scores 0.83 on validation and test. Comparing against it shows whether the model actually forecasts or only reproduces the recent level.

### 18. RECOMMENDED · COMMUNICATION
**Criterion:** Reports validation error broken down by held-out station and by forecast hour or pollutant, and comments on where the error concentrates.

**Rationale:** Suburban and rural sites (Changping, Dingling, Shunyi) and the NO2/O3 columns carry most of the error. For example, Dingling's NO2 error is more than double that of the urban sites, because it sits far from traffic. Breakdowns show which parts of the problem the model has not solved.

### 19. REQUIRED · CODE_QUALITY
**Criterion:** Writes `submission.csv` with columns `id,value`:
- one row for every id in `test.csv` (41,491 rows);
- no duplicate ids;
- no missing, non-finite or negative values.

**Rationale:** The grader rejects malformed submissions. Common slips are predicting only some pollutants, forgetting the long id format (`F001_Gucheng_h07_PM2.5`), or leaving NaN where context features were missing.

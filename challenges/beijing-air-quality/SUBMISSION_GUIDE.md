# Submission guide: revision 2 (reviewer feedback of 2026-10-08)

## Reviewer feedback and what changed

1. **"Duplicacy issue"**
   - The pollutant values were duplicated: every value appeared both in `history.csv` and in `train.csv`. Now pollutant labels exist **only** in `train.csv`, and the history file is replaced by a weather-only `weather.csv`. No value is repeated across files.
   - The metric now includes a worst-site term, which makes the task clearly different from similar problems on the platform.
2. **Removed the section** "What makes this a from-scratch modelling problem rather than a feature-table exercise".
3. **Removed every mention of the data source.** The words "UCI", "Beijing" and the weather agency no longer appear in the problem description, and the title no longer says "Beijing".
4. **Anonymised the data.** All 12 site names are replaced by ids `S01`–`S12`, assigned in salted-hash order. No place name appears in any prepared file (verified by search). Ids look like `F001_S03_h07_PM2.5`.

## What to change in the form (new version)

- **Challenge Title:** `From-Scratch Neural Forecasting of Air Quality at Unmonitored Sites`
- **Problem Description:** replace it with `challenge/problem_description.md`, without the first title line.
- **Grading Script:** replace it with `challenge/grade.py`. The metric is now site-robust RMSLE.
- **Grading Configuration:** keep Minimize, minimum 0, maximum 100.
- **Pipeline:** click **New Pipeline Version**, paste `challenge/prepare.py`, then click **Run Prepare**.
- **Tags:** feature-engineering. **Compute:** CPU. **Difficulty:** Medium.
- Click **Run checks**, then **Run agents** (this uses round 2 of 3).

**Expected prepared files:**
- `public/weather.csv`: 315,648 rows
- `public/train.csv`: 1,224,554 rows
- `public/test_context.csv`: 84,096 rows
- `public/test.csv`: 41,491 rows
- `public/sample_submission.csv`: 41,491 rows
- `private/answers.csv`: 41,491 rows

The sample submission scores **1.042**.

## Scores (site-robust RMSLE, lower is better)

**Baselines:**
- training median: 1.042
- climatology: 0.918
- persistence: 0.870

**Reference notebooks** (PyTorch, trained from scratch, CPU, each executed end-to-end and graded):
- **v1** feed-forward: **0.6256** (0.8 min)
- **v2** GRU encoder–decoder: **0.6142** (1.2 min)
- **v3** GRU + network dropout + 5 seeds: **0.5979** (3.9 min)

Earlier agent runs scored about 0.50 under the old pooled metric. Under the new metric, expect roughly 0.55–0.60, still below 0.7.

## Your responsibility
The Eris rules say "no LLM outputs in the submission". Review every file and make it your own before submitting.

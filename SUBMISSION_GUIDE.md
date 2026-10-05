# Submission guide: Beijing Virtual Air-Quality Stations

Follow the steps in order. Every file named here is in this repository.

---

## Step 1. Finish the draft dataset (no new upload needed)

Open your draft **Beijing Multi-Site Air Quality** dataset.

1. **Title:** keep `Beijing Multi-Site Air Quality`.
2. **Description:** replace it with the contents of `dataset/DATASET_DESCRIPTION.md`. The current draft has three problems:
   - it has an empty "Features" section (the column table is missing);
   - it ends with a stray `a`;
   - it says wind direction is not used and does not name the held-out stations. Both are now out of date.
3. **Data Files:** keep `PRSA2017_Data_20130301-20170228.zip` (7.6 MB). It matches `dataset/raw/` exactly (SHA-256 `d1b9261c…b0b8`).
4. **License & Source:** CC BY 4.0. Use source URL `https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data`; the direct zip URL you already entered also works.
5. Click **Run validation checks**, then **Mark as Ready**.

## Step 2. Create the challenge

**New Challenge**, then select the dataset above.

| Field | Value |
|---|---|
| Domain | Tabular (time series / spatial) |
| Difficulty | Medium |
| Challenge Title | `Beijing Virtual Air-Quality Stations` |
| Problem Description | paste `challenge/problem_description.md` |
| Prepare script | paste `challenge/prepare.py` |
| Grading script | paste `challenge/grade.py` |
| Config | `challenge/config.yaml` (minimize, min 0, max 100) |

Then click **Rebuild / Run checks**. The prepared dataset must contain:
- `public/train.csv`: 280,512 rows
- `public/test.csv`: 140,256 rows
- `public/sample_submission.csv`: 140,256 rows
- `private/answers.csv`: 140,256 rows

The sample submission should grade at **1.029**.

`prepare.py` accepts the platform's auto-extracted folder (`PRSA_Data_20130301-20170228/*.csv`) and also the raw zip if it is not extracted. Both give byte-identical output, which I checked.

## Step 3. Rubrics (16 criteria: 8 REQUIRED, 8 RECOMMENDED)

Enter each block of `challenge/rubrics.md` as one rubric item with its Importance, Type, Criterion and Rationale. All of them are specific to this task, for example:
- leave-self-out network features;
- LOSO validation;
- per-pollutant handling of NA;
- score thresholds of 0.47 and 0.445;
- never using the public copies of the dataset.

## Step 4. Solutions

Each notebook reads `./dataset/public/`, writes only `./working/submission.csv`, and uses only pandas, numpy and lightgbm. Each was executed end-to-end on 4 CPU cores and graded with `grade.py`:

| File | What changes | LOSO MCRMSLE | Test MCRMSLE | Runtime |
|---|---|---|---|---|
| `solution/solution_v1.ipynb` | Baseline: weather and calendar, circular wind, leave-self-out network aggregates, LightGBM per pollutant | 0.4596 | **0.4578** | 16 min |
| `solution/solution_v2.ipynb` | Adds temporal context (rolling, lag and lead of the network), weather vs network offsets, Ox; tuned regularisation | 0.4650 | **0.4359** | 6.5 min |
| `solution/solution_v3.ipynb` | Final: blends a direct model and a residual-from-network model, with weights picked on LOSO; per-station diagnostics | **0.4562** | **0.4367** | 11.5 min |

For comparison, the hourly network-mean baseline scores 0.4815 on test (LOSO 0.4804).

LOSO with only 8 stations is noisy. v2's LOSO is slightly worse than v1's even though v2 is much better on test, mainly because of the weather-vs-network offsets, which help Huairou. The notebooks say this openly rather than over-claiming.

Upload `solution_v3.ipynb` (renamed to `solution.ipynb`) as the challenge's reference solution. If you also post iterative solutions, submit v1, then v2, then v3. The notebooks are saved with their outputs, so reviewers can see the scores without re-running them.

v3 is the final because it has the best validation score (LOSO 0.4562 against 0.4650 for v2). Its test score ties v2's within noise (0.4367 against 0.4359). Choosing on validation rather than test is the correct practice, and reviewers can see this in the notebook.

To regenerate the notebooks from their sources, run `python solution/src/build_notebooks.py`. This produces notebooks without outputs.

## Step 5. Answers for likely reviewer questions

- **"This dataset is well known."** The usual tasks on it are single-station PM2.5 forecasting. This challenge is different: it reconstructs **all six pollutants at unseen sites**, holding out four complete stations, with a fixed split. The problem statement forbids the public copies of the dataset, which would leak the answers, and rubric 8 enforces this.
- **Why these four test stations?** They cover a mix of site types:
  - Gucheng, an urban site in the west;
  - Nongzhanguan, an urban site in the east whose weather twins are in train;
  - Wanliu, an urban site in the north-west;
  - Huairou, a suburban site in the north with no weather twin.

  Dingling, the rural background site, was moved into train on purpose. Its weather is identical to Changping's, so its large NO2 offset (about −0.9 in log space) could not be learned from any provided input. That would have added noise rather than a skill to test.
- **What is hard for an agent?**
  - Leaking a station's own reading into its network features, which inflates validation.
  - Validating with random row splits instead of station-held-out folds.
  - Falling below the surprisingly strong network-mean baseline.
  - Getting the suburban NO2 and O3 offsets right.
  - Handling per-pollutant NA labels.
- **Grader edge cases tested.** These are rejected with clear messages: wrong row count, duplicate ids, unknown or missing ids, missing columns, NaN, inf, negative values and non-numeric values. Rows in any order, extra columns and padded headers are accepted. NA ground truth is skipped for that pollutant only.

## Checklist (from the Eris docs)

- [x] Dataset description documents all 18 columns and their types
- [x] 420,768 rows
- [x] prepare.py is deterministic (checked with SHA-256 over two runs; `bash tools_check.sh`)
- [x] License CC BY 4.0 allows commercial use; source URL documented
- [x] Problem description is complete; metric formula and code given; submission format exact
- [x] grade.py scores valid submissions and rejects invalid ones with clear errors
- [x] 16 task-specific rubrics, all REQUIRED or RECOMMENDED
- [x] Solutions run end-to-end well under 1 hour and beat every baseline

## Important: your responsibility

The Eris rules say *"You may not use any LLM outputs as part of your submission."* Everything in this repository was drafted by an AI assistant. Before submitting, read every file, check that you agree with it, and edit it into your own work. If you are unsure how Shipd applies that rule to AI-assisted authoring, ask them first.

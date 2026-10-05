# Submission guide: Cold-Start Air-Quality Forecasting at Unmonitored Beijing Sites

This is version 2 of the challenge, rebuilt after the first validation run failed. All paste-in texts use plain bullet lists, because Shipd's text boxes don't accept pasted tables.

## What changed, and which failed check each change addresses

- **Prepared Data Integrity: blank answers.** Test rows are now created only where a real measurement exists, so `answers.csv` has no blank cells.
- **Target Recoverability: no supported target.** There is now a single target column, `value`, in a long format with one row per (episode, site, hour, pollutant).
- **Evaluator Contract: known answer, baselines, public/private.** The grader now scores the true answers (0.0) and ranks them above the baselines. It also accepts scoring on subsets of rows, which is needed for public/private leaderboards.
- **Domain Routing: classified as "Regression".** The task is now time-series forecasting: next-day hourly forecasts from 72-hour context windows.
- **Novelty: incremental.** The reviewer suggested a domain shift. The task now has three shifts together:
  - cold-start sites with no pollutant history at all;
  - an unseen later year (temporal shift);
  - strict no-look-ahead forecasting.

## Step 1. Dataset (no new upload)

Keep the same zip. Optionally, replace the dataset description with `dataset/DATASET_DESCRIPTION.md`, if Shipd still lets you edit it; only its last "Notes" bullet changed. If editing is locked, that's fine: those notes are informational.

## Step 2. Edit the challenge (same draft, new version)

- **Difficulty:** Medium
- **Compute Tier:** CPU
- **Challenge Title:** `Cold-Start Air-Quality Forecasting at Unmonitored Beijing Sites`
- **Problem Description:** replace it with all of `challenge/problem_description.md`
- **Tags:** feature-engineering
- **Grading Configuration:** **Minimize**, minimum 0, maximum 100
- **Grading Script:** Custom; replace it with all of `challenge/grade.py`
- **Pipeline → prepare.py:** replace it with all of `challenge/prepare.py`, then click **Re-run Prepare**

**Expected prepared files:**
- `public/train.csv`: 315,648 rows
- `public/test_context.csv`: 84,096 rows
- `public/test.csv`: 41,491 rows
- `public/sample_submission.csv`: 41,491 rows
- `private/answers.csv`: 41,491 rows, with no blank values

The sample submission grades at **1.014**.

## Step 3. Run checks

Click **Run checks** and send me any failure messages and the novelty score.

## Step 4. Solutions

Each notebook was executed end-to-end in a clean folder (`./dataset/public` in, `./working/submission.csv` out) and graded with `grade.py`:

- **solution_v1.ipynb (baseline):** validation 0.616, test **0.6055**, runtime 1.2 min
- **solution_v2.ipynb (target-day weather, site offsets, network dynamics):** validation 0.580, test **0.5669**, runtime 2.2 min
- **solution_v3.ipynb (final; direct + residual blend chosen on validation):** validation 0.574, test **0.5657**, runtime 4.7 min

Baselines on test:
- network persistence: 0.831
- climatology: 0.876
- training median: 1.014

Upload `solution_v3.ipynb` (renamed to `solution.ipynb`) when the platform asks for a reference solution. If you post iterative solutions, submit v1, then v2, then v3.

## Step 5. Answers for likely reviewer questions

- **Why can't solvers just interpolate from the network?** The target day's network readings are hidden; only the 72 hours before midnight are given. Test episodes are separated by an unused day, carry no calendar date, and have ids in hashed rather than chronological order. That makes it impractical to chain episodes to recover a target day.
- **Why is the target-day weather given?** It stands in for a numerical weather forecast, which real air-quality forecasters always have. Without it the task reduces to persistence.
- **What is hard for an agent?**
  - Rebuilding training episodes that match the test structure from continuous data.
  - Leave-self-out network features, since a target site has no history of its own.
  - Validation that holds out both sites and a later time.
  - Using weather dynamics (clean-up fronts, rain) and site offsets.
  - Long-format submission bookkeeping.
- **Grader edge cases tested.** These are rejected with clear messages: missing ids, duplicates, missing columns, NaN, inf, negative and non-numeric values. Rows in any order and extra ids are accepted.

## Important: your responsibility

The Eris rules say *"You may not use any LLM outputs as part of your submission."* Everything here was drafted by an AI assistant. Review every file and make it your own before submitting. If you're unsure how Shipd applies that rule, ask them first.

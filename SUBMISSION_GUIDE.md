# Submission guide: From-Scratch Neural Forecasting of Air Quality at Unmonitored Beijing Sites

Version 3, built for the **From Scratch** domain. The accepted domains are NLP, Computer Vision, Object Detection, Recommendation, Sequence to Sequence, Prompt Engineering, RAG, Fine-Tuning, From Scratch and LLM Evaluation. Regression and Forecasting are closed.

## How each failed check is addressed

- **Domain Routing ("Forecasting" closed):** the challenge now has a "Required From-Scratch Setting" section, modelled on how your accepted challenge declared its Fine-Tuning setting. The intended solution is a neural network trained from random initialisation with no pre-trained weights, and all reference notebooks are PyTorch models trained from scratch.
- **Prepared Data Integrity ("Train and test feature columns must match"):** `train.csv` now has exactly the columns of `test.csv` plus `value`, as labelled next-day forecast rows. The continuous hourly data moved to `history.csv`.
- **Checks that already passed** stay passing: evaluator contract, target recoverability (0/100), specification and originality. The answers file and the grader are unchanged.

## What to change in the form

1. **Challenge Title:** `From-Scratch Neural Forecasting of Air Quality at Unmonitored Beijing Sites`
2. **Problem Description:** replace it with `challenge/problem_description.md`, without the first line, which is the title.
3. **Grading Script:** keep it. `grade.py` is unchanged.
4. **Grading Configuration:** keep Minimize, minimum 0, maximum 100.
5. **Pipeline:** click **New Pipeline Version**, paste the new `challenge/prepare.py`, then click **Run Prepare**.
6. Click **Run checks**.

**Expected prepared files:**
- `public/history.csv`: 315,648 rows
- `public/train.csv`: 1,221,145 rows
- `public/test_context.csv`: 84,096 rows
- `public/test.csv`: 41,491 rows
- `public/sample_submission.csv`: 41,491 rows
- `private/answers.csv`: 41,491 rows

The sample submission grades at **1.014**.

## Solutions

All three are PyTorch models trained from scratch on CPU. Each was executed end-to-end in a clean folder and graded with `grade.py`. Training uses fixed seeds, so the scores reproduce exactly.

- **solution_v1.ipynb (feed-forward network on context summaries + target-day weather):** test **0.5784**, runtime 0.4 min
- **solution_v2.ipynb (GRU encoder–decoder):** test **0.5614**, runtime 1.3 min
- **solution_v3.ipynb (final: GRU + network-dropout augmentation + 5-seed ensemble):** test **0.5477**, runtime 4.9 min

Baselines on test:
- network persistence: 0.831
- climatology: 0.876
- training median: 1.014

For comparison, a tuned LightGBM pipeline reached 0.5657, so the from-scratch neural approach is genuinely the stronger one here.

## Important: your responsibility

The Eris rules say *"You may not use any LLM outputs as part of your submission."* Everything here was drafted by an AI assistant. Review every file and make it your own before submitting. If you're unsure how Shipd applies that rule, ask them first.

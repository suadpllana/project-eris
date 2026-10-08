# Project Eris (Shipd) challenge workspace

This repo builds ML benchmark challenges for Shipd's **Project Eris** quest. Each challenge is a dataset, a `prepare.py`, a `grade.py`, a problem description, rubrics, and 3+ solution notebooks. The creator is paid **$400–500 per accepted challenge, scaled by novelty**.

Read this file before designing or editing any challenge. Every rule below comes from a real failed check or review.

## Layout

```
challenges/<slug>/
  dataset/raw/                 raw upload (exactly what goes to Shipd; zips are auto-extracted there)
  dataset/DATASET_DESCRIPTION.md
  challenge/problem_description.md   challenge/prepare.py   challenge/grade.py
  challenge/config.yaml        challenge/rubrics.md
  solution/solution_v1..v3.ipynb  (executed, with outputs)   solution/src/ (sources + build_notebooks.py)
  SUBMISSION_GUIDE.md          README.md
tools/eris_check.py            local replica of the platform's automated checks
```

**Before handing anything to the user, run `python tools/eris_check.py challenges/<slug>`. It must report 0 FAIL.** The platform-only checks (domain, novelty, originality, agent difficulty) are covered by the design rules below.

## Hard platform rules

### Domains
Only these domains accept submissions:
- NLP
- Computer Vision
- Object Detection
- Recommendation
- Sequence to Sequence
- Prompt Engineering
- RAG
- Fine-Tuning
- From Scratch
- LLM Evaluation

Rules for getting routed correctly:
- **Regression and Forecasting are closed.** Tabular classification is very likely closed too.
- The domain is classified automatically from the problem description. Plain tabular or time-series prediction gets routed to a closed domain.
- To steer routing, add an explicit section such as `## Required <Domain> Setting`. It should state the intended solution setting, add "the requirement concerns the intended solution setting, not the formal target", and be backed by matching reference solutions. This is how the accepted CVE challenge (Fine-Tuning) and the Beijing challenge (From Scratch) were routed.
- The form shows "Design challenges that require using pre-trained models for significantly better payouts." Prefer domains where pre-trained or deep models matter.

### File contract (Prepared Data Integrity, Target Recoverability, Evaluator Contract)
- `public/train.csv` must have **exactly the columns of `public/test.csv` plus the target column(s)**. Put auxiliary data (history, context windows, images, metadata) in separate files.
- `private/answers.csv` must be the id column plus **one target column** with **no missing values**. For multi-output tasks, use long format (one row per id × output, for example `F001_Gucheng_h07_PM2.5,value`). Only create test rows where ground truth exists.
- `sample_submission.csv` has the same columns and ids as `answers.csv`.
- The grader must:
  - score the true answers at the best value;
  - rank them above simple baselines;
  - be deterministic and independent of row order;
  - **score subsets of `answers`** (public/private leaderboard split). Score on the answer ids and ignore extra submission ids.
- The grader must raise on a missing id, a duplicate id, a missing column, and NaN, inf or negative values.
- Columns that are 50% or more missing trigger a warning. If that's by design, say so in the description.
- `prepare.py` must be deterministic (no unseeded randomness, sorted output). Use only Kaggle-image libraries. It must accept the raw folder as Shipd provides it, with zips already extracted.

### Novelty (gate at 5/10 to submit; aim for 7 or more)
The checker searches the literature and public datasets, names the closest prior work, and suggests changes. History:
- **Beijing reconstruction:** "incremental". The UCI Beijing data is heavily studied, and "virtual station" papers exist.
- **Beijing cold-start forecasting:** suggestions only. It asked for randomised withheld groups, probabilistic forecasts, or a withheld season or extreme event.
- **Beijing From Scratch:** **5/10**. It credited "a combination not found in any single prior benchmark", then suggested a worst-site or calibration-aware metric, or a wider geographic scope.

How to reach 7 or more:
1. **Dataset:** use recent or little-known data. Avoid famous UCI/Kaggle datasets, because the checker finds papers built on them, which caps novelty.
2. **Non-standard output structure:** sets, permutations or matching, decompositions, structured multi-part outputs, or per-group constraints. The accepted CVE challenge used six-way permutation matching with batch consistency.
3. **A custom metric that encodes the real objective:**
   - combine accuracy with **calibration** (Brier, pinball or quantile loss, CRPS);
   - add a **worst-group term**, such as mean plus minimum over hidden domains or sites, like the shadow challenge;
   - give the formula, explain it, and implement it exactly.
4. **Engineered shift:** unseen domains or groups, disjoint source groups (vendor-disjoint, site-disjoint), held-out styles, seasons or extreme events.
5. **A "Why This Task Is Distinct" section** that names the closest benchmarks or methods and explains what they can't do here.
6. Novelty is judged on the design, not the wording. Every claim in the distinctiveness section must be true in the data and the metric.

### Difficulty (agent runs)
- Three AI agents attempt the challenge, then a difficulty check runs. You get **3 evaluation rounds per challenge**. A round only counts if at least 2 agents return valid scores.
- The user's guidance: **the agent score should stay below 0.7.** Beijing passed with agents at 0.50–0.53 RMSLE (lower is better). Design so the task is clearly solvable but not maxed out.
- Agents are strong. Beijing's agents (0.50) beat the reference notebook (0.548). Make the final reference solution at least match a strong agent, because reviewers compare them. Solvers must also beat the AI baseline to be paid.

### Shipd form gotchas
- Text boxes don't accept pasted markdown tables. Use bullet lists in every paste-in text (descriptions and problem statement).
- **Challenge Title** is a separate field. Don't leave the `# Title` line at the top of the problem description.
- **Grading Configuration defaults to Maximize.** For error metrics, set it to Minimize and enter min and max.
- Once Run Prepare has been executed, the Pipeline is locked. Click **New Pipeline Version** before pasting a new `prepare.py`.
- You can upload only one new dataset per 24 hours, so reuse a Ready dataset when possible.
- Running checks and editing drafts is free. Agent rounds are limited to 3.
- Rubrics are currently disabled platform-wide. Keep `rubrics.md` anyway (5+ items, mostly REQUIRED or RECOMMENDED, task-specific, approach-neutral).
- The Eris rules say "no LLM outputs in the submission". Always remind the user to review and own every file.

## Human review lessons (Revision Requested, 2026-10-08)
- **Anonymise identifying values.** Replace place names, site or station names, vendor names and similar values with opaque ids (salted-hash order, such as `S01`–`S12`). No real names in any public file. `prepare.py` should do this, and a search over the public CSVs should find zero names.
- **Don't name the data source in the problem description.** No "UCI", "Kaggle", city names, agency names or dataset titles. Say "do not use external data or try to identify the sources" instead. Source attribution belongs only in the dataset description.
- **No duplicated data ("duplicacy").** A value must not appear in two public files. For example, don't ship a history file with pollutants *and* a `train.csv` with the same labels. Keep labels only in `train.csv` and put features (weather, metadata) in auxiliary files. Also avoid problems that look like duplicates of existing platform problems; differentiate with the metric and output design.
- **No meta or salesy sections** such as "What makes this a … problem rather than …". Keep the description factual: overview, required setting, evaluation, dataset, submission.

## Workflow for a new challenge
1. Choose an **open domain first**, then a dataset that fits it (recent, licence allows commercial use, source URL documented).
2. Design for novelty of 7 or more using the recipe above. Write the "Why This Task Is Distinct" and "Required <Domain> Setting" sections.
3. Write `prepare.py` to the file contract. Write `grade.py` with subset support and strict validation.
4. Run `tools/eris_check.py` until it reports 0 FAIL.
5. Compute baselines (constant, simple heuristic) and quote them in the description.
6. Build 3 notebooks (baseline → improved → final), executed end-to-end in `./dataset/public` → `./working/submission.csv` and graded. The final one should be competitive with strong agents.
7. Write `SUBMISSION_GUIDE.md` with field-by-field paste-in instructions, expected row counts and the sample-submission score.

## Shell note
Don't put `rm -rf` inside `bash -c` strings or scripts launched by them. The harness blocks it. Use fresh timestamped directories instead.

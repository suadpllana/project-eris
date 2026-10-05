# Project Eris challenge: Beijing Virtual Air-Quality Stations

A complete Shipd Eris submission package built on the draft dataset **"Beijing Multi-Site Air Quality"** (UCI id 501, CC BY 4.0).

**Task.** 8 Beijing monitoring sites are fully labelled. 4 held-out sites (Gucheng, Huairou, Nongzhanguan, Wanliu) come with weather only. The solver reconstructs all 6 pollutants (PM2.5, PM10, SO2, NO2, CO, O3) at the held-out sites for every hour from 2013-03 to 2017-02, about 140k rows × 6 targets.

**Metric.** Mean column-wise RMSLE, where lower is better.

## Repository layout

| Path | What it is | Where it goes on Shipd |
|---|---|---|
| `dataset/raw/PRSA2017_Data_20130301-20170228.zip` | Raw upload (already attached to your draft dataset) | Dataset → Data Files |
| `dataset/DATASET_DESCRIPTION.md` | Corrected dataset description (adds the missing Features column list) | Dataset → Description |
| `challenge/problem_description.md` | Problem statement the agent sees | Challenge → Problem Description |
| `challenge/prepare.py` | Deterministic public/private split | Challenge → Prepare script |
| `challenge/grade.py` | MCRMSLE grader with input validation | Challenge → Grading script |
| `challenge/config.yaml` | Name / difficulty / domain / score range | Challenge → Config / Domain & Difficulty |
| `challenge/rubrics.md` | 16 rubric criteria (8 REQUIRED, 8 RECOMMENDED) | Challenge → Rubrics |
| `solution/solution_v1.ipynb` … `solution_v3.ipynb` | Reference notebooks: baseline → improved → final | Solutions (upload each as `solution.ipynb`) |
| `solution/src/` | Plain-Python sources the notebooks are built from | not uploaded |
| `tools_check.sh` | Local determinism + grading check | not uploaded |

See `SUBMISSION_GUIDE.md` for the click-by-click steps and the verified scores.

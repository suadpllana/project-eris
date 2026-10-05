# Project Eris challenge: From-Scratch Neural Forecasting of Air Quality at Unmonitored Beijing Sites

A complete Shipd Eris submission package built on the draft dataset **"Beijing Multi-Site Air Quality"** (UCI id 501, CC BY 4.0).

**Task (domain: From Scratch).** Train a neural network from random initialisation to produce next-day, hour-by-hour forecasts of 6 pollutants at 4 Beijing sites that never had a pollutant monitor. The forecasts use 72 hours of readings from 8 monitored network sites, plus weather. Training covers 2013-03 to 2016-02. The test is 73 independent forecast episodes from the following year.

**Metric.** RMSLE over 41,491 (episode, site, hour, pollutant) rows; lower is better.

## Repository layout

- `dataset/raw/PRSA2017_Data_20130301-20170228.zip`: raw upload, already attached to your dataset (Dataset → Data Files)
- `dataset/DATASET_DESCRIPTION.md`: dataset description (Dataset → Description)
- `challenge/problem_description.md`: problem statement the agent sees (Challenge → Problem Description)
- `challenge/prepare.py`: deterministic public/private build (Challenge → Pipeline → prepare.py)
- `challenge/grade.py`: RMSLE grader with input validation (Challenge → Grading Script, Custom)
- `challenge/config.yaml`: values for the Grading Configuration and Difficulty fields
- `challenge/rubrics.md`: 19 rubric criteria (10 REQUIRED, 9 RECOMMENDED), in case rubrics are re-enabled
- `solution/solution_v1.ipynb` … `solution_v3.ipynb`: executed PyTorch reference notebooks, baseline → final (upload as `solution.ipynb`)
- `solution/src/`: plain-Python sources the notebooks are built from (not uploaded)
- `tools_check.sh`: local determinism and grading check (not uploaded)

See `SUBMISSION_GUIDE.md` for the click-by-click steps.

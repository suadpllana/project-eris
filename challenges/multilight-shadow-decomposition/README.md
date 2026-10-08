# Project Eris challenge: Multi-Light Fractional Shadow Decomposition

Computer Vision, Hard, A10G. Status: **Revision Requested** (reviewer brianaltan, 2026-10-03). Agent runs before the revision were 0.6202 / 0.6050 / 0.6115. Novelty is 6/10.

**Task.** For each 192×160 RGB scene, predict three light-specific soft shadow-attribution fields plus residual mass. Lights are identified by four-dot codes in the header. Training uses studio and blueprint renderings. The test uses unseen thermal and low-contrast-scan renderings.

**Metric.** Per light channel, 0.7 × soft Dice + 0.3 × normalised L1 skill, averaged per image. Then 0.5 × mean + 0.5 × minimum over the hidden domains. Higher is better, on a scale of 0 to 1.

## Layout

- `challenge/problem_description.md`: paste-in problem statement, with no title line
- `challenge/prepare.py`: deterministic public/private build. Paste it into a new pipeline version.
- `challenge/grade.py`: grader (Custom)
- `challenge/config.yaml`: Grading Configuration values (Maximize, 0 to 1)
- `dataset/DATASET_DESCRIPTION.md`: dataset description (unchanged; the dataset is already Ready)
- `tools/measure_payload_sizes.py`: run it locally on your downloaded `public.zip` to measure real target sizes in the old and new encodings
- `SUBMISSION_GUIDE.md`: what changed for the revision, the form steps and the reviewer reply

The raw data (`dataset/raw/`) is not in git. It's the Ready Shipd dataset, about 157 MB.

To run the local checks, put the raw files in `dataset/raw/`, then run `python tools/eris_check.py challenges/multilight-shadow-decomposition` from the repo root.

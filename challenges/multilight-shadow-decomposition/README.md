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

## Novelty: 6/10, and what would raise it

The platform scored the pre-revision design **6/10**, so it passes the 5/10 gate. The checker said:
- **Closest prior work:** LuxRemix (CVPR 2026), which does single-image per-light OLAT decomposition with synthetic multi-light supervision. It also named POLAR and SP+M-Net.
- **What it credited:**
  - sum-to-one fractional attribution channels with a residual;
  - light identity from in-image coded markers;
  - soft-Dice + normalised-L1 scoring with mean + minimum over domains.
- **What it suggested:** a real-world capture set (real OLAT or multi-modal scans) for evaluation.

The size revision (96×80 grid, PNG payload) changes the encoding only, not the design the checker credits. **Expect about 6 again, not 7**, if novelty is re-scored.

Options to reach 7 or more, cheapest first:
1. **Name the neighbours.** Rewrite "Distinction from Related Shadow Benchmarks" to name LuxRemix, POLAR and SP+M-Net, and state the differences the checker itself credited.
   - This costs no grader or data change and no agent round.
   - The effect is small, because novelty is judged on design, not wording.
   - Every claim must stay true.
2. **Add an overlap or worst-light metric term.** For example, score separately on cells where two or more lights contribute, or add a minimum over the three light channels.
   - This is a real design change, but it modifies `grade.py`.
   - It needs another agent round (3 per challenge) and may move agent scores, which must stay under 0.7.
3. **Real captured evaluation data,** as the checker suggested. This needs a new dataset upload, so it amounts to a new challenge, not a revision.

Recommendation for this revision: do option 1 at most. Keep the design stable while addressing the reviewer, because 6/10 already clears the gate.

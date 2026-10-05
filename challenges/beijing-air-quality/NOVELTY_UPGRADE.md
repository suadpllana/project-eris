# Novelty upgrade plan (apply only if review requests changes)

The live version scored **5/10** on novelty. The checker's suggestions were:
- a worst-site or calibration-aware metric;
- a wider geographic scope.

Don't edit the challenge while it is Pending Review. If a reviewer sends it back, apply the changes below in a new version. They keep the dataset, the domain (From Scratch) and the single-target file contract.

## 1. Probabilistic output (calibration-aware)
- Solvers forecast three quantiles (q10, q50, q90) for every hour.
- `test.csv` and `answers.csv` stay single-target: each id gets a `_q10`, `_q50` or `_q90` suffix, and `value` holds the quantile forecast. Answers repeat the measured value for each quantile row.
- The metric becomes pinball (quantile) loss in log1p space, normalised by the persistence baseline's loss.

## 2. Worst-site term
```
SITE_SCORE(s) = mean pinball loss over site s (log1p space)
SCORE = 0.5 * mean over the 4 sites + 0.5 * max over the 4 sites
```
Lower is better. This rewards models that transfer to *every* unmonitored site, not only the urban ones. It mirrors the mean-plus-worst-domain structure of the shadow challenge.

## 3. A "Why This Task Is Distinct" section
It names the prior work the checker found and explains the difference:
- **DRoL (Wang, Bühlmann & Guo):** distributionally robust transfer to unlabelled stations; point prediction, no forecasting horizon.
- **AirRadar:** deep inference at unmonitored locations; reconstruction, not next-day forecasting, and no calibrated intervals.
- **This challenge:** cold-start sites, a strict no-look-ahead 24-hour horizon, quantile calibration and a worst-site objective under a temporal shift, all in one benchmark.

## Work required
- `prepare.py`: id suffixes and three answer rows per measurement.
- `grade.py`: pinball loss plus the worst-site aggregation, with subset support kept.
- The notebooks: a quantile head on the GRU (three outputs per pollutant, pinball loss).
- Then re-run `tools/eris_check.py` and all three notebooks.

This costs one agent round. Expect novelty around 6–7 rather than a guaranteed 7. The UCI Beijing data itself limits the score, so a fresh, little-known dataset is the surest route to 7+ on the next challenge.

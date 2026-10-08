# Submission guide: revision 3 (frozen unseen-domain inference)

## Reviewer feedback (2026-10-08)

> the trajectory shows why the domain shift is easier than the framing sounds. raw model gets only 0.2419 public, then agent inspects test and sees basically 2 photometric regimes (147 low-contrast scan / 153 thermal), uses per-image normalization + heavy brightness/contrast/color/blur/noise augmentation, and jumps straight to ~0.62. so a lot of the unseen-domain difficulty can be normalized away rather than requiring a harder change in the underlying decomposition
>
> rn that test inspection is not actually cheating because the rules never ban pooled test statistics / test-distribution adaptation. if we really want frozen unseen-domain generalization i'd explicitly ban aggregate test stats, domain clustering, test-set calibration, etc, while still allowing per-image normalization

## Fix

The problem description has a new section, **"Frozen Per-Image Inference (No Test-Set Adaptation)"**. It does exactly what the reviewer asked:
- **Required:**
  - The pipeline is built, tuned and selected on training data and the description only, then frozen.
  - Each test prediction depends only on that one test image, independent of batch and order.
  - Normalization layers run in inference mode.
- **Allowed:**
  - per-image normalization;
  - per-image test-time augmentation;
  - training-time augmentation chosen from the training data and the domain descriptions in the text;
  - grouped validation, including leaving one training style out.
- **Banned:**
  - statistics pooled across test images;
  - clustering or domain identification on test images;
  - test-set calibration;
  - test-time adaptation (weight or normalization-statistic updates, entropy minimization, pseudo-labeling, self-training);
  - using test images for validation or model selection;
  - inspecting test images to choose preprocessing or augmentation.

Two existing lines now point to the section:
- the Restrictions bullet "Use only ./dataset/public/";
- the leakage-controls bullet about the hidden domain names.

Nothing else changed: data, `prepare.py`, `grade.py`, metric and grading configuration are all the same.

## Form changes

1. **Problem Description:** replace it with `challenge/problem_description.md`. The only differences from revision 2 are the new section and the two pointer bullets.
2. Everything else stays as in revision 2. If revision 2's pipeline and grader are already in the form, there's no new pipeline version and no Run Prepare.
3. Click **Run checks**.

## Reference notebook

It must follow the same rules:
- `model.eval()` for inference;
- no statistics computed over the test folder;
- no choices made by looking at test images.

Per-image normalization and per-image test-time augmentation are fine.

## Reply to the reviewer

> Thanks, agreed. The description now has an explicit "Frozen Per-Image Inference (No Test-Set Adaptation)" section. The whole pipeline must be built, tuned and selected on training data only and then frozen. Each test prediction must depend only on that single test image (identical alone, in any batch or order), with normalization layers in inference mode. It explicitly bans pooled test statistics, clustering or domain identification on test images, test-set calibration, test-time adaptation (weight/normalization-statistic updates, entropy minimization, pseudo-labeling, self-training), using test images for validation or model selection, and inspecting test images to choose preprocessing or augmentation. Per-image normalization, per-image TTA and training-time augmentation chosen from the training data and the stated domain descriptions remain allowed, as you suggested.

---

# Revision 2 (submission size)

## Reviewer feedback

> the trajectory shows a 10 mb submission limit that the prompt never mentions. the agent removed small predicted contributions to fit the file, sacrificing some soft-shadow accuracy. document the limit and provide an encoding approach that fits without forcing predictions to change. instead of nerfing the submissions, optimize the submission size

## Root cause

The old format was Base64 of a zlib-compressed 4×160×192 uint8 tensor. That is 122,880 raw bytes per image, with a redundant fourth channel and no spatial prediction filter.
- **Even perfect answers barely fit:** `labels.csv` is 80 MB for 2,700 rows, so a target is about 30 KB of Base64. Three hundred perfect rows come to about 9 MB of the 10 MB limit.
- **Real model outputs don't fit:** they carry small non-zero softmax tails and rounding flicker everywhere, and zlib can't compress those. On synthetic look-alike scenes, a realistic prediction came to 16–33 MB in the old format.
- **A lossless codec alone isn't enough:** a PNG at full resolution with three channels still came to 15–19 MB for those predictions.
- **No full-resolution 8-bit format can guarantee a fit:** 300 × 30,720 pixels × 3 channels must fit in 7.5 MB of binary. That leaves 2.2 bits per value.

## Fix

These changes guarantee the file fits for every possible prediction. The metric is unchanged.

1. **Output grid is 96×80.** Each output cell is the mean of a 2×2 block of input pixels. Targets are pooled exactly that way in `prepare.py`, with largest-remainder rounding so the channels still sum to 255. Values stay fractional at full 8-bit precision, so soft penumbra and multi-light overlap are still scored.
2. **The payload is a lossless 8-bit RGB PNG.** R, G and B hold lights 1–3. The residual is implied as 255 − R − G − B. The column is renamed `attribution_b64` → `attribution_png`, so the old format can't be confused with the new one. The grader rejects the old zlib payload with a clear error.
3. **Size guarantee:** incompressible data falls back to stored deflate blocks. A 96×80 RGB PNG therefore can't exceed about 31 KB of Base64.
   - Measured worst case: 30,652 chars for pure noise.
   - A 300-row file is at most about 9.3 MB, under the 10 MB limit, whatever the model outputs.
   - On the synthetic look-alike scenes, realistic predictions came to 4–5 MB.
4. **The description documents the 10 MB limit.** It also includes a tested `encode_attribution` / `decode_attribution` / `pool_to_output` snippet that rounds predictions with at most 1/255 change and never thresholds them.
5. **Grader hardening, required by the CLAUDE.md contract:**
   - It now scores any subset of answer ids (public/private split) and ignores extra submission ids.
   - It rejects PNGs that are 16-bit, palette, grey, RGBA or the wrong size, and channel sums above 255.
   - It caps payload length.
6. **`train_targets.csv` uses the same PNG encoding.** Solvers decode targets and encode predictions with the same code. The public download also shrinks.

## Form changes

1. **Problem Description:** replace the whole text with `challenge/problem_description.md`. It has no title line.
   - It now contains one fenced Python code block, the encoder. If the text box mangles it, paste the code lines as they are, without the ``` fences.
   - There are no tables.
2. **Grading Script:** choose Custom and replace it with `challenge/grade.py`.
3. **Grading Configuration:** no change. Keep Maximize, minimum 0, maximum 1.
4. **Pipeline:** click **New Pipeline Version**, paste `challenge/prepare.py`, then click **Run Prepare**. The dataset itself is unchanged, so don't upload a new one.
5. **Difficulty, Title, Tags, Compute:** no change. Keep Hard, the same title, segmentation/image/multimodal and A10G.
6. Click **Run checks**.

**Expected prepared files:**
- `public/train.csv`: 2,400 rows (id, image, scene_id)
- `public/train_targets.csv`: 2,400 rows (id, attribution_png)
- `public/test.csv`: 300 rows
- `public/sample_submission.csv`: 300 rows (id, attribution_png), with an all-black PNG = all residual
- `private/answers.csv`: 300 rows (id, eval_domain, attribution_png)

## Reference solution

Your reference notebook must be updated before you resubmit. It isn't in this repo.
- **Inputs:** read targets with `decode_attribution` (from the description) instead of the old zlib decoder. Pool them, or train at full resolution and pool predictions with `pool_to_output`.
- **Output:** write `id,attribution_png` with `encode_attribution`.
- **Thresholding:** remove any pruning of small contributions. It's no longer needed.

Run it end to end and check that `submission.csv` is under 10 MB.

## Before resubmitting

- **Real sizes:** run `python tools/measure_payload_sizes.py ~/Downloads/public.zip` on the old download. It prints the real old and new target sizes. The new 300-row file should come out at a fraction of a MB to a few MB.
- **Agent scores:** agents no longer need to prune, so their scores may rise slightly from about 0.61. If you get another agent round, check that they stay under 0.7.

## Reply to the reviewer

> Thanks — fixed. The 10 MB limit is now documented in the description. The submission encoding was redesigned so every possible prediction fits without thresholding: predictions are on a 96×80 grid (2×2 block means of the 192×160 image; targets are pooled the same way and remain fractional at 8-bit precision), and each payload is a lossless 8-bit RGB PNG holding the three light channels with the residual implied. A 96×80 RGB PNG cannot exceed ~31 KB of Base64 even for pure noise, so a 300-row file is bounded at ~9.3 MB regardless of model output; smooth predictions are far smaller. The description includes a ready-to-use encoder/decoder (rounding only, at most 1/255 change). Training targets use the same encoding, and the grader now validates the PNG format strictly and scores any subset of ids.

## Important: your responsibility

The Eris rules say *"You may not use any LLM outputs as part of your submission."* Everything here was drafted by an AI assistant. Review every file and make it your own before submitting.

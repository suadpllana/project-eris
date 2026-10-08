## Objective

Decompose each RGB scene into four calibrated contribution fields on a 96 by 80 output grid: shadow from header light 1, shadow from header light 2, shadow from header light 3, and residual unshadowed mass. The four fields sum to one in every output cell.

This benchmark does not retrieve or segment one referred shadow. A cell can receive simultaneous fractional contributions from several lights, and all three light-specific soft fields must be reconstructed together. Binary thresholding discards required attribution information.

## Distinction from Related Shadow Benchmarks

Referring shadow detection selects one shadow instance from an image or video, usually from a language expression, and produces a binary mask. Per-light rendering datasets commonly expose isolated light passes or known scene geometry. This benchmark provides only one composite RGB image and coded light identities. It requires blind decomposition into three simultaneous calibrated contribution fields, including fractional multi-light overlap and soft penumbra, plus residual mass. Solvers therefore need mixture separation and per-cell calibration strategies that a binary referred-mask pipeline does not provide.

Shading and illumination decomposition methods, such as layered shading decomposition for photo retouching, split an image into generic shading or transport layers. They do not assign shadow to specific, identified light sources. Here, each output channel is tied to one coded light, and the fields obey a compositional constraint: the three lights plus residual sum to one in every cell.

This design exposes failures that standard metrics hide:

- Light swapping: a prediction with the correct total shadow assigned to the wrong light scores perfectly on a light-agnostic shadow mask. Here it is penalized in both affected channels.
- Overlap miscalibration: where two shadows overlap, the fractions must split correctly between the lights. Over-claiming one light necessarily under-claims another or the residual.
- Domain collapse: the worst-domain half of the score exposes a model that works on one unseen rendering style and fails on the other. A pooled average would hide that.

## Visual Encoding

- Three four-dot codes in the header define output channels 1, 2, and 3 from left to right.
- The same codes appear beside the corresponding lights in the scene.
- Light position, marker code, and light color are independently randomized per scene.
- Five to seven opaque polygonal objects create overlapping shadows with independently sampled opacity and penumbra.
- Object interiors have zero shadow attribution.
- The fourth field is residual or unshadowed mass.

Training scenes use studio and blueprint rendering. Evaluation scenes use unseen thermal and low-contrast scan domains with stronger rotation, blur, noise, and reduced contrast.

## Output Grid

Input images are 192 by 160 pixels. Targets and predictions use a 96 by 80 grid. Output cell (row r, column c) is the mean attribution over input pixels in rows 2r to 2r+1 and columns 2c to 2c+1. The generator's full-resolution fields were averaged this way and rounded to integers out of 255 that still sum to 255 per cell. Soft penumbra and multi-light overlap therefore remain fractional at the output resolution. A model may predict at full resolution and average each 2 by 2 block, or predict on the 96 by 80 grid directly.

## Public Files

- train.csv — 2,400 labeled-image metadata rows from 1,200 scene families.
- train_targets.csv — ID-keyed training targets in the submission encoding.
- test.csv — 300 unlabeled images from disjoint scene families.
- sample_submission.csv — valid all-residual prediction for every test ID.
- train_images/ and test_images/ — 192 by 160 RGB PNG files.

## Columns

- id — opaque unique identifier.
- image — relative image path.
- scene_id — opaque family identifier used for grouped validation.
- attribution_png — Base64 text of a lossless 8-bit RGB PNG, 96 pixels wide and 80 pixels high.

The PNG's red, green, and blue channels hold light 1, light 2, and light 3 in header order, as integers from 0 to 255 out of 255. Residual mass is implied: residual = 255 minus red minus green minus blue. Every cell must therefore satisfy red + green + blue ≤ 255. Decoded values divided by 255 give fractions in [0,1].

## Submission Size Limit and Encoding

The platform accepts submission files up to 10 MB. The encoding above fits within this limit for every possible prediction, so predictions never need to be thresholded, sparsified, or coarsened to fit:

- The residual channel is implied, so only three channels are stored.
- PNG row filtering plus deflate compresses smooth fields losslessly. Decoding returns exactly the submitted integers.
- A PNG of this size can never exceed about 31 KB of Base64, even for pure noise. When data does not compress, deflate falls back to stored blocks. That bounds the whole 300-row file at about 9.3 MB. Smooth model outputs are much smaller.

Use any standard PNG writer with compression enabled. A ready-to-use encoder:

```python
import base64, io
import numpy as np
from PIL import Image

def encode_attribution(prob):
    """prob: float array (4, 80, 96) = three light fields + residual, any non-negative scale."""
    p = np.clip(np.asarray(prob, dtype=np.float64), 0, None)
    p = 255.0 * p / np.maximum(p.sum(axis=0, keepdims=True), 1e-12)
    q = np.rint(p[:3]).astype(np.int16)
    over = (q.sum(axis=0) > 255).astype(np.int16)  # rounding can overshoot by at most 1
    rows, cols = np.indices(over.shape)
    q[q.argmax(axis=0), rows, cols] -= over
    buffer = io.BytesIO()
    Image.fromarray(q.transpose(1, 2, 0).astype(np.uint8)).save(buffer, format="PNG", compress_level=9)
    return base64.b64encode(buffer.getvalue()).decode("ascii")

def decode_attribution(payload):
    light = np.asarray(Image.open(io.BytesIO(base64.b64decode(payload))), dtype=np.float32).transpose(2, 0, 1) / 255.0
    return np.concatenate([light, 1.0 - light.sum(axis=0, keepdims=True)])  # (4, 80, 96)

def pool_to_output(prob_full):
    """(4, 160, 192) full-resolution field -> (4, 80, 96) scored grid (mean of each 2x2 block)."""
    return np.asarray(prob_full).reshape(4, 80, 2, 96, 2).mean(axis=(2, 4))
```

Rounding to integers out of 255 changes any value by at most 1/255. The same decoder reads train_targets.csv.

## Generalization and Leakage Controls

- Complete scenes remain in one split.
- Training provides two rendering styles per scene; evaluation provides one rendering of an unseen scene.
- Evaluation styles never occur in training.
- Public test files omit targets and domain names. Recovering the domain split from test images is banned (see below).
- Marker codes, colors, positions, IDs, filenames, and row order are independent of attribution values.
- Every image has unique bytes and no scene is reused across splits.

## Frozen Per-Image Inference (No Test-Set Adaptation)

This benchmark measures frozen generalization to unseen rendering domains. Test images are inference inputs only. The rules below are part of the task. A solution that breaks them is invalid, whatever its score.

Required:

- Build, tune, and select the full pipeline using only train.csv, train_targets.csv, train_images/, and this description. That includes weights, preprocessing, augmentation, hyperparameters, thresholds, calibration, and ensemble weights.
- Freeze the pipeline before test inference. Each test prediction must be a function of that one test image and the frozen pipeline only. It must be identical if the image were processed alone, in any order, or in any batch.
- Run every normalization layer in inference mode with statistics frozen from training, for example model.eval() in PyTorch.

Allowed:

- Per-image normalization computed from that single image alone, such as per-image mean and standard deviation, contrast stretching, or histogram equalization.
- Per-image test-time augmentation, such as flips, rotations, or rescaling, averaged within that image.
- Training-time photometric and geometric augmentation chosen from the training data and the domain descriptions in this text, including brightness, contrast, color, blur, noise, and rotation.
- Validation on training scenes grouped by scene_id. Leaving one training style out is a useful proxy for the unseen-domain shift.

Banned:

- Statistics pooled over two or more test images: intensity or color means, histograms, PCA, embeddings, or feature statistics.
- Clustering, grouping, or domain identification across test images, or any attempt to infer the hidden domain split.
- Test-set calibration: fitting temperatures, thresholds, scale factors, histogram matching, or per-group corrections to test images or test predictions.
- Test-time adaptation or training on test images. This includes weight updates, normalization-statistic updates, entropy minimization, self-training, pseudo-labeling, distillation, and self-supervised or unsupervised domain adaptation.
- Using test images for validation, early stopping, model or checkpoint selection, or ensemble weighting.
- Viewing, plotting, or computing statistics over test images to choose preprocessing, augmentation, or hyperparameters.

## Evaluation

Decode each payload to the three light fields with values in [0,1] on the 96 by 80 grid. For each of the three light channels, compute soft Dice and normalized attribution L1 skill over all cells.

- SOFT_DICE = (2 sum(prediction times truth) + epsilon) divided by (sum(prediction squared) + sum(truth squared) + epsilon), with epsilon = 1e-7.
- L1_SKILL = clip(1 minus sum absolute error divided by (sum of predicted and true attribution mass + epsilon), 0, 1).
- IMAGE = 0.70 times mean-channel SOFT_DICE plus 0.30 times mean-channel L1_SKILL.
- DOMAIN = mean IMAGE over one hidden rendering domain.
- SCORE = 0.50 times mean DOMAIN plus 0.50 times minimum DOMAIN.

Soft Dice receives 70% because correct spatial attribution is primary. Normalized L1 skill receives 30% to reward calibrated penumbra and overlap fractions without granting near-perfect credit to an all-background prediction. An empty light map receives zero skill. Both terms evaluate the same three contribution fields. Equal mean and worst-domain weights prevent one rendering style from hiding failure on another. The score is maximized and bounded from 0 to 1.

## Submission

Write ./working/submission.csv with exactly 300 rows and columns id,attribution_png in that order. Every test ID must occur exactly once. Each payload must be nonempty valid Base64 of an 8-bit RGB PNG (PNG colour type 2, no alpha, no palette, not 16-bit), 96 pixels wide and 80 pixels high, with red + green + blue ≤ 255 in every cell. The file must be at most 10 MB; the encoding above always satisfies this.

## Restrictions

- Compute tier: A10G.
- Use only ./dataset/public/. Test images are for frozen per-image inference only, as defined in "Frozen Per-Image Inference".
- No private targets, raw labels, generator internals, external images, external annotations, hosted APIs, ID shortcuts, filename shortcuts, or metadata shortcuts.
- Public general-purpose pretrained vision weights already available offline are allowed.
- Complete training and inference within approximately one hour.

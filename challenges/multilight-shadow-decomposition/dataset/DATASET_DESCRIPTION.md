# Dataset Description

## Overview

Multi-Light Fractional Shadow Decomposition is a fully synthetic dataset for dense multi-source attribution. Each image contains three coded lights, opaque objects, soft penumbrae, and overlapping shadow effects. Labels give three simultaneous light-specific contribution maps plus a residual map.

## Source and License

- No external source URL.
- Generated offline with fixed seeds and no external assets. Repeated generation is byte-identical inside a fixed Python, NumPy, and Pillow environment; Pillow rasterizer versions may differ at anti-aliased polygon boundary pixels.
- Released under CC0 1.0 Universal.

## Raw Files

- images/ — 2,700 unique 192 by 160 RGB PNGs.
- labels.csv — IDs, filenames, scene groups, private domains, and compressed attribution tensors.
- generation_manifest.json — generation settings.
- generate_dataset.py — complete generator.
- README_DATASET.md — concise reproduction notes.
- LICENSE_DATA.txt — license statement.

## Raw Columns

- id — opaque image identifier.
- filename — PNG filename.
- scene_id — semantic scene family.
- eval_domain — private rendering domain.
- attribution_b64 — zlib-compressed Base64 uint8 tensor with shape 4 by 160 by 192.

## Split and Limitations

- 1,200 training scenes have two public rendering styles.
- 300 held-out scenes use thermal or low-contrast scan rendering.
- Scene groups and rendering styles are disjoint across the split.
- Scenes are procedural two-dimensional approximations and do not model full physical light transport.

#!/usr/bin/env python3
"""
Measure real target payload sizes in the old and new submission encodings.

Run it locally on the *old* public.zip you downloaded from Shipd (the version
whose train_targets.csv still has the full-resolution zlib `attribution_b64`
column):

    python measure_payload_sizes.py ~/Downloads/public.zip

It re-encodes every training target the way the new prepare.py does
(2x2 mean pool -> 96x80 RGB PNG, residual implied) and prints what a
300-row submission of perfect predictions would weigh in each format.
Needs numpy, pandas and Pillow only.
"""
import base64
import io
import sys
import zipfile
import zlib

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "challenge"))
from prepare import decode_source, encode_png, pool_tensor  # noqa: E402


def main(path):
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith("train_targets.csv"))
        targets = pd.read_csv(archive.open(name), dtype=str)
    old, new = [], []
    for value in targets.attribution_b64:
        old.append(len(value))
        new.append(len(encode_png(pool_tensor(decode_source(value))[:3])))
    old, new = np.array(old), np.array(new)
    row = 300 / 1e6
    print(f"{len(targets)} training targets")
    print(f"old zlib 4x160x192:  mean {old.mean():8.0f} chars, max {old.max():6d} -> 300 perfect rows ~ {old.mean() * row:.2f} MB")
    print(f"new PNG 3x80x96:     mean {new.mean():8.0f} chars, max {new.max():6d} -> 300 perfect rows ~ {new.mean() * row:.2f} MB")
    print("new format hard bound (pure noise): ~31,000 chars per row -> ~9.3 MB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "public.zip")

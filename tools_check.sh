#!/bin/bash
# Local end-to-end check: prepare twice (determinism), then grade the sample submission.
# usage: bash tools_check.sh <scratch_dir>
set -e
OUT=${1:-/tmp/eris_check}
rm -rf "$OUT"
python3 challenge/prepare.py dataset/raw "$OUT/run1/public" "$OUT/run1/private"
python3 challenge/prepare.py dataset/raw "$OUT/run2/public" "$OUT/run2/private"
diff <(cd "$OUT/run1" && sha256sum public/* private/*) <(cd "$OUT/run2" && sha256sum public/* private/*) \
  && echo "prepare.py is deterministic"
(cd challenge && python3 -c "
import pandas as pd; from grade import grade
P='$OUT/run1/'
print('sample_submission RMSLE:', round(grade(pd.read_csv(P+'public/sample_submission.csv'), pd.read_csv(P+'private/answers.csv')), 4))")

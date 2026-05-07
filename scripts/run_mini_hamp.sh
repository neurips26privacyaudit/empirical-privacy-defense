#!/bin/bash
# Mini experiment: 3 conditions, ~5 min total on a GPU.
# Uses --n_df 5000 (5k MNIST samples) so each model trains fast.
# Uses mislabeled canary: a '3' labeled as '1' — persistently high gradient
# norm, maximally detectable by the filter, easily memorized under HAMP.

set -e
OUT=/tmp/hamp_mini

# ---------- 1. Baseline: no defense ----------
echo "=== [1/3] defense_type=none ==="
python3 parallel_audit_model_hamp.py \
  --data_name mnist --model_name cnn \
  --n_reps 40 --n_epochs 100 \
  --n_df 5000 \
  --target_type mislabeled --mislabeled_target_class 1 \
  --defense_type none \
  --out "$OUT/none"

# ---------- 2. HAMP defense (should still be broken by the label-only attack) ----------
echo "=== [2/3] defense_type=hamp ==="
python3 parallel_audit_model_hamp.py \
  --data_name mnist --model_name cnn \
  --n_reps 40 --n_epochs 100 \
  --n_df 5000 \
  --target_type mislabeled --mislabeled_target_class 1 \
  --defense_type hamp --hamp_gamma 0.95 \
  --out "$OUT/hamp"

# ---------- 3. Gradient-norm filter + ascent (should defend) ----------
echo "=== [3/3] defense_type=filter ==="
python3 parallel_audit_model_hamp.py \
  --data_name mnist --model_name cnn \
  --n_reps 40 --n_epochs 100 \
  --n_df 5000 \
  --target_type mislabeled --mislabeled_target_class 1 \
  --defense_type filter \
  --defense_k 5 --defense_apply_ascent \
  --out "$OUT/filter"

echo ""
echo "=== Results ==="
python3 compare_hamp_results.py --out_root "$OUT"

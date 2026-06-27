#!/usr/bin/env bash
# Runs the slice-CLIP baseline over all three datasets (val + test) and writes
# one combined Kaggle submission. CPU-only on this machine (no CUDA/MPS).
set -euo pipefail

PROJECT_ROOT="D:/Projects/Hackathons/EHL_Hackathon_Cross-modal_Content-based_Retrieval_for_3D_Medical_Images"
DATA_ROOT="$PROJECT_ROOT/kaggle_dataset"
BASELINE="$PROJECT_ROOT/ehl-paris-2026-medical-retrieval-main/ehl-paris-2026-medical-retrieval-main/slice_clip_baseline.py"
OUT="$PROJECT_ROOT/slice_clip_submission.csv"

uv run "$BASELINE" \
  --data-root "$DATA_ROOT" \
  --train-pair-csv "$DATA_ROOT/dataset1/train_pairs.csv" \
  --query-csv "$DATA_ROOT/dataset1/val_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset1/val_gallery.csv" \
  --query-csv "$DATA_ROOT/dataset1/test_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset1/test_gallery.csv" \
  --query-csv "$DATA_ROOT/dataset2/val_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset2/val_gallery.csv" \
  --query-csv "$DATA_ROOT/dataset2/test_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset2/test_gallery.csv" \
  --query-csv "$DATA_ROOT/dataset3/val_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset3/val_gallery.csv" \
  --query-csv "$DATA_ROOT/dataset3/test_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset3/test_gallery.csv" \
  --out "$OUT"

echo "Wrote submission to $OUT"

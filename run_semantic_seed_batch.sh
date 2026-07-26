#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Build guideline passages from PDF files
#
# Usage:
# 1. Update PDF_DIR and the output paths below.
# 2. Ensure that pypdf is installed in the active Python environment.
# 3. Run:
#      bash run_build_guideline_passages.sh
#
# Input:
#   data/guideline_pdfs/*.pdf
#
# Outputs:
#   data/guideline_passages.jsonl
#   data/guideline_passages.meta.json
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PDF_DIR="$SCRIPT_DIR/data/guideline_pdfs"
OUT_JSONL="$SCRIPT_DIR/data/guideline_passages.jsonl"
OUT_META_JSON="$SCRIPT_DIR/data/guideline_passages.meta.json"

mkdir -p "$SCRIPT_DIR/data"

python -u "$SCRIPT_DIR/build_guideline_passages.py" \
  --pdf_dir "$PDF_DIR" \
  --out_jsonl "$OUT_JSONL" \
  --out_meta_json "$OUT_META_JSON" \
  --min_words 100 \
  --max_words 220 \
  --overlap_sentences 1 \
  --ref_score_threshold 0.75
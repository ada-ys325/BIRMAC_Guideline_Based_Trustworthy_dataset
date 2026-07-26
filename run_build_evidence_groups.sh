#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Build within-guideline evidence candidate pools.
#
# Pipeline:
#   Semantic passage seeds
#   -> sentence-transformer embeddings
#   -> within-guideline nearest-neighbor grouping
#   -> deduplicated evidence candidate pools
#
# Usage:
# 1. Update the input and output paths below if necessary.
# 2. Ensure that sentence-transformers and numpy are installed.
# 3. Run:
#      bash run_build_evidence_groups.sh
#
# Input:
#   data/semantic_question_seeds.jsonl
#
# Outputs:
#   data/evidence_candidate_pools.jsonl
#   data/evidence_candidate_pools.stats.json
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_JSONL="$SCRIPT_DIR/data/semantic_question_seeds.jsonl"
OUT_JSONL="$SCRIPT_DIR/data/evidence_candidate_pools.jsonl"
OUT_STATS="$SCRIPT_DIR/data/evidence_candidate_pools.stats.json"

MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE=128

MIN_PASSAGES=15
MAX_PASSAGES=30
MIN_SIM=0.42
DEDUP_JACCARD=0.55

mkdir -p "$SCRIPT_DIR/data"

python -u "$SCRIPT_DIR/build_evidence_groups.py" \
  --input_jsonl "$INPUT_JSONL" \
  --out_jsonl "$OUT_JSONL" \
  --out_stats "$OUT_STATS" \
  --model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --min_passages "$MIN_PASSAGES" \
  --max_passages "$MAX_PASSAGES" \
  --min_sim "$MIN_SIM" \
  --dedup_jaccard "$DEDUP_JACCARD"

# Optional arguments:
#
# Include the original passage text in the embedding representation:
#   --use_source_text_for_embedding
#
# Allow candidate pools to contain passages from different guidelines:
#   --cross_guideline
#
# The default example above uses semantic seeds only and keeps every
# candidate pool within a single guideline.
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Evaluate atomic claims with an AutoAIS-style factual
# consistency model.
#
# Pipeline:
#   Atomic medical claims with cited passages
#   -> joint passage-level NLI verification
#   -> per-claim entailment decisions
#   -> per-answer support rates
#   -> dataset-level summary statistics
#
# Usage:
# 1. Update the input, output, or model settings if necessary.
# 2. Ensure that torch, transformers, sentencepiece, and tqdm
#    are installed.
# 3. Run:
#      bash run_atomic_fcm_veriscore.sh
#
# Input:
#   data/answers_atomic_facts.jsonl
#
# Outputs:
#   data/answers_atomic_fcm_veriscore.per_claim.jsonl
#   data/answers_atomic_fcm_veriscore.per_item.jsonl
#   data/answers_atomic_fcm_veriscore.summary.json
#   data/answers_atomic_fcm_veriscore.augmented.jsonl
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_JSONL="$SCRIPT_DIR/data/answers_atomic_facts.jsonl"
OUT_PREFIX="$SCRIPT_DIR/data/answers_atomic_fcm_veriscore"

# The model may be a Hugging Face model identifier or a local path.
export AUTOAIS_MODEL="google/t5_xxl_true_nli_mixture"

# Set to 1 when the model is already cached and execution should
# remain completely offline.
export LOCAL_FILES_ONLY=0

export FCM_MAX_INPUT_TOKENS=4096

mkdir -p "$SCRIPT_DIR/data"

python -u "$SCRIPT_DIR/eval_atomic_fcm_veriscore.py" \
  --input "$INPUT_JSONL" \
  --out_prefix "$OUT_PREFIX" \
  --start_idx 0 \
  --end_idx -1 \
  --limit -1 \
  --write_augmented

# Optional argument:
#
# Also evaluate every cited passage separately for diagnostic
# analysis. This is slower than joint evidence evaluation:
#   --individual_diagnostics
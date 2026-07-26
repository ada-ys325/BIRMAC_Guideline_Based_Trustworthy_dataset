#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Extract citation-grounded atomic medical claims using the
# OpenAI Batch API.
#
# Pipeline:
#   Cited patient-facing answers
#   -> GPT-4o atomic claim extraction
#   -> sentence-level claim decomposition
#   -> atomic claims with inherited citations
#
# Usage:
# 1. Update the input and output paths below if necessary.
# 2. Set the OpenAI API key:
#      export OPENAI_API_KEY="your-api-key"
# 3. Run:
#      bash run_atomic_fact_gpt4o.sh
#
# This script builds and submits split batch jobs.
#
# Input:
#   data/answers_cited_atomic_ready.jsonl
#
# Batch files:
#   data/atomic_fact_batch_jobs/
#
# Final output after consuming completed batches:
#   data/answers_atomic_facts.jsonl
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC_JSONL="$SCRIPT_DIR/data/answers_cited_atomic_ready.jsonl"
BATCH_WORKDIR="$SCRIPT_DIR/data/atomic_fact_batch_jobs"

OPENAI_MODEL="gpt-4o"
MAX_OUTPUT_TOKENS=1600
TEMPERATURE=0

START_IDX=0
END_IDX=-1
CHUNK_SIZE=50
JOB_NAME="atomic_fact_batch"

mkdir -p "$BATCH_WORKDIR"

# Build and submit the atomic-claim extraction batch jobs.
python -u "$SCRIPT_DIR/atomic_fact_gpt4o.py" run_split \
  --src "$SRC_JSONL" \
  --out_dir "$BATCH_WORKDIR" \
  --start_idx "$START_IDX" \
  --end_idx "$END_IDX" \
  --chunk_size "$CHUNK_SIZE" \
  --job_name "$JOB_NAME" \
  --openai_model "$OPENAI_MODEL" \
  --max_output_tokens "$MAX_OUTPUT_TOKENS" \
  --temperature "$TEMPERATURE" \
  --completion_window "24h"

# ============================================================
# After the batches have completed, set TSV_PATH to the batch
# map generated in BATCH_WORKDIR and run the commands below.
#
# Example:
#
# TSV_PATH="$BATCH_WORKDIR/atomic_fact_batch.batches_YYYYMMDD_HHMMSS.tsv"
# OUT_JSONL="$SCRIPT_DIR/data/answers_atomic_facts.jsonl"
#
# python -u "$SCRIPT_DIR/atomic_fact_gpt4o.py" status_from_tsv \
#   --tsv_path "$TSV_PATH"
#
# python -u "$SCRIPT_DIR/atomic_fact_gpt4o.py" consume_from_tsv \
#   --tsv_path "$TSV_PATH" \
#   --src "$SRC_JSONL" \
#   --out_jsonl "$OUT_JSONL" \
#   --raw_out_dir "$BATCH_WORKDIR/raw_outputs" \
#   --failed_out_dir "$BATCH_WORKDIR/failed_ids" \
#   --start_idx "$START_IDX" \
#   --end_idx "$END_IDX" \
#   --openai_model "$OPENAI_MODEL" \
#   --overwrite
# ============================================================
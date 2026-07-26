#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Generate cited patient-facing answers using the OpenAI Batch API.
#
# Pipeline:
#   Questions with retrieved guideline passages
#   -> GPT-4o answer generation
#   -> sentence-level citations
#   -> structured answers prepared for atomic claim extraction
#
# Usage:
# 1. Update the input and output paths below if necessary.
# 2. Set the OpenAI API key:
#      export OPENAI_API_KEY="your-api-key"
# 3. Run:
#      bash run_answer_batch.sh
#
# This script builds and submits split batch jobs.
#
# Input:
#   data/group_questions_retrieved_top5.jsonl
#
# Batch files:
#   data/answer_batch_jobs/
#
# Final output after consuming completed batches:
#   data/answers_cited_atomic_ready.jsonl
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC_JSONL="$SCRIPT_DIR/data/group_questions_retrieved_top5.jsonl"
BATCH_WORKDIR="$SCRIPT_DIR/data/answer_batch_jobs"

OPENAI_MODEL="gpt-4o"
MAX_OUTPUT_TOKENS=1800
TOP_N_PASSAGES=5
MAX_PASSAGE_CHARS=1800
TEMPERATURE=0.2

START_IDX=0
END_IDX=-1
CHUNK_SIZE=30
JOB_NAME="answer_batch"

mkdir -p "$BATCH_WORKDIR"

# Build and submit the answer-generation batch jobs.
python -u "$SCRIPT_DIR/answer_batch.py" run_split \
  --src "$SRC_JSONL" \
  --out_dir "$BATCH_WORKDIR" \
  --start_idx "$START_IDX" \
  --end_idx "$END_IDX" \
  --chunk_size "$CHUNK_SIZE" \
  --job_name "$JOB_NAME" \
  --openai_model "$OPENAI_MODEL" \
  --max_output_tokens "$MAX_OUTPUT_TOKENS" \
  --top_n_passages "$TOP_N_PASSAGES" \
  --max_passage_chars "$MAX_PASSAGE_CHARS" \
  --temperature "$TEMPERATURE" \
  --completion_window "24h"

# ============================================================
# After the batches have completed, set TSV_PATH to the batch
# map generated in BATCH_WORKDIR and run the commands below.
#
# Example:
#
# TSV_PATH="$BATCH_WORKDIR/answer_batch.batches_YYYYMMDD_HHMMSS.tsv"
# OUT_JSONL="$SCRIPT_DIR/data/answers_cited_atomic_ready.jsonl"
#
# python -u "$SCRIPT_DIR/answer_batch.py" status_from_tsv \
#   --tsv_path "$TSV_PATH"
#
# python -u "$SCRIPT_DIR/answer_batch.py" consume_from_tsv \
#   --tsv_path "$TSV_PATH" \
#   --src "$SRC_JSONL" \
#   --out_jsonl "$OUT_JSONL" \
#   --raw_out_dir "$BATCH_WORKDIR/raw_outputs" \
#   --failed_out_dir "$BATCH_WORKDIR/failed_ids" \
#   --start_idx "$START_IDX" \
#   --end_idx "$END_IDX" \
#   --top_n_passages "$TOP_N_PASSAGES" \
#   --overwrite
# ============================================================
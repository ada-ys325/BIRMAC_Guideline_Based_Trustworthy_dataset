#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Build a FAISS passage index with Qwen3-Embedding-4B.
#
# Pipeline:
#   Annotated guideline passages
#   -> Qwen3 passage embeddings
#   -> normalized embedding vectors
#   -> FAISS IndexFlatIP index
#
# Usage:
# 1. Update the input and output paths below if necessary.
# 2. Ensure that torch, faiss, sentence-transformers, and
#    transformers are installed.
# 3. Ensure that Qwen/Qwen3-Embedding-4B is available locally
#    or can be downloaded from Hugging Face.
# 4. Run:
#      bash run_build_faiss_passage_index.sh
#
# Input:
#   data/semantic_question_seeds.jsonl
#
# Output directory:
#   data/faiss_passage_index/
#
# Generated files:
#   index.faiss
#   passages.jsonl
#   embeddings.npy
#   meta.json
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_JSONL="$SCRIPT_DIR/data/semantic_question_seeds.jsonl"
OUT_DIR="$SCRIPT_DIR/data/faiss_passage_index"

MODEL_NAME="Qwen/Qwen3-Embedding-4B"
BATCH_SIZE=4
MAX_SEQ_LENGTH=1024
DEVICE="cuda"
TORCH_DTYPE="float16"

SMOKE_QUERY="What treatments are recommended for this condition?"
SMOKE_TOP_K=5

mkdir -p "$SCRIPT_DIR/data"

python -u "$SCRIPT_DIR/build_faiss_passage_index.py" \
  --input_jsonl "$INPUT_JSONL" \
  --out_dir "$OUT_DIR" \
  --model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --device "$DEVICE" \
  --torch_dtype "$TORCH_DTYPE" \
  --save_embeddings \
  --smoke_query "$SMOKE_QUERY" \
  --smoke_top_k "$SMOKE_TOP_K"

# Optional arguments:
#
# Store the complete embedding text in passages.jsonl:
#   --keep_embedding_text
#
# Replace files in an existing non-empty output directory:
#   --overwrite
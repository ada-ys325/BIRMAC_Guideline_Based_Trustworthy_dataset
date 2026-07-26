#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Retrieve and rerank guideline passages for generated questions.
#
# Pipeline:
#   Patient questions
#   -> Qwen3-Embedding-4B query encoding
#   -> FAISS top-k retrieval
#   -> Qwen3-Reranker-4B reranking
#   -> top-5 passages per question
#
# Usage:
# 1. Update the input and output paths below if necessary.
# 2. Ensure that torch, faiss, sentence-transformers, and
#    transformers are installed.
# 3. Ensure that both Qwen models are available locally or can
#    be downloaded from Hugging Face.
# 4. Run:
#      bash run_retrieve_questions_with_faiss_rerank.sh
#
# Inputs:
#   data/group_questions.jsonl
#   data/faiss_passage_index/
#
# Output:
#   data/group_questions_retrieved_top5.jsonl
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

QUESTIONS_JSONL="$SCRIPT_DIR/data/group_questions.jsonl"
INDEX_DIR="$SCRIPT_DIR/data/faiss_passage_index"
OUT_JSONL="$SCRIPT_DIR/data/group_questions_retrieved_top5.jsonl"

EMBEDDING_MODEL="Qwen/Qwen3-Embedding-4B"
RERANKER_MODEL="Qwen/Qwen3-Reranker-4B"

TOP_K_RECALL=50
TOP_K_OUTPUT=5

EMBED_BATCH_SIZE=32
RERANK_BATCH_SIZE=8

EMBEDDING_MAX_SEQ_LENGTH=1024
RERANK_MAX_LENGTH=2048
RERANK_DOC_MAX_CHARS=2500
OUTPUT_TEXT_MAX_CHARS=2000

DEVICE="cuda"
TORCH_DTYPE="float16"

START_IDX=0
LIMIT=-1

mkdir -p "$SCRIPT_DIR/data"

python -u "$SCRIPT_DIR/retrieve_questions_with_faiss_rerank.py" \
  --questions_jsonl "$QUESTIONS_JSONL" \
  --index_dir "$INDEX_DIR" \
  --out_jsonl "$OUT_JSONL" \
  --embedding_model "$EMBEDDING_MODEL" \
  --reranker_model "$RERANKER_MODEL" \
  --top_k_recall "$TOP_K_RECALL" \
  --top_k_output "$TOP_K_OUTPUT" \
  --embed_batch_size "$EMBED_BATCH_SIZE" \
  --rerank_batch_size "$RERANK_BATCH_SIZE" \
  --embedding_max_seq_length "$EMBEDDING_MAX_SEQ_LENGTH" \
  --rerank_max_length "$RERANK_MAX_LENGTH" \
  --rerank_doc_max_chars "$RERANK_DOC_MAX_CHARS" \
  --output_text_max_chars "$OUTPUT_TEXT_MAX_CHARS" \
  --device "$DEVICE" \
  --torch_dtype "$TORCH_DTYPE" \
  --start_idx "$START_IDX" \
  --limit "$LIMIT" \
  --include_anchors_in_faiss_query

# Optional arguments:
#
# Include retrieval anchor terms in the reranker query:
#   --include_anchors_in_rerank_query
#
# Replace an existing output file instead of resuming:
#   --overwrite
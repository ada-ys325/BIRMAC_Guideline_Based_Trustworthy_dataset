#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Group-level question generation OpenAI Batch wrapper
#
# Commands:
#   build
#   build_split
#   submit
#   run_split / submit_split
#   status
#   status_from_tsv
#   consume
#   consume_from_tsv
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MYPROJ="${MYPROJ:-$HOME/links/projects/def-lokkerc/ys325}"
IMAGE="${IMAGE:-$MYPROJ/containers/nv-pytorch_24.09-py3.sif}"

HOST_BIRMAC_DIR="${HOST_BIRMAC_DIR:-$SCRIPT_DIR}"
CONT_BIRMAC_DIR="${CONT_BIRMAC_DIR:-/workspace/BIRMAC_last_version}"

PKG_OPENAI="${PKG_OPENAI:-$MYPROJ/pkgs/openai_send}"
PKG_JSON="${PKG_JSON:-$MYPROJ/pkgs/factscore_atomic}"
PKG_ST="${PKG_ST:-$MYPROJ/pkgs/factscore_st}"

SRC_JSONL="${SRC_JSONL:-data/evidence_candidate_pools_2025_all_within_guideline_top30_min15.jsonl}"
BATCH_WORKDIR="${BATCH_WORKDIR:-data/group_question_batch_jobs}"
OUT_JSONL="${OUT_JSONL:-data/group_questions_2025_all_gpt4o_question_only_raw.jsonl}"

OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-900}"
MAX_SOURCE_CHARS="${MAX_SOURCE_CHARS:-1000}"
COMPLETION_WINDOW="${COMPLETION_WINDOW:-24h}"

START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:--1}"
CHUNK_SIZE="${CHUNK_SIZE:-30}"
JOB_NAME="${JOB_NAME:-group_questions_${START_IDX}_${END_IDX}}"

INPUT_JSONL="${INPUT_JSONL:-}"
META_JSON="${META_JSON:-}"
BATCH_ID="${BATCH_ID:-}"
TSV_PATH="${TSV_PATH:-}"
RAW_OUT_JSONL="${RAW_OUT_JSONL:-}"
FAILED_IDS_OUT="${FAILED_IDS_OUT:-}"
RAW_OUT_DIR="${RAW_OUT_DIR:-$BATCH_WORKDIR/raw_outputs}"
FAILED_OUT_DIR="${FAILED_OUT_DIR:-$BATCH_WORKDIR/failed_ids}"
OVERWRITE_OUT="${OVERWRITE_OUT:-0}"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$MYPROJ/pip_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$MYPROJ/xdg_cache}"
export HF_HOME="${HF_HOME:-$MYPROJ/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$MYPROJ/hf_home/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$MYPROJ/hf_home/hub}"

mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MYPROJ/.logs"

module purge || true
module load StdEnv/2023
module load apptainer/1.3.5

if command -v apptainer >/dev/null 2>&1; then
  CTR_CMD="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  CTR_CMD="singularity"
else
  echo "❌ Neither apptainer nor singularity found in PATH" >&2
  exit 99
fi

[[ -f "$IMAGE" ]] || { echo "❌ Missing image: $IMAGE" >&2; exit 2; }
[[ -d "$HOST_BIRMAC_DIR" ]] || { echo "❌ Missing HOST_BIRMAC_DIR: $HOST_BIRMAC_DIR" >&2; exit 3; }
[[ -f "$HOST_BIRMAC_DIR/group_question_batch.py" ]] || { echo "❌ Missing group_question_batch.py: $HOST_BIRMAC_DIR/group_question_batch.py" >&2; exit 4; }

CMD="${1:-}"
shift || true

if [[ -z "$CMD" ]]; then
  cat >&2 <<EOF
Usage:
  bash run_group_question_batch.sh build
  bash run_group_question_batch.sh build_split
  bash run_group_question_batch.sh submit
  bash run_group_question_batch.sh run_split
  bash run_group_question_batch.sh status
  bash run_group_question_batch.sh status_from_tsv
  bash run_group_question_batch.sh consume
  bash run_group_question_batch.sh consume_from_tsv
EOF
  exit 2
fi

needs_key=0
case "$CMD" in
  submit|submit_split|run_split|status|status_from_tsv|consume|consume_from_tsv)
    needs_key=1
    ;;
esac

if [[ "$needs_key" == "1" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY is required for CMD=$CMD" >&2
  exit 10
fi

echo "============================================================"
echo "[INFO] Container cmd       : $CTR_CMD"
echo "[INFO] Image               : $IMAGE"
echo "[INFO] HOST_BIRMAC_DIR     : $HOST_BIRMAC_DIR"
echo "[INFO] CONT_BIRMAC_DIR     : $CONT_BIRMAC_DIR"
echo "[INFO] SRC_JSONL           : $SRC_JSONL"
echo "[INFO] BATCH_WORKDIR       : $BATCH_WORKDIR"
echo "[INFO] OUT_JSONL           : $OUT_JSONL"
echo "[INFO] CMD                 : $CMD"
echo "[INFO] OPENAI_MODEL        : $OPENAI_MODEL"
echo "[INFO] MAX_OUTPUT_TOKENS   : $MAX_OUTPUT_TOKENS"
echo "[INFO] MAX_SOURCE_CHARS    : $MAX_SOURCE_CHARS"
echo "[INFO] START_IDX           : $START_IDX"
echo "[INFO] END_IDX             : $END_IDX"
echo "[INFO] CHUNK_SIZE          : $CHUNK_SIZE"
echo "[INFO] JOB_NAME            : $JOB_NAME"
echo "[INFO] BATCH_ID            : ${BATCH_ID:-<none>}"
echo "[INFO] TSV_PATH            : ${TSV_PATH:-<none>}"
echo "[INFO] OVERWRITE_OUT       : $OVERWRITE_OUT"
echo "============================================================"

unset SSL_CERT_FILE REQUESTS_CA_BUNDLE PIP_CERT || true

"$CTR_CMD" exec --nv \
  --bind "$MYPROJ:$MYPROJ" \
  --bind "$HOST_BIRMAC_DIR:$CONT_BIRMAC_DIR" \
  --env OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  --env CONT_BIRMAC_DIR="$CONT_BIRMAC_DIR" \
  --env HOST_BIRMAC_DIR="$HOST_BIRMAC_DIR" \
  --env OPENAI_MODEL="$OPENAI_MODEL" \
  --env MAX_OUTPUT_TOKENS="$MAX_OUTPUT_TOKENS" \
  --env MAX_SOURCE_CHARS="$MAX_SOURCE_CHARS" \
  --env COMPLETION_WINDOW="$COMPLETION_WINDOW" \
  --env SRC_JSONL="$SRC_JSONL" \
  --env BATCH_WORKDIR="$BATCH_WORKDIR" \
  --env OUT_JSONL="$OUT_JSONL" \
  --env CMD="$CMD" \
  --env START_IDX="$START_IDX" \
  --env END_IDX="$END_IDX" \
  --env CHUNK_SIZE="$CHUNK_SIZE" \
  --env JOB_NAME="$JOB_NAME" \
  --env INPUT_JSONL="$INPUT_JSONL" \
  --env META_JSON="$META_JSON" \
  --env BATCH_ID="$BATCH_ID" \
  --env TSV_PATH="$TSV_PATH" \
  --env RAW_OUT_JSONL="$RAW_OUT_JSONL" \
  --env FAILED_IDS_OUT="$FAILED_IDS_OUT" \
  --env RAW_OUT_DIR="$RAW_OUT_DIR" \
  --env FAILED_OUT_DIR="$FAILED_OUT_DIR" \
  --env OVERWRITE_OUT="$OVERWRITE_OUT" \
  --env PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env HF_HOME="$HF_HOME" \
  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  --env HF_HUB_DISABLE_XET=1 \
  --env HF_HUB_MAX_WORKERS=1 \
  --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONNOUSERSITE=1 \
  --env TOKENIZERS_PARALLELISM=false \
  --env PKG_OPENAI="$PKG_OPENAI" \
  --env PKG_JSON="$PKG_JSON" \
  --env PKG_ST="$PKG_ST" \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  --env PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
  "$IMAGE" /bin/bash -lc '
    set -euo pipefail
    cd "$CONT_BIRMAC_DIR"
    export PYTHONPATH="${PKG_OPENAI}:${PKG_JSON}:${PKG_ST}:${PYTHONPATH:-}"

    echo "[INFO] Inside container pwd=$(pwd)"
    echo "[INFO] python=$(command -v python)"
    python -V
    echo "[INFO] group_question_batch.py:"
    ls -l ./group_question_batch.py
    echo "[INFO] PYTHONPATH=$PYTHONPATH"

    mkdir -p "$BATCH_WORKDIR" "$(dirname "$OUT_JSONL")" "$RAW_OUT_DIR" "$FAILED_OUT_DIR"

    case "$CMD" in
      build)
        INPUT_JSONL="${INPUT_JSONL:-$BATCH_WORKDIR/${JOB_NAME}.jsonl}"
        python ./group_question_batch.py build \
          --src "$SRC_JSONL" \
          --out_jsonl "$INPUT_JSONL" \
          --start_idx "$START_IDX" \
          --end_idx "$END_IDX" \
          --openai_model "$OPENAI_MODEL" \
          --max_output_tokens "$MAX_OUTPUT_TOKENS" \
          --max_source_chars "$MAX_SOURCE_CHARS"
        echo "[INFO] build done: $INPUT_JSONL"
        ;;

      build_split)
        python ./group_question_batch.py build_split \
          --src "$SRC_JSONL" \
          --out_dir "$BATCH_WORKDIR" \
          --start_idx "$START_IDX" \
          --end_idx "$END_IDX" \
          --chunk_size "$CHUNK_SIZE" \
          --job_name "$JOB_NAME" \
          --openai_model "$OPENAI_MODEL" \
          --max_output_tokens "$MAX_OUTPUT_TOKENS" \
          --max_source_chars "$MAX_SOURCE_CHARS"
        ;;

      submit)
        INPUT_JSONL="${INPUT_JSONL:-$BATCH_WORKDIR/${JOB_NAME}.jsonl}"
        META_JSON="${META_JSON:-$BATCH_WORKDIR/${JOB_NAME}.meta.json}"
        python ./group_question_batch.py submit \
          --input_jsonl "$INPUT_JSONL" \
          --openai_model "$OPENAI_MODEL" \
          --completion_window "$COMPLETION_WINDOW" \
          --metadata_name "$JOB_NAME" \
          --batch_meta_out "$META_JSON"
        ;;

      submit_split|run_split)
        python ./group_question_batch.py run_split \
          --src "$SRC_JSONL" \
          --out_dir "$BATCH_WORKDIR" \
          --start_idx "$START_IDX" \
          --end_idx "$END_IDX" \
          --chunk_size "$CHUNK_SIZE" \
          --job_name "$JOB_NAME" \
          --openai_model "$OPENAI_MODEL" \
          --max_output_tokens "$MAX_OUTPUT_TOKENS" \
          --max_source_chars "$MAX_SOURCE_CHARS" \
          --completion_window "$COMPLETION_WINDOW"
        echo "[INFO] Latest TSV:"
        ls -t "$BATCH_WORKDIR"/"${JOB_NAME}".batches_*.tsv 2>/dev/null | head -1 || true
        ;;

      status)
        if [[ -z "${BATCH_ID:-}" ]]; then
          echo "Usage: BATCH_ID=batch_xxx bash run_group_question_batch.sh status" >&2
          exit 2
        fi
        python ./group_question_batch.py status --batch_id "$BATCH_ID"
        ;;

      status_from_tsv)
        if [[ -z "${TSV_PATH:-}" ]]; then
          echo "Usage: TSV_PATH=/path/to/batches.tsv bash run_group_question_batch.sh status_from_tsv" >&2
          exit 2
        fi
        python ./group_question_batch.py status_from_tsv --tsv_path "$TSV_PATH"
        ;;

      consume)
        if [[ -z "${BATCH_ID:-}" ]]; then
          echo "Usage: BATCH_ID=batch_xxx bash run_group_question_batch.sh consume" >&2
          exit 2
        fi
        RAW_OUT_JSONL="${RAW_OUT_JSONL:-$BATCH_WORKDIR/${JOB_NAME}.output.jsonl}"
        FAILED_IDS_OUT="${FAILED_IDS_OUT:-$BATCH_WORKDIR/${JOB_NAME}.failed_ids.txt}"
        overwrite_args=()
        if [[ "$OVERWRITE_OUT" == "1" ]]; then overwrite_args+=(--overwrite); fi
        python ./group_question_batch.py consume \
          --batch_id "$BATCH_ID" \
          --src "$SRC_JSONL" \
          --out_jsonl "$OUT_JSONL" \
          --raw_out_jsonl "$RAW_OUT_JSONL" \
          --failed_ids_out "$FAILED_IDS_OUT" \
          --start_idx "$START_IDX" \
          --end_idx "$END_IDX" \
          "${overwrite_args[@]}"
        ;;

      consume_from_tsv)
        if [[ -z "${TSV_PATH:-}" ]]; then
          echo "Usage: TSV_PATH=/path/to/batches.tsv bash run_group_question_batch.sh consume_from_tsv" >&2
          exit 2
        fi
        overwrite_args=()
        if [[ "$OVERWRITE_OUT" == "1" ]]; then overwrite_args+=(--overwrite); fi
        python ./group_question_batch.py consume_from_tsv \
          --tsv_path "$TSV_PATH" \
          --src "$SRC_JSONL" \
          --out_jsonl "$OUT_JSONL" \
          --raw_out_dir "$RAW_OUT_DIR" \
          --failed_out_dir "$FAILED_OUT_DIR" \
          --start_idx "$START_IDX" \
          --end_idx "$END_IDX" \
          "${overwrite_args[@]}"
        ;;

      *)
        echo "Unknown CMD=$CMD" >&2
        exit 2
        ;;
    esac
  ' "$@"

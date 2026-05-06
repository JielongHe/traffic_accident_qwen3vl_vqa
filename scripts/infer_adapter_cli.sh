#!/usr/bin/env bash
set -euo pipefail

: "${ADAPTER:?Please set ADAPTER to your LoRA checkpoint directory, e.g. output/.../checkpoint-xxx}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
TEST="${TEST:-data/swift/test.jsonl}"
OUT="${OUT:-outputs/test_pred_swift.jsonl}"

mkdir -p "$(dirname "$OUT")"

swift infer \
  --model "$MODEL" \
  --adapters "$ADAPTER" \
  --val_dataset "$TEST" \
  --result_path "$OUT" \
  --infer_backend pt \
  --max_new_tokens 2048 \
  --stream false

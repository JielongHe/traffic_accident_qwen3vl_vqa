#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-768}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-96}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
TRAIN="${TRAIN:-data/swift/train.jsonl}"
VAL="${VAL:-data/swift/val.jsonl}"
OUT="${OUT:-output/qwen3vl-2b-accident-lora}"

swift sft \
  --model "$MODEL" \
  --dataset "$TRAIN" \
  --val_dataset "$VAL" \
  --load_from_cache_file true \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 4 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --attn_impl flash_attn \
  --padding_free true \
  --learning_rate 8e-5 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_vit true \
  --freeze_aligner true \
  --packing false \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing false \
  --gradient_accumulation_steps 16 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 3 \
  --logging_steps 5 \
  --max_length 4096 \
  --output_dir "$OUT" \
  --warmup_ratio 0.05 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4

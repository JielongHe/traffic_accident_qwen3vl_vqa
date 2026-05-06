# Traffic Accident Video Report-only Fine-tuning with Qwen3-VL and ms-swift

This version fine-tunes Qwen3-VL to output **only `report_enriched` text** for each traffic accident or near-miss video.

Important:

- Assistant target = `report_enriched` only.
- `video_summary` is ignored.
- No JSON output target.
- No `event_type`, `visible_attributes`, or `extended_attributes` output target.

## Data

Converted files:

```text
data/swift/train.jsonl
data/swift/val.jsonl
data/swift/test.jsonl
data/swift/test_infer.jsonl
data/swift/test_ref.jsonl
```

Each training example is:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<video>\nGenerate only the enriched traffic accident analysis report..."},
    {"role": "assistant", "content": "At a signalized intersection ..."}
  ],
  "videos": ["data/crash/crash_000001.mp4"]
}
```

## Video layout

Place the MP4 files according to the original relative paths:

```text
traffic_accident_qwen3vl_vqa/
  data/
    crash/
      crash_000001.mp4
    nearmiss/
      nearmiss_000001.mp4
```

## Environment

```bash
conda create -n qwen3vl_accident python=3.10 -y
conda activate qwen3vl_accident

python -m pip install -U pip setuptools wheel
python -m pip install -U modelscope accelerate peft deepspeed
python -m pip install -U "ms-swift>=4.0" "transformers>=4.57" "qwen_vl_utils>=0.0.14"
python -m pip install -U torchcodec decord
python -m pip install "flash-attn==2.8.3" --no-build-isolation
```

## Regenerate data

```bash
python scripts/prepare_swift_data.py --input_dir data/raw --output_dir data/swift
```

Validate:

```bash
python scripts/check_swift_data.py --data data/swift/train.jsonl
python scripts/check_swift_data.py --data data/swift/val.jsonl
python scripts/check_swift_data.py --data data/swift/test.jsonl
python scripts/check_swift_data.py --data data/swift/train.jsonl --check_video
```

## Train

Recommended for about 1,000 samples: LoRA SFT, freeze vision encoder and aligner.

Two-GPU 4B:

```bash
bash scripts/train_lora_4b_2gpu.sh
```

Single-GPU 2B:

```bash
bash scripts/train_lora_2b_1gpu.sh
```

## Inference

```bash
python scripts/infer_batch_keep_id.py   --model Qwen/Qwen3-VL-4B-Instruct   --adapter output/qwen3vl-4b-accident-lora/vx-xxxx/checkpoint-xxx   --data data/swift/test_infer.jsonl   --out outputs/test_pred.jsonl   --batch_size 1
```

## Evaluate

```bash
python scripts/evaluate_predictions.py   --pred outputs/test_pred.jsonl   --ref data/swift/test_ref.jsonl   --out outputs/metrics.json
```

Useful metrics:

- `non_empty_rate`
- `json_like_output_rate`, should be near 0
- `report_rouge_l`
- `report_token_f1`

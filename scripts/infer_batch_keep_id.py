#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def batched(xs, n):
    batch = []
    for x in xs:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter", default="", help="LoRA checkpoint path. Leave empty for base model.")
    parser.add_argument("--data", default="data/swift/test_infer.jsonl")
    parser.add_argument("--out", default="outputs/test_pred.jsonl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--fps_max_frames", type=int, default=int(os.environ.get("FPS_MAX_FRAMES", "16")))
    parser.add_argument("--video_max_token_num", type=int, default=int(os.environ.get("VIDEO_MAX_TOKEN_NUM", "128")))
    parser.add_argument("--image_max_token_num", type=int, default=int(os.environ.get("IMAGE_MAX_TOKEN_NUM", "1024")))
    parser.add_argument("--cuda_visible_devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["FPS_MAX_FRAMES"] = str(args.fps_max_frames)
    os.environ["VIDEO_MAX_TOKEN_NUM"] = str(args.video_max_token_num)
    os.environ["IMAGE_MAX_TOKEN_NUM"] = str(args.image_max_token_num)

    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig

    adapters = [args.adapter] if args.adapter else None
    engine = TransformersEngine(
        args.model,
        adapters=adapters,
        max_batch_size=args.batch_size,
        attn_impl="flash_attn",
    )
    request_config = RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    rows = list(read_jsonl(Path(args.data)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with Path(args.out).open("w", encoding="utf-8") as fout:
        for batch in batched(rows, args.batch_size):
            requests = [
                InferRequest(messages=r["messages"], videos=r.get("videos", []))
                for r in batch
            ]
            responses = engine.infer(requests, request_config=request_config, use_tqdm=False)
            for r, resp in zip(batch, responses):
                content = resp.choices[0].message.content
                fout.write(json.dumps({
                    "id": r.get("id"),
                    "video_path": r.get("source_video_path") or (r.get("videos") or [None])[0],
                    "prediction": content,
                }, ensure_ascii=False) + "\n")
                done += 1
                if done % 20 == 0:
                    print(f"[OK] inferred {done}/{len(rows)}")
    print(f"[OK] wrote {args.out}")

if __name__ == "__main__":
    main()

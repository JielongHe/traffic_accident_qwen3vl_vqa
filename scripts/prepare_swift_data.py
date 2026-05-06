#!/usr/bin/env python3
import argparse, json
from pathlib import Path

SYSTEM_PROMPT = 'You are a traffic accident video semantic analysis assistant.\nYour task is to analyze dashcam or road-scene videos and produce a factual enriched traffic incident report.\nRules:\n1. Use visual evidence from the video and do not invent unsupported details.\n2. Output the report text only.\n3. Do not output JSON, markdown, bullet points, field names, labels, or explanations.\n4. Do not output video_summary. The target is report_enriched.\n5. The report should cover scene, participants, motions, trigger/cause, crash or near-miss outcome, and uncertainty when details are not visible.\n'
USER_PROMPT = '<video>\nGenerate only the enriched traffic accident analysis report for this video.\n\nRequirements:\n- Output one report paragraph only.\n- Do not output JSON.\n- Do not output field names, bullet points, markdown, or extra explanation.\n- Do not output video_summary.\n- Describe road scene, lighting/weather/traffic context when visible, ego motion, other participant motion, trigger/cause, impact or avoidance process, outcome, and uncertainty if details are unclear.\n'

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def make_video_path(video_path, video_root):
    if not video_root:
        return video_path
    p = Path(video_path)
    return str(p if p.is_absolute() else Path(video_root) / p)

def build_answer(r):
    if "report_enriched" not in r:
        raise ValueError(f"record {r.get('id')} has no report_enriched")
    return (r.get("report_enriched") or "").strip()

def make_row(r, video_root="", include_answer=True, keep_id=False):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": build_answer(r)})
    out = {"messages": messages, "videos": [make_video_path(r["video_path"], video_root)]}
    if keep_id:
        out["id"] = r.get("id")
        out["source_video_path"] = r.get("video_path")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/raw")
    ap.add_argument("--output_dir", default="data/swift")
    ap.add_argument("--video_root", default="")
    args = ap.parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    for split in ["train", "val", "test"]:
        rows = list(read_jsonl(input_dir / f"{split}.jsonl"))
        write_jsonl(output_dir / f"{split}.jsonl", [make_row(r, args.video_root, True, False) for r in rows])
        print(f"[OK] wrote {output_dir / f'{split}.jsonl'}: {len(rows)} rows")
    test_rows = list(read_jsonl(input_dir / "test.jsonl"))
    write_jsonl(output_dir / "test_infer.jsonl", [make_row(r, args.video_root, False, True) for r in test_rows])
    write_jsonl(output_dir / "test_ref.jsonl", [
        {"id": r.get("id"), "video_path": make_video_path(r.get("video_path"), args.video_root), "reference": build_answer(r)}
        for r in test_rows
    ])
    print("[OK] assistant target is report_enriched text only; video_summary was ignored.")

if __name__ == "__main__":
    main()

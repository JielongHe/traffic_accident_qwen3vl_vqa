#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from collections import Counter

def json_like(text):
    text = (text or "").strip()
    return text.startswith("{") or text.startswith("[") or '"report"' in text or '"event_type"' in text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--check_video", action="store_true")
    args = ap.parse_args()

    n, empty, json_outputs, forbidden, missing_video = 0, 0, 0, 0, 0
    roles = Counter()
    errors = []

    with open(args.data, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                errors.append(f"line {line_no}: invalid JSONL row: {e}")
                continue

            messages = obj.get("messages")
            videos = obj.get("videos")
            if not isinstance(messages, list) or len(messages) < 2:
                errors.append(f"line {line_no}: messages must contain at least system and user")
                continue

            for m in messages:
                roles[m.get("role")] += 1

            user_text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
            if "<video>" not in user_text:
                errors.append(f"line {line_no}: user prompt missing <video>")

            if not isinstance(videos, list) or len(videos) != 1:
                errors.append(f"line {line_no}: videos must be a list with exactly one path")
            elif args.check_video and not Path(videos[0]).exists():
                missing_video += 1

            assistant = [m for m in messages if m.get("role") == "assistant"]
            if assistant:
                content = assistant[-1].get("content", "")
                empty += int(not content.strip())
                json_outputs += int(json_like(content))
                forbidden += int("video_summary" in content)

    print(f"rows={n}")
    print(f"roles={dict(roles)}")
    print(f"empty_reports={empty}")
    print(f"json_like_assistant_targets={json_outputs}")
    print(f"contains_video_summary={forbidden}")
    if args.check_video:
        print(f"missing_video_files={missing_video}")

    if errors or empty or json_outputs or forbidden:
        print("ERRORS:")
        for e in errors[:50]:
            print("-", e)
        raise SystemExit(1)
    print("[OK] report-only SFT dataset looks valid")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from collections import Counter

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def clean_report(x):
    text = str(x or "").strip()
    text = re.sub(r"^```(?:json|text)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "report" in obj:
                return str(obj["report"]).strip()
        except Exception:
            pass
    return text

def tok(s):
    return re.findall(r"[A-Za-z0-9_]+", str(s or "").lower())

def lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j-1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]

def rouge_l(pred, ref):
    p, r = tok(pred), tok(ref)
    if not p or not r:
        return 0.0
    l = lcs_len(p, r)
    precision, recall = l / len(p), l / len(r)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

def token_f1(pred, ref):
    p, r = tok(pred), tok(ref)
    if not p or not r:
        return 0.0
    cp, cr = Counter(p), Counter(r)
    overlap = sum((cp & cr).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    refs = {r["id"]: clean_report(r["reference"]) for r in read_jsonl(args.ref)}
    preds = list(read_jsonl(args.pred))

    total = non_empty = json_like = 0
    rouge_scores, f1_scores, failures = [], [], []

    for row in preds:
        rid = row.get("id")
        if rid not in refs:
            failures.append({"id": rid, "error": "missing_reference"})
            continue
        raw = row.get("prediction") or row.get("response") or row.get("answer") or row.get("content") or ""
        pred, ref = clean_report(raw), refs[rid]
        total += 1
        non_empty += int(bool(pred.strip()))
        json_like += int(str(raw).strip().startswith("{") or '"report"' in str(raw))
        rouge_scores.append(rouge_l(pred, ref))
        f1_scores.append(token_f1(pred, ref))

    metrics = {
        "total": total,
        "non_empty_rate": non_empty / total if total else 0,
        "json_like_output_rate": json_like / total if total else 0,
        "report_rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0,
        "report_token_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
        "num_failures": len(failures),
        "failure_examples": failures[:20],
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()

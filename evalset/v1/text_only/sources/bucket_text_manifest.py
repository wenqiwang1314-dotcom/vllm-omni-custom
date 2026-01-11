#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Any, List, Tuple

from transformers import AutoTokenizer


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_bands(bands_str: str) -> List[Tuple[int, int]]:
    # "80-150,150-300,300-600"
    bands: List[Tuple[int, int]] = []
    for part in bands_str.split(","):
        part = part.strip()
        if not part:
            continue
        lo, hi = part.split("-", 1)
        bands.append((int(lo.strip()), int(hi.strip())))
    if not bands:
        raise ValueError("bands is empty")
    return bands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="Input jsonl (cleaned)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B", help="HF model name for tokenizer")
    ap.add_argument("--out_dir", required=True, help="Output directory for bucket manifests")
    ap.add_argument("--buckets", default="128,256,512", help="Comma-separated token buckets")
    ap.add_argument("--bands", default="80-150,150-300,300-600",
                    help="Comma-separated word bands aligned with buckets order")
    ap.add_argument("--min_chars", type=int, default=20, help="Skip outputs if decoded text too short")
    ap.add_argument("--require_full_bucket", action="store_true",
                    help="Require original_tokens >= bucket; otherwise skip (ensures bucket difference is real)")
    args = ap.parse_args()

    buckets = [int(x.strip()) for x in args.buckets.split(",") if x.strip()]
    buckets = sorted(set(buckets))

    bands = parse_bands(args.bands)
    if len(bands) != len(buckets):
        raise ValueError(f"--bands count ({len(bands)}) must match bucket count ({len(buckets)}). "
                         f"Got buckets={buckets}, bands={bands}")

    bucket2band = {b: bands[i] for i, b in enumerate(buckets)}

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    data = read_jsonl(args.in_jsonl)

    bucket_out: Dict[int, List[Dict[str, Any]]] = {b: [] for b in buckets}

    for item in data:
        prompt = item.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            continue

        # 词数（优先用 meta.word_count）
        meta = item.get("meta", {}) if isinstance(item.get("meta", {}), dict) else {}
        word_count = int(meta.get("word_count", len(prompt.split())))

        # token 数
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        total_tokens = len(ids)

        for b in buckets:
            lo, hi = bucket2band[b]

            # 按档位筛选：word_count 必须落在对应区间
            if not (lo <= word_count <= hi):
                continue

            # 强制“够长再截断”，否则 128/256/512 会趋同
            if args.require_full_bucket and total_tokens < b:
                continue

            cut_ids = ids[:b] if total_tokens > b else ids
            cut_text = tokenizer.decode(cut_ids, skip_special_tokens=True).strip()
            if len(cut_text) < args.min_chars:
                continue

            out = {
                "id": f"{item.get('id','unknown')}_t{b}",
                "modality": "text",
                "prompt": cut_text,
                "meta": {
                    **meta,
                    "source_id": item.get("id", "unknown"),
                    "target_tokens": b,
                    "actual_tokens": len(cut_ids),
                    "original_tokens": total_tokens,
                    "word_count": word_count,
                    "word_band": [lo, hi],
                    "bucketed": True,
                },
            }
            bucket_out[b].append(out)

    os.makedirs(args.out_dir, exist_ok=True)
    for b in buckets:
        out_path = os.path.join(args.out_dir, f"pdf_clean_{b}.jsonl")
        write_jsonl(out_path, bucket_out[b])
        print(f"[OK] bucket={b} band={bucket2band[b]} -> {out_path} (n={len(bucket_out[b])})")

    # 如果某个 bucket 产出为 0，说明 band + require_full_bucket 过严，需要：
    # - 增加 build_text_from_pdfs 的 per_pdf
    # - 或扩大对应 band 的上界
    # - 或先不启用 require_full_bucket（不推荐，因为会导致 bucket 差异不明显）


if __name__ == "__main__":
    main()

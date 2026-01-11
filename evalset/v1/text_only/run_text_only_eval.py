#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import requests


def now_iso():
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://127.0.0.1:8091/v1")
    ap.add_argument("--model", default="Qwen-Omni")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max_tokens", type=int, default=128)  # 生成长度，和输入bucket无关
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    url = args.base_url.rstrip("/") + "/chat/completions"

    session = requests.Session()

    with out_path.open("w", encoding="utf-8") as fout:
        for sample in read_jsonl(manifest_path):
            sid = sample["id"]
            prompt = sample["prompt"]

            for r in range(1, args.repeats + 1):
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                }

                t0 = time.perf_counter()
                start = now_iso()
                try:
                    resp = session.post(url, json=payload, timeout=args.timeout)
                    dt = time.perf_counter() - t0
                    end = now_iso()

                    ok = resp.status_code == 200
                    j = resp.json() if ok else None

                    out_text = ""
                    usage = {}
                    if ok and j:
                        out_text = j["choices"][0]["message"].get("content", "") or ""
                        usage = j.get("usage", {}) or {}

                    rec = {
                        "ts_start": start,
                        "ts_end": end,
                        "latency_s": dt,
                        "status_code": resp.status_code,
                        "ok": ok,
                        "sample_id": sid,
                        "repeat": r,
                        "bucket_target_tokens": sample.get("meta", {}).get("target_tokens"),
                        "input_chars": len(prompt),
                        "output_chars": len(out_text),
                        "usage": usage,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()

                    print(f"[{sid}] rep={r} status={resp.status_code} latency={dt:.3f}s")

                except Exception as e:
                    dt = time.perf_counter() - t0
                    end = now_iso()
                    rec = {
                        "ts_start": start,
                        "ts_end": end,
                        "latency_s": dt,
                        "status_code": None,
                        "ok": False,
                        "sample_id": sid,
                        "repeat": r,
                        "error": repr(e),
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    print(f"[{sid}] rep={r} ERROR latency={dt:.3f}s {e}")


if __name__ == "__main__":
    main()

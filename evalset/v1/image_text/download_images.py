#!/usr/bin/env python3
import argparse, hashlib, json, os, re, time
from pathlib import Path

import requests

def safe_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s.strip("_")[:180]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url: str, out_path: Path, timeout: int, retries: int, sleep: float) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if exists
    if out_path.exists() and out_path.stat().st_size > 0:
        return {"status": "skipped_exists", "bytes": out_path.stat().st_size}

    last_err = None
    for _ in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
                if r.status_code != 200:
                    return {"status": "http_error", "http_status": r.status_code}
                ct = r.headers.get("Content-Type", "")
                if ct and (not ct.startswith("image/")):
                    return {"status": "bad_content_type", "content_type": ct}
                tmp = out_path.with_suffix(out_path.suffix + ".part")
                total = 0
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                            total += len(chunk)
                tmp.replace(out_path)
                return {"status": "ok", "bytes": total, "content_type": ct}
        except Exception as e:
            last_err = str(e)
            time.sleep(sleep)
    return {"status": "exception", "error": last_err}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", default="coco_eval.jsonl")
    ap.add_argument("--out_dir", default="images")
    ap.add_argument("--out_jsonl", default="coco_eval_local.jsonl")
    ap.add_argument("--report", default="reports/download_report.json")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    results = []
    ok = skip = fail = 0

    with in_path.open("r", encoding="utf-8") as fin, Path(args.out_jsonl).open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            item_id = item.get("id") or f"item{i}"
            coco_id = item.get("coco_id")
            url = (item.get("image") or {}).get("original_url") or item.get("image_url") or item.get("url")

            rec = {"id": item_id, "coco_id": coco_id, "url": url}
            if not url:
                rec["status"] = "no_url"
                results.append(rec)
                fail += 1
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue

            # filename strategy
            fname = safe_name(f"{item_id}_coco{coco_id if coco_id is not None else 'na'}.jpg")
            path = out_dir / fname

            r = download(url, path, args.timeout, args.retries, args.sleep)
            rec.update(r)
            rec["path"] = str(path)

            if rec["status"] in ("ok", "skipped_exists"):
                rec["filesize"] = path.stat().st_size
                rec["sha256"] = sha256_file(path)
                item["local_path"] = str(path)
                ok += 1 if rec["status"] == "ok" else 0
                skip += 1 if rec["status"] == "skipped_exists" else 0
            else:
                fail += 1

            results.append(rec)
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    report = {
        "input": str(in_path),
        "output_jsonl": args.out_jsonl,
        "out_dir": str(out_dir),
        "summary": {"ok": ok, "skipped": skip, "failed": fail, "total": ok + skip + fail},
        "results": results,
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["summary"], ensure_ascii=False))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_eval.py
Batch image->text (captioning) evaluation for vLLM-Omni OpenAI-compatible server.

Default expects:
  - input jsonl: coco_eval_local.jsonl (each line has: id, local_path, captions, image.original_url, etc.)
  - images stored under: ./images/...
  - vLLM server: http://127.0.0.1:8091, model name: Qwen-Omni
Outputs:
  - results/preds.jsonl
  - results/run_summary.json
  - logs/run_eval_YYYYmmdd_HHMMSS.log

Usage:
  python3 run_eval.py
  python3 run_eval.py --input coco_eval.jsonl
  python3 run_eval.py --max-samples 50
  python3 run_eval.py --prompt "Describe this image in one concise sentence."
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


# -----------------------------
# Utilities
# -----------------------------

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_http_server(root_dir: Path, host: str, port: int, log_fp) -> Tuple[str, Optional[subprocess.Popen]]:
    """
    Ensure images are accessible via http://host:port/<path>.
    If port is already open, assume server is running.
    Otherwise start python http.server in background.
    Returns (base_url, process_or_none).
    """
    base_url = f"http://{host}:{port}/"
    if is_port_open(host, port):
        log_fp.write(f"[INFO] Static file server already running at {base_url}\n")
        log_fp.flush()
        return base_url, None

    log_fp.write(f"[INFO] Starting static file server: python3 -m http.server {port} (root={root_dir})\n")
    log_fp.flush()

    # Start server in root_dir
    # Redirect stdout/stderr to log file
    proc = subprocess.Popen(
        ["python3", "-m", "http.server", str(port), "--bind", host],
        cwd=str(root_dir),
        stdout=log_fp,
        stderr=log_fp,
    )

    # Wait briefly for server to come up
    for _ in range(30):
        if is_port_open(host, port):
            log_fp.write(f"[INFO] Static file server started at {base_url}\n")
            log_fp.flush()
            return base_url, proc
        time.sleep(0.1)

    # If still not up, terminate and fail
    try:
        proc.terminate()
    except Exception:
        pass
    raise RuntimeError(f"Failed to start http.server on {host}:{port}")


def safe_get(item: Dict[str, Any], path: str, default=None):
    """
    safe_get(d, "image.original_url") -> value
    """
    cur = item
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_jsonl(fp: Path):
    with fp.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ln, json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {ln} of {fp}: {e}")


def write_jsonl_line(fp, obj: Dict[str, Any]):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()


# -----------------------------
# vLLM OpenAI-compatible client
# -----------------------------

def make_payload(model: str, prompt: str, image_url: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    # OpenAI-compatible multimodal content (text + image_url)
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def call_chat_completions(
    api_base: str,
    payload: Dict[str, Any],
    timeout_s: int,
    session: requests.Session,
) -> str:
    url = api_base.rstrip("/") + "/v1/chat/completions"
    r = session.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    # Typical structure: choices[0].message.content
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected response schema: {e}; response={data}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="coco_eval_local.jsonl", help="Input JSONL file (default: coco_eval_local.jsonl)")
    parser.add_argument("--output", default="results/preds.jsonl", help="Output JSONL predictions file")
    parser.add_argument("--api-base", default="http://127.0.0.1:8091", help="vLLM OpenAI-compatible server base URL")
    parser.add_argument("--model", default="Qwen-Omni", help="served-model-name used in vllm serve")
    parser.add_argument("--prompt", default="Describe the image in one concise sentence.", help="Caption prompt")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens for caption output")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP request timeout seconds")
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep between requests (seconds)")
    parser.add_argument("--max-samples", type=int, default=0, help="If >0, only run this many samples")
    parser.add_argument("--image-host", default="127.0.0.1", help="Host for local static file server")
    parser.add_argument("--image-port", type=int, default=8000, help="Port for local static file server")
    parser.add_argument("--images-root", default=".", help="Root dir to serve files from (default current dir)")
    args = parser.parse_args()

    root = Path(args.images_root).resolve()
    in_fp = (root / args.input).resolve()
    out_fp = (root / args.output).resolve()
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"run_eval_{now_str()}.log"

    results_summary_path = root / "results" / "run_summary.json"

    if not in_fp.exists():
        print(f"[FATAL] Input file not found: {in_fp}", file=sys.stderr)
        sys.exit(2)

    # Open log file
    with log_path.open("a", encoding="utf-8") as log_fp:
        log_fp.write(f"[INFO] run_eval start: {datetime.now().isoformat()}\n")
        log_fp.write(f"[INFO] input={in_fp}\n")
        log_fp.write(f"[INFO] output={out_fp}\n")
        log_fp.write(f"[INFO] api_base={args.api_base}\n")
        log_fp.write(f"[INFO] model={args.model}\n")
        log_fp.write(f"[INFO] prompt={args.prompt}\n")
        log_fp.flush()

        # Basic server reachability check
        try:
            health_url = args.api_base.rstrip("/") + "/v1/models"
            r = requests.get(health_url, timeout=10)
            if r.status_code >= 400:
                log_fp.write(f"[WARN] GET {health_url} returned {r.status_code}\n")
            else:
                log_fp.write(f"[INFO] Server reachable: {health_url}\n")
            log_fp.flush()
        except Exception as e:
            log_fp.write(f"[WARN] Could not reach server at {args.api_base}: {e}\n")
            log_fp.write("[WARN] Will still attempt requests; ensure vllm serve is running.\n")
            log_fp.flush()

        # Start or reuse static server for local images
        static_proc = None
        try:
            image_base_url, static_proc = ensure_http_server(
                root_dir=root,
                host=args.image_host,
                port=args.image_port,
                log_fp=log_fp,
            )

            # Prepare output (append mode to allow resume)
            # We'll also build a set of already-done IDs to skip duplicates
            done_ids = set()
            if out_fp.exists():
                try:
                    with out_fp.open("r", encoding="utf-8") as rf:
                        for line in rf:
                            line = line.strip()
                            if not line:
                                continue
                            obj = json.loads(line)
                            if "id" in obj:
                                done_ids.add(obj["id"])
                    log_fp.write(f"[INFO] Resume mode: found {len(done_ids)} completed ids in {out_fp}\n")
                    log_fp.flush()
                except Exception:
                    log_fp.write("[WARN] Could not parse existing output for resume; will append anyway.\n")
                    log_fp.flush()

            session = requests.Session()

            total = 0
            ok = 0
            fail = 0
            error_types: Dict[str, int] = {}
            t0 = time.time()

            with out_fp.open("a", encoding="utf-8") as wf:
                for ln, item in load_jsonl(in_fp):
                    if args.max_samples and total >= args.max_samples:
                        break

                    sid = item.get("id") or f"line_{ln}"
                    if sid in done_ids:
                        continue

                    local_path = item.get("local_path")
                    if not local_path:
                        # Fallback: try to infer from id if your downloader uses a standard naming
                        local_path = f"images/{sid}.jpg"

                    img_file = (root / local_path).resolve()
                    img_url = image_base_url + local_path.replace("\\", "/")

                    record: Dict[str, Any] = {
                        "id": sid,
                        "input_line": ln,
                        "local_path": local_path,
                        "image_url": img_url,
                        "coco_id": item.get("coco_id"),
                        "explore_url": item.get("explore_url"),
                        "original_url": safe_get(item, "image.original_url"),
                        "prompt": args.prompt,
                        "ts": datetime.now().isoformat(),
                    }

                    total += 1

                    # Check local image exists; if not, still try original_url if available
                    if not img_file.exists():
                        orig_url = safe_get(item, "image.original_url")
                        if orig_url:
                            record["image_url"] = orig_url
                            record["note"] = "local image missing; using original_url"
                            img_url = orig_url
                        else:
                            record["error"] = f"local image missing and no original_url (expected {img_file})"
                            write_jsonl_line(wf, record)
                            fail += 1
                            error_types["missing_image"] = error_types.get("missing_image", 0) + 1
                            continue

                    payload = make_payload(
                        model=args.model,
                        prompt=args.prompt,
                        image_url=img_url,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )

                    try:
                        pred = call_chat_completions(
                            api_base=args.api_base,
                            payload=payload,
                            timeout_s=args.timeout,
                            session=session,
                        )
                        record["pred"] = pred
                        # Keep ground-truth captions for later offline scoring
                        if "captions" in item:
                            record["captions"] = item["captions"]

                        write_jsonl_line(wf, record)
                        ok += 1
                        log_fp.write(f"[OK] {sid}: {pred}\n")
                        log_fp.flush()

                    except requests.HTTPError as e:
                        record["error"] = f"HTTPError: {str(e)}"
                        try:
                            record["http_status"] = e.response.status_code if e.response is not None else None
                            record["http_body"] = e.response.text[:2000] if e.response is not None else None
                        except Exception:
                            pass
                        write_jsonl_line(wf, record)
                        fail += 1
                        k = f"http_{record.get('http_status','unknown')}"
                        error_types[k] = error_types.get(k, 0) + 1
                        log_fp.write(f"[ERR] {sid}: {record['error']}\n")
                        log_fp.flush()

                    except requests.Timeout:
                        record["error"] = "Timeout"
                        write_jsonl_line(wf, record)
                        fail += 1
                        error_types["timeout"] = error_types.get("timeout", 0) + 1
                        log_fp.write(f"[ERR] {sid}: Timeout\n")
                        log_fp.flush()

                    except Exception as e:
                        record["error"] = f"{type(e).__name__}: {str(e)}"
                        write_jsonl_line(wf, record)
                        fail += 1
                        error_types[type(e).__name__] = error_types.get(type(e).__name__, 0) + 1
                        log_fp.write(f"[ERR] {sid}: {record['error']}\n")
                        log_fp.flush()

                    if args.sleep:
                        time.sleep(args.sleep)

            dt = time.time() - t0
            summary = {
                "started_at": datetime.fromtimestamp(t0).isoformat(),
                "finished_at": datetime.now().isoformat(),
                "input": str(in_fp),
                "output": str(out_fp),
                "api_base": args.api_base,
                "model": args.model,
                "prompt": args.prompt,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "timeout": args.timeout,
                "sleep": args.sleep,
                "max_samples": args.max_samples,
                "counts": {"total_attempted": total, "ok": ok, "fail": fail},
                "error_types": error_types,
                "elapsed_sec": dt,
                "throughput_img_per_sec": (ok / dt) if dt > 0 else None,
            }

            results_summary_path.parent.mkdir(parents=True, exist_ok=True)
            with results_summary_path.open("w", encoding="utf-8") as sf:
                json.dump(summary, sf, ensure_ascii=False, indent=2)

            log_fp.write(f"[INFO] Summary written: {results_summary_path}\n")
            log_fp.write(f"[INFO] Done. total={total}, ok={ok}, fail={fail}, elapsed={dt:.2f}s\n")
            log_fp.flush()

        finally:
            # Stop local http server if we started it
            if static_proc is not None:
                try:
                    log_fp.write("[INFO] Stopping static file server...\n")
                    log_fp.flush()
                    static_proc.terminate()
                    static_proc.wait(timeout=3)
                except Exception:
                    try:
                        static_proc.kill()
                    except Exception:
                        pass


if __name__ == "__main__":
    main()

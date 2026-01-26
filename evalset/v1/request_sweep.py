#!/usr/bin/env python3
import argparse
import json
import time
import uuid
from typing import Any, Dict, Optional

import requests


def now_s() -> float:
    return time.perf_counter()


def make_rid() -> str:
    # 短一点，日志里更好看
    return uuid.uuid4().hex[:12]


def post_chat_completion(
    url: str,
    model: str,
    prompt: str,
    rid: str,
    max_tokens: int,
    timeout_s: int = 600,
) -> Dict[str, Any]:
    """
    OpenAI-compatible: /v1/chat/completions
    把 rid 同时放在:
      - HTTP header X-Request-Id (如果你服务端有接)
      - prompt 里 (万一你用 prompt 做 fallback)
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"[RID:{rid}] {prompt}"},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": rid,
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def safe_get_usage_tokens(resp_json: Dict[str, Any]) -> Optional[int]:
    usage = resp_json.get("usage")
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if isinstance(total, int):
            return total
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8091")
    ap.add_argument("--model", required=True, help="served model name, e.g. Qwen-Omni")
    ap.add_argument("--percent", type=int, required=True, help="VLLM_GREEN_SM_PERCENT for this run")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    total = args.warmup + args.trials
    for idx in range(total):
        rid = make_rid()
        is_warmup = idx < args.warmup

        t0 = now_s()
        ok = True
        err = ""
        status_code = None
        usage_tokens = None

        try:
            r = post_chat_completion(
                url=url,
                model=args.model,
                prompt=args.prompt,
                rid=rid,
                max_tokens=args.max_tokens,
            )
            usage_tokens = safe_get_usage_tokens(r)
        except requests.HTTPError as e:
            ok = False
            err = f"HTTPError: {e}"
            if e.response is not None:
                status_code = e.response.status_code
        except Exception as e:
            ok = False
            err = f"Exception: {e}"

        t1 = now_s()
        rec = {
            "rid": rid,
            "percent": args.percent,
            "is_warmup": is_warmup,
            "t_start_perf": t0,
            "t_end_perf": t1,
            "latency_s": (t1 - t0),
            "ok": ok,
            "status_code": status_code,
            "usage_total_tokens": usage_tokens,
            "url": url,
        }

        with open(args.out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 控制台简报
        tag = "WARMUP" if is_warmup else "TRIAL"
        if ok:
            print(f"[{tag}] pct={args.percent:3d} rid={rid} latency={rec['latency_s']:.4f}s tokens={usage_tokens}")
        else:
            print(f"[{tag}] pct={args.percent:3d} rid={rid} FAILED latency={rec['latency_s']:.4f}s err={err}")


if __name__ == "__main__":
    main()


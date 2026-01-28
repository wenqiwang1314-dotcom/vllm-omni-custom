#!/usr/bin/env python3
import argparse
import base64
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


def now_s() -> float:
    return time.perf_counter()


def make_rid() -> str:
    return uuid.uuid4().hex[:12]


def file_to_data_url(img_path: str) -> str:
    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {img_path}")
    suffix = p.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def post_chat_completion_mm(
    url: str,
    model: str,
    prompt: str,
    image_url_or_dataurl: str,
    rid: str,
    max_tokens: int,
    temperature: float = 0.2,
    timeout_s: int = 600,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"[RID:{rid}] {prompt}"},
                    {"type": "image_url", "image_url": {"url": image_url_or_dataurl}},
                ],
            }
        ],
        "temperature": temperature,
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


def safe_get_usage(resp_json: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    usage = resp_json.get("usage")
    if not isinstance(usage, dict):
        return None, None, None
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    tt = usage.get("total_tokens")
    return (pt if isinstance(pt, int) else None,
            ct if isinstance(ct, int) else None,
            tt if isinstance(tt, int) else None)


def safe_get_text(resp_json: Dict[str, Any]) -> Optional[str]:
    """
    Robustly extract text output from OpenAI-compatible response
    even if audio is also returned.
    Strategy:
      1) iterate all choices
      2) accept message.content if it's a non-empty string
      3) if content is a list, join all text blocks
    """
    choices = resp_json.get("choices")
    if not isinstance(choices, list):
        return None

    best = None

    for ch in choices:
        msg = ch.get("message") if isinstance(ch, dict) else None
        if not isinstance(msg, dict):
            continue

        content = msg.get("content")

        # Case 1: plain string
        if isinstance(content, str):
            s = content.strip()
            if s:
                return s  # strongest signal: direct text

        # Case 2: structured content list
        if isinstance(content, list):
            parts = []
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") == "text" and isinstance(blk.get("text"), str):
                    t = blk["text"].strip()
                    if t:
                        parts.append(t)
            if parts:
                joined = "\n".join(parts)
                best = best or joined

    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8091")
    ap.add_argument("--model", required=True)
    ap.add_argument("--percent", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--prompt", default="Describe the image in one concise sentence.")

    # ✅ 支持二选一：image-url 或 image-path
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image-url", help="http(s) URL or data URL")
    g.add_argument("--image-path", help="local image file path")

    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    # ✅ 如果给的是本地文件，转成 data URL（避免 server 拉取 URL）
    if args.image_path:
        image_input = file_to_data_url(args.image_path)
        image_meta = args.image_path
    else:
        image_input = args.image_url
        image_meta = args.image_url

    total = args.warmup + args.trials
    for idx in range(total):
        rid = make_rid()
        is_warmup = idx < args.warmup

        t0 = now_s()
        ok = True
        err = ""
        status_code = None

        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        pred = None

        try:
            r = post_chat_completion_mm(
                url=url,
                model=args.model,
                prompt=args.prompt,
                image_url_or_dataurl=image_input,
                rid=rid,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            prompt_tokens, completion_tokens, total_tokens = safe_get_usage(r)
            pred = safe_get_text(r)

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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "pred": pred,
            "image": image_meta,
            "prompt": args.prompt,
            "url": url,
        }

        with open(args.out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        tag = "WARMUP" if is_warmup else "TRIAL"
        if ok:
            print(f"[{tag}] pct={args.percent:3d} rid={rid} latency={rec['latency_s']:.4f}s "
                  f"pt={prompt_tokens} ct={completion_tokens} tt={total_tokens}")
        else:
            print(f"[{tag}] pct={args.percent:3d} rid={rid} FAILED latency={rec['latency_s']:.4f}s err={err}")


if __name__ == "__main__":
    main()

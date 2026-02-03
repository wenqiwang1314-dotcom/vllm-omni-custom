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


def guess_audio_format_and_mime(p: Path) -> Tuple[str, str]:
    """
    Return (fmt, mime)
    fmt: wav/mp3/flac/ogg/m4a/aac...
    mime: audio/wav, audio/mpeg, audio/flac, ...
    """
    suf = p.suffix.lower().lstrip(".")
    if suf in ("wav", "wave"):
        return "wav", "audio/wav"
    if suf in ("mp3",):
        return "mp3", "audio/mpeg"
    if suf in ("flac",):
        return "flac", "audio/flac"
    if suf in ("ogg", "oga"):
        return "ogg", "audio/ogg"
    if suf in ("m4a",):
        # Some servers use audio/mp4 for m4a
        return "m4a", "audio/mp4"
    if suf in ("aac",):
        return "aac", "audio/aac"
    # fallback
    return suf or "wav", f"audio/{suf or 'wav'}"


def audio_file_to_data_url(audio_path: str) -> Tuple[str, str, str]:
    """
    Return (data_url, b64, fmt)
    - data_url: data:audio/...;base64,....
    - b64: raw base64 (no header)
    - fmt: wav/mp3/...
    """
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    fmt, mime = guess_audio_format_and_mime(p)
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"
    return data_url, b64, fmt


def post_chat_completion_audio(
    url: str,
    model: str,
    prompt: str,
    rid: str,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    audio_mode: str,
    audio_data_url: Optional[str] = None,
    audio_b64: Optional[str] = None,
    audio_format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    audio_mode:
      - "audio_url": send {"type":"audio_url","audio_url":{"url": data_url_or_http}}
      - "input_audio": send {"type":"input_audio","input_audio":{"data": b64, "format": "wav"}}
    """
    if audio_mode not in ("audio_url", "input_audio"):
        raise ValueError(f"invalid audio_mode: {audio_mode}")

    if audio_mode == "audio_url":
        if not audio_data_url:
            raise ValueError("audio_data_url required for audio_url mode")
        audio_blk = {"type": "audio_url", "audio_url": {"url": audio_data_url}}
    else:
        if not audio_b64 or not audio_format:
            raise ValueError("audio_b64 and audio_format required for input_audio mode")
        audio_blk = {"type": "input_audio", "input_audio": {"data": audio_b64, "format": audio_format}}

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"[RID:{rid}] {prompt}"},
                    audio_blk,
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
    return (
        pt if isinstance(pt, int) else None,
        ct if isinstance(ct, int) else None,
        tt if isinstance(tt, int) else None,
    )


def safe_get_text(resp_json: Dict[str, Any]) -> Optional[str]:
    """
    Robustly extract TEXT only (ignore any audio fields).
    Works whether message.content is str or list of blocks.
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
                return s

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
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompt", default="Identify the sound. Answer with one word.")
    ap.add_argument("--timeout-s", type=int, default=600)

    # 音频发送模式：默认 audio_url（data URL），不行再切 input_audio
    ap.add_argument("--audio-mode", choices=["audio_url", "input_audio"], default="audio_url")

    # ✅ 支持二选一：audio-path 或 audio-url
    #    但你说“都用本地路径”，那就用 --audio-path
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--audio-path", help="local audio file path (wav/mp3/flac/...)")
    g.add_argument("--audio-url", help="http(s) URL or data URL")

    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    audio_meta = None
    audio_data_url = None
    audio_b64 = None
    audio_fmt = None

    if args.audio_path:
        audio_data_url, audio_b64, audio_fmt = audio_file_to_data_url(args.audio_path)
        audio_meta = args.audio_path
    else:
        # 允许你直接传 data URL 或 http URL
        audio_data_url = args.audio_url
        audio_meta = args.audio_url
        # input_audio 模式要求 b64+format；如果你传的是 url，这里就不支持 input_audio
        if args.audio_mode == "input_audio":
            raise ValueError("--audio-mode input_audio requires --audio-path (local file)")

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
            r = post_chat_completion_audio(
                url=url,
                model=args.model,
                prompt=args.prompt,
                rid=rid,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_s=args.timeout_s,
                audio_mode=args.audio_mode,
                audio_data_url=audio_data_url,
                audio_b64=audio_b64,
                audio_format=audio_fmt,
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
            "pred": pred,              # ✅ 只记录文本
            "audio": audio_meta,       # 记录音频来源（本地路径或 URL）
            "audio_mode": args.audio_mode,
            "prompt": args.prompt,
            "url": url,
        }

        with open(args.out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        tag = "WARMUP" if is_warmup else "TRIAL"
        if ok:
            print(
                f"[{tag}] pct={args.percent:3d} rid={rid} latency={rec['latency_s']:.4f}s "
                f"pt={prompt_tokens} ct={completion_tokens} tt={total_tokens}"
            )
        else:
            print(
                f"[{tag}] pct={args.percent:3d} rid={rid} FAILED latency={rec['latency_s']:.4f}s err={err}"
            )


if __name__ == "__main__":
    main()

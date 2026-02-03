#!/usr/bin/env python3
"""
Reusable request sweep script for OpenAI-compatible /v1/chat/completions.

Features:
- Supports text-only, image, and audio inputs
- Local files -> data URL (default) or raw base64 (for input_audio)
- Keeps only TEXT output (pred_text) even if model returns audio blocks
- Writes JSONL records for warmup+trial with timings + usage + basic metadata
"""

import argparse
import base64
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests


# -----------------------
# Utilities
# -----------------------
def now_s() -> float:
    return time.perf_counter()


def make_rid() -> str:
    return uuid.uuid4().hex[:12]


def b64_of_file(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("utf-8")


def ensure_exists(path: str, kind: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")
    if not p.is_file():
        raise FileNotFoundError(f"{kind} is not a file: {path}")
    return p


# -----------------------
# MIME guessing
# -----------------------
def guess_image_mime(p: Path) -> Tuple[str, str]:
    suf = p.suffix.lower().lstrip(".")
    if suf in ("jpg", "jpeg"):
        return "jpeg", "image/jpeg"
    if suf in ("png",):
        return "png", "image/png"
    if suf in ("webp",):
        return "webp", "image/webp"
    raise ValueError(f"Unsupported image suffix: .{suf} (supported: jpg/jpeg/png/webp)")


def guess_audio_fmt_mime(p: Path) -> Tuple[str, str]:
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
        return "m4a", "audio/mp4"
    if suf in ("aac",):
        return "aac", "audio/aac"
    # fallback (may still work)
    return suf or "wav", f"audio/{suf or 'wav'}"


def file_to_data_url(path: str, kind: str) -> Tuple[str, str, str]:
    """
    Return (data_url, b64, fmt)
    - data_url: data:<mime>;base64,<...>
    - b64: raw base64 without prefix
    - fmt: format tag (jpeg/png/webp, wav/mp3/...)
    """
    p = ensure_exists(path, kind)
    if kind == "image":
        fmt, mime = guess_image_mime(p)
    elif kind == "audio":
        fmt, mime = guess_audio_fmt_mime(p)
    else:
        raise ValueError(f"invalid kind: {kind}")

    b64 = b64_of_file(p)
    data_url = f"data:{mime};base64,{b64}"
    return data_url, b64, fmt


# -----------------------
# Response parsing (TEXT only)
# -----------------------
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


def safe_get_text_only(resp_json: Dict[str, Any]) -> Optional[str]:
    choices = resp_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    # 1) Prefer choices[0] first (matches vLLM-Omni examples: text first, audio second)
    ordered = choices[:1] + choices[1:]

    def extract_from_choice(ch: Dict[str, Any]) -> Optional[str]:
        msg = ch.get("message") if isinstance(ch, dict) else None
        if not isinstance(msg, dict):
            return None

        # If this choice carries audio, it's likely not the pure text branch
        if "audio" in msg and msg["audio"] is not None:
            return None

        content = msg.get("content")
        if isinstance(content, str):
            s = content.strip()
            return s or None

        if isinstance(content, list):
            parts = []
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text" and isinstance(blk.get("text"), str):
                    t = blk["text"].strip()
                    if t:
                        parts.append(t)
            return "\n".join(parts).strip() or None

        return None

    # 2) Try ordered choices
    for ch in ordered:
        if isinstance(ch, dict):
            s = extract_from_choice(ch)
            if s:
                return s

    # 3) Fallback: nothing usable
    return None



# -----------------------
# Input spec (reusable)
# -----------------------
@dataclass
class InputSpec:
    mode: str  # "text" | "image" | "audio"
    # For image/audio: one of them is used depending on send_mode
    data_url: Optional[str] = None
    b64: Optional[str] = None
    fmt: Optional[str] = None
    meta: Optional[str] = None  # user-facing reference (path/url)


def build_input_spec(
    mode: str,
    image_path: Optional[str],
    image_url: Optional[str],
    audio_path: Optional[str],
    audio_url: Optional[str],
) -> InputSpec:
    if mode == "text":
        return InputSpec(mode="text", meta="(text-only)")

    if mode == "image":
        if image_path:
            data_url, b64, fmt = file_to_data_url(image_path, "image")
            return InputSpec(mode="image", data_url=data_url, b64=b64, fmt=fmt, meta=image_path)
        if image_url:
            # allow http(s) or data URL
            return InputSpec(mode="image", data_url=image_url, meta=image_url)
        raise ValueError("mode=image requires --image-path or --image-url")

    if mode == "audio":
        if audio_path:
            data_url, b64, fmt = file_to_data_url(audio_path, "audio")
            return InputSpec(mode="audio", data_url=data_url, b64=b64, fmt=fmt, meta=audio_path)
        if audio_url:
            return InputSpec(mode="audio", data_url=audio_url, meta=audio_url)
        raise ValueError("mode=audio requires --audio-path or --audio-url")

    raise ValueError(f"invalid mode: {mode}")


# -----------------------
# Request builder
# -----------------------
def build_messages(
    prompt: str,
    rid: str,
    inp: InputSpec,
    audio_send_mode: str,
) -> List[Dict[str, Any]]:
    """
    Build OpenAI-compatible messages array.
    For multimodal, we use user.content blocks.
    """
    blocks: List[Dict[str, Any]] = [{"type": "text", "text": f"[RID:{rid}] {prompt}"}]

    if inp.mode == "image":
        if not inp.data_url:
            raise ValueError("image InputSpec missing data_url")
        blocks.append({"type": "image_url", "image_url": {"url": inp.data_url}})

    if inp.mode == "audio":
        if audio_send_mode not in ("audio_url", "input_audio"):
            raise ValueError(f"invalid audio_send_mode: {audio_send_mode}")

        if audio_send_mode == "audio_url":
            if not inp.data_url:
                raise ValueError("audio_url mode requires inp.data_url (data URL or http URL)")
            blocks.append({"type": "audio_url", "audio_url": {"url": inp.data_url}})
        else:
            # input_audio: requires raw b64 + format
            if not inp.b64 or not inp.fmt:
                raise ValueError("input_audio mode requires inp.b64 and inp.fmt (use --audio-path)")
            blocks.append({"type": "input_audio", "input_audio": {"data": inp.b64, "format": inp.fmt}})

    return [{"role": "user", "content": blocks}]


def post_chat_completion(
    url: str,
    model: str,
    messages: List[Dict[str, Any]],
    rid: str,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    modalities: Optional[List[str]] = None,   # 👈 NEW
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if modalities is not None:
        payload["modalities"] = modalities      # 👈 NEW (per vLLM-Omni docs)
    headers = {"Content-Type": "application/json", "X-Request-Id": rid}
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()



# -----------------------
# Main sweep
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8091")
    ap.add_argument("--model", required=True)
    ap.add_argument("--percent", type=int, required=True)

    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompt", default="Describe the input in one concise sentence.")
    ap.add_argument("--timeout-s", type=int, default=600)

    ap.add_argument("--mode", choices=["text", "image", "audio"], required=True)

    # Image inputs
    g_img = ap.add_mutually_exclusive_group()
    g_img.add_argument("--image-path", help="local image file (jpg/jpeg/png/webp)")
    g_img.add_argument("--image-url", help="http(s) URL or data URL")

    # Audio inputs
    g_aud = ap.add_mutually_exclusive_group()
    g_aud.add_argument("--audio-path", help="local audio file (wav/mp3/flac/ogg/m4a/aac)")
    g_aud.add_argument("--audio-url", help="http(s) URL or data URL")
    ap.add_argument("--audio-send-mode", choices=["audio_url", "input_audio"], default="audio_url")

    ap.add_argument("--out-jsonl", required=True)

    # Debug knob
    ap.add_argument("--save-raw", action="store_true", help="save raw response JSON to each record (big!)")

    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    inp = build_input_spec(
        mode=args.mode,
        image_path=args.image_path,
        image_url=args.image_url,
        audio_path=args.audio_path,
        audio_url=args.audio_url,
    )

    total = args.warmup + args.trials
    for idx in range(total):
        rid = make_rid()
        is_warmup = idx < args.warmup

        t0 = now_s()
        ok = True
        err = ""
        status_code: Optional[int] = None

        pt = ct = tt = None
        pred_text: Optional[str] = None
        raw: Optional[Dict[str, Any]] = None

        try:
            messages = build_messages(
                prompt=args.prompt,
                rid=rid,
                inp=inp,
                audio_send_mode=args.audio_send_mode,
            )
            want_modalities = ["text"]  # ✅ 强制只输出 text，避免乱码来源
            resp_json = post_chat_completion(
                url=url,
                model=args.model,
                messages=messages,
                rid=rid,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_s=args.timeout_s,
                modalities=want_modalities,
            )
            pt, ct, tt = safe_get_usage(resp_json)
            pred_text = safe_get_text_only(resp_json)
            if args.save_raw:
                raw = resp_json

        except requests.HTTPError as e:
            ok = False
            err = f"HTTPError: {e}"
            if e.response is not None:
                status_code = e.response.status_code
        except Exception as e:
            ok = False
            err = f"Exception: {e}"

        t1 = now_s()
        rec: Dict[str, Any] = {
            "rid": rid,
            "percent": args.percent,
            "is_warmup": is_warmup,
            "latency_s": (t1 - t0),
            "ok": ok,
            "status_code": status_code,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,

            # ✅ only keep text output
            "pred_text": pred_text,

            # meta
            "mode": args.mode,
            "input": inp.meta,
            "prompt": args.prompt,
            "url": url,
            "audio_send_mode": (args.audio_send_mode if args.mode == "audio" else None),
        }
        if args.save_raw:
            rec["raw"] = raw

        with open(args.out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        tag = "WARMUP" if is_warmup else "TRIAL"
        if ok:
            print(
                f"[{tag}] pct={args.percent:3d} rid={rid} latency={rec['latency_s']:.4f}s "
                f"pt={pt} ct={ct} tt={tt} text_len={len(pred_text) if pred_text else 0}"
            )
        else:
            print(
                f"[{tag}] pct={args.percent:3d} rid={rid} FAILED latency={rec['latency_s']:.4f}s err={err}"
            )


if __name__ == "__main__":
    main()

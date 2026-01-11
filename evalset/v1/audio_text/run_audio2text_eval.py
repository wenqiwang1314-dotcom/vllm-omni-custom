#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio -> Text functional test for vLLM-Omni (OpenAI-compatible server)

Works with:
  vllm serve Qwen/Qwen2.5-Omni-7B --omni --served-model-name Qwen-Omni --port 8091 ...

Supports:
  - Single WAV test: --wav /path/to.wav
  - Batch test from manifest.jsonl (each line contains {"id","label","path"}): --manifest manifest.jsonl
Outputs:
  - results/audio2text_results.jsonl
  - prints each response + latency

Important:
  This script sends audio as base64 using OpenAI-style multimodal message:
    {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests
from openai import OpenAI


DEFAULT_PROMPT = (
    "Listen to the audio and describe what is happening in 1–2 concise sentences. "
    "Focus on the main sound sources and the environment. "
    "Do not invent details that are not clearly audible."
)


def wav_to_base64(wav_path: Path) -> str:
    data = wav_path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def load_manifest_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {ln} in {path}: {e}")
    return items


def ensure_server_alive(base_url_v1: str, timeout_s: int = 10) -> List[str]:
    """
    Returns model IDs from /v1/models if reachable.
    base_url_v1 example: http://127.0.0.1:8091/v1
    """
    r = requests.get(f"{base_url_v1}/models", timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return [m["id"] for m in data.get("data", [])]


def pick_model(preferred: str, available: List[str]) -> str:
    if preferred in available:
        return preferred
    if available:
        return available[0]
    return preferred  # best-effort fallback


def call_audio_to_text(
    client: OpenAI,
    model: str,
    prompt: str,
    audio_b64: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    Single call to /v1/chat/completions with multimodal content: text + input_audio(base64)
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav",
                        },
                    },
                ],
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # vLLM returns OpenAI-style response; content should be in message.content
    msg = resp.choices[0].message
    return (msg.content or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://127.0.0.1:8091/v1", help="vLLM OpenAI base URL")
    ap.add_argument("--model", default="Qwen-Omni", help="served-model-name")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="English prompt for audio description")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=60, help="Timeout for /v1/models reachability check only (seconds)")
    ap.add_argument("--sleep", type=float, default=0.05, help="Sleep between requests (seconds)")
    ap.add_argument("--out", default="results/audio2text_results.jsonl", help="Output JSONL file")
    ap.add_argument("--max_samples", type=int, default=0, help="If >0, limit number of samples in batch mode")

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", help="Single wav file path for one-off test")
    group.add_argument("--manifest", help="manifest.jsonl path for batch test")

    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Server check + model selection
    try:
        available_models = ensure_server_alive(base_url, timeout_s=args.timeout)
    except Exception as e:
        raise SystemExit(
            f"[FATAL] Cannot reach vLLM server at {base_url}. "
            f"Ensure `vllm serve ... --port 8091` is running. Error: {e}"
        )

    model = pick_model(args.model, available_models)

    print(f"[INFO] base_url={base_url}")
    print(f"[INFO] available_models={available_models}")
    print(f"[INFO] using_model={model}")
    print(f"[INFO] out={out_path}")

    client = OpenAI(api_key="EMPTY", base_url=base_url)

    # 2) Prepare items
    items: List[Dict[str, Any]] = []
    if args.wav:
        wav_path = Path(args.wav).expanduser().resolve()
        if not wav_path.exists():
            raise SystemExit(f"[FATAL] WAV not found: {wav_path}")
        items = [{"id": wav_path.stem, "label": None, "path": str(wav_path)}]
    else:
        manifest_path = Path(args.manifest).expanduser().resolve()
        if not manifest_path.exists():
            raise SystemExit(f"[FATAL] Manifest not found: {manifest_path}")
        items = load_manifest_jsonl(manifest_path)
        if args.max_samples and args.max_samples > 0:
            items = items[: args.max_samples]

    # 3) Run
    n_ok = 0
    n_fail = 0
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as wf:
        for idx, it in enumerate(items, start=1):
            sample_id = it.get("id") or f"sample_{idx:04d}"
            label = it.get("label")
            wav_path = Path(it["path"]).expanduser().resolve()

            record: Dict[str, Any] = {
                "id": sample_id,
                "label": label,
                "wav": str(wav_path),
                "prompt": args.prompt,
                "model": model,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            if not wav_path.exists():
                record["error"] = "missing_wav"
                wf.write(json.dumps(record, ensure_ascii=False) + "\n")
                wf.flush()
                print(f"[{idx}/{len(items)}] [ERR] {sample_id} missing wav: {wav_path}")
                n_fail += 1
                continue

            try:
                audio_b64 = wav_to_base64(wav_path)
                start = time.time()
                text = call_audio_to_text(
                    client=client,
                    model=model,
                    prompt=args.prompt,
                    audio_b64=audio_b64,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                latency = time.time() - start

                record["latency_s"] = round(latency, 4)
                record["response"] = text

                wf.write(json.dumps(record, ensure_ascii=False) + "\n")
                wf.flush()

                n_ok += 1
                print(f"[{idx}/{len(items)}] [OK] {sample_id} | {latency:.3f}s | {text}")

            except Exception as e:
                record["error"] = f"{type(e).__name__}: {str(e)}"
                wf.write(json.dumps(record, ensure_ascii=False) + "\n")
                wf.flush()
                n_fail += 1
                print(f"[{idx}/{len(items)}] [ERR] {sample_id} | {record['error']}")

            if args.sleep:
                time.sleep(args.sleep)

    dt = time.time() - t0
    print(f"[DONE] ok={n_ok} fail={n_fail} elapsed={dt:.2f}s")

    # Exit status for CI/regression usage
    if n_fail > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


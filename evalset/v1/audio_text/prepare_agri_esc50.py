#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


DEFAULT_TARGET_CLASSES = [
    # Livestock / farm animals
    "cow",
    "sheep",
    "rooster",
    # Farm ecology / insects / birds (proxy)
    "crickets",
    "chirping_birds",
    # Farm environment
    "rain",
    "wind",
    "thunderstorm",
    # Tools / machinery (proxy)
    "chainsaw",
]


def read_esc50_meta(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Common ESC-50 meta columns: filename, fold, target, category, esc10, src_file, take
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc_csv", required=True, help="Path to esc50.csv (meta)")
    ap.add_argument("--audio_dir", required=True, help="Directory containing ESC-50 wav files")
    ap.add_argument("--out_dir", default="agri_samples_esc50", help="Output directory for selected wav files")
    ap.add_argument("--per_class", type=int, default=2, help="How many samples per class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--classes", nargs="*", default=DEFAULT_TARGET_CLASSES)
    args = ap.parse_args()

    random.seed(args.seed)

    esc_csv = Path(args.esc_csv).expanduser().resolve()
    audio_dir = Path(args.audio_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    out_wav_dir = out_dir / "wav"
    out_wav_dir.mkdir(parents=True, exist_ok=True)

    rows = read_esc50_meta(esc_csv)

    # Group by category
    by_cat = defaultdict(list)
    for r in rows:
        cat = r.get("category", "").strip()
        fn = r.get("filename", "").strip()
        if cat and fn:
            by_cat[cat].append(r)

    selected = []
    for cat in args.classes:
        candidates = by_cat.get(cat, [])
        if not candidates:
            print(f"[WARN] No samples found for class: {cat}")
            continue
        k = min(args.per_class, len(candidates))
        picks = random.sample(candidates, k=k)
        selected.extend(picks)

    # Copy wavs + write manifest
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        for i, r in enumerate(selected):
            fn = r["filename"]
            src = audio_dir / fn
            if not src.exists():
                print(f"[WARN] Missing wav file: {src}")
                continue

            # To keep names unique and readable
            cat = r["category"].strip()
            dst_name = f"{cat}__{fn}"
            dst = out_wav_dir / dst_name
            shutil.copy2(src, dst)

            # Minimal jsonl (avoid external deps)
            item_id = f"esc50_{i:04d}"
            line = (
                "{"
                f"\"id\":\"{item_id}\","
                f"\"label\":\"{cat}\","
                f"\"path\":\"{dst.as_posix()}\""
                "}\n"
            )
            mf.write(line)

    print(f"[OK] Selected samples written to: {out_dir}")
    print(f"[OK] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()


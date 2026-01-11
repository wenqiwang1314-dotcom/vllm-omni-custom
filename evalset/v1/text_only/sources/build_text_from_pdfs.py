#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path

import pdfplumber


def normalize_ws(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_sentences(text: str) -> list[str]:
    # 简单句子切分：对 PDF 提取文本足够用
    text = text.replace("\n", " ")
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    # 句号/问号/叹号后切分；保留标点
    parts = re.split(r"(?<=[。！？.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_pages(pdf_path: Path, max_pages: int | None) -> list[str]:
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        n = len(pdf.pages) if max_pages is None else min(len(pdf.pages), max_pages)
        for i in range(n):
            t = pdf.pages[i].extract_text() or ""
            t = normalize_ws(t)
            if t:
                pages.append(t)
    return pages


def sample_chunk(sentences: list[str], min_words: int, max_words: int, max_tries: int = 50) -> str | None:
    if not sentences:
        return None

    # 为了更容易拿到 300–600 words，允许一次拼更多句子
    n = len(sentences)
    for _ in range(max_tries):
        start = random.randint(0, max(0, n - 1))
        buf = []
        wc = 0
        # 最多拼 60 句，避免无限增长
        for j in range(start, min(n, start + 60)):
            s = sentences[j]
            w = len(s.split())
            # 跳过过短句子
            if w < 3:
                continue
            if wc + w > max_words and wc >= min_words:
                break
            if wc + w > max_words and wc < min_words:
                # 还没到 min_words 但超了 max_words：这次失败，重来
                buf = []
                wc = 0
                break
            buf.append(s)
            wc += w
            if wc >= min_words:
                # 到了最小词数，可以接受（不一定要贴近 max_words）
                break

        if wc >= min_words and buf:
            return " ".join(buf).strip()

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="Directory containing PDFs")
    ap.add_argument("--out_jsonl", required=True, help="Output jsonl for raw prompts")
    ap.add_argument("--max_pages", type=int, default=6, help="Max pages to read per PDF (speed control)")
    ap.add_argument("--per_pdf", type=int, default=3, help="How many samples to draw per PDF")
    ap.add_argument("--min_words", type=int, default=80)
    ap.add_argument("--max_words", type=int, default=650)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    pdf_dir = Path(args.pdf_dir)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found under: {pdf_dir}")

    with out_path.open("w", encoding="utf-8") as f:
        for idx, pdf in enumerate(pdfs, start=1):
            pages = extract_pages(pdf, args.max_pages)
            text = "\n\n".join(pages)
            sents = split_sentences(text)

            # 每个 PDF 采样多条
            for k in range(args.per_pdf):
                chunk = sample_chunk(sents, args.min_words, args.max_words)
                if not chunk:
                    continue
                wc = len(chunk.split())

                rec = {
                    "id": f"pdf_{idx:03d}_s{k+1}",
                    "modality": "text",
                    "prompt": chunk,
                    "meta": {
                        "pdf_path": str(pdf),
                        "word_count": wc,
                        "min_words": args.min_words,
                        "max_words": args.max_words,
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote raw prompts -> {out_path}")


if __name__ == "__main__":
    main()

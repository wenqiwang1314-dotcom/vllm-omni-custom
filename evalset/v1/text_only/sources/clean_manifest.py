#!/usr/bin/env python3
import argparse, json, re, hashlib
from collections import Counter, defaultdict

BAD_HARD = [
    r"ytilibatius",  # reversed artifact
]

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    # remove punctuation-ish for similarity hashing
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s

def simhash_key(s: str) -> str:
    # lightweight fingerprint: sha1 of normalized first 512 chars
    n = normalize(s)[:512]
    return hashlib.sha1(n.encode("utf-8")).hexdigest()

def looks_bad(s: str) -> bool:
    # hard-kill only
    for pat in BAD_HARD:
        if re.search(pat, s, flags=re.IGNORECASE):
            return True

    # chart axes: lots of tick numbers + very short words density
    nums = re.findall(r"\b-?\d+(\.\d+)?\b", s)
    if len(nums) >= 30:
        return True

    return False

def sentence_trim(s: str, min_len=160, max_len=1400) -> str:
    s = re.sub(r"\b\d{1,3}\s+\d{1,3}\b", " ", s)  # "38 39"
    s = re.sub(r"\s+", " ", s).strip()

    # drop sentences that look like figure/legend boilerplate
    # split roughly by period/semicolon
    parts = re.split(r"(?<=[\.\!\?])\s+", s)
    kept = []
    for p in parts:
        if re.search(r"\bFigure\s+\d+(\.\d+)?\b", p, re.IGNORECASE):
            continue
        if re.search(r"green-white-purple|colour scheme|maps represent", p, re.IGNORECASE):
            continue
        if re.search(r"\bassessment@dpi\.nsw\.gov\.au\b", p, re.IGNORECASE):
            continue
        kept.append(p)

    s = " ".join(kept).strip()
    s = re.sub(r"\s+", " ", s).strip()

    if len(s) > max_len:
        s = s[:max_len].rsplit(" ", 1)[0]

    return s if len(s) >= min_len else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--max_per_page", type=int, default=1, help="cap samples per page")
    ap.add_argument("--target_n", type=int, default=20, help="target number of samples after cleaning")
    args = ap.parse_args()

    kept = []
    per_page = Counter()
    seen_fp = set()
    dropped = defaultdict(int)

    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            txt = o.get("prompt", "")

            if looks_bad(txt):
                dropped["bad_pattern_or_format"] += 1
                continue

            txt2 = sentence_trim(txt)
            if not txt2:
                dropped["too_short_after_trim"] += 1
                continue

            fp = simhash_key(txt2)
            if fp in seen_fp:
                dropped["duplicate"] += 1
                continue

            page = o.get("meta", {}).get("page", None)
            if page is not None and per_page[page] >= args.max_per_page:
                dropped["page_cap"] += 1
                continue

            # accept
            o["prompt"] = txt2
            o.setdefault("meta", {})
            o["meta"]["cleaned"] = True
            o["meta"]["fingerprint"] = fp[:12]

            kept.append(o)
            seen_fp.add(fp)
            if page is not None:
                per_page[page] += 1

    # if fewer than target_n, just output what we have (do not fabricate)
    kept = kept[: args.target_n]

    with open(args.out_jsonl, "w", encoding="utf-8") as w:
        for o in kept:
            w.write(json.dumps(o, ensure_ascii=False) + "\n")

    print("Kept:", len(kept))
    print("Per-page:", dict(per_page))
    print("Dropped:", dict(dropped))

if __name__ == "__main__":
    main()

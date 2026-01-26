#!/usr/bin/env python3
import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ==== Regex ====
# 你的 ts 前缀: [2026-01-26 10:56:34]
TS_RE = re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+(?P<rest>.*)$")

# Stage 前缀: [Stage-0]
STAGE_RE = re.compile(r"\[Stage-(?P<stage>\d+)\]")

# GreenContext init
GC_INIT_RE = re.compile(
    r"\[GreenContext\]\s+enabled\s+percent=(?P<pct>\d+)\s+actual_sm=(?P<sm>\d+)\s+total_sm=(?P<total>\d+)\s+align=(?P<align>\d+)\s+max_valid=(?P<maxv>\d+)"
)

# GreenContext enter/exit (宽松抓 rid/req_id/pct/sm)
GC_EVT_RE = re.compile(
    r"\[GreenContext\]\s+(?P<evt>enter|exit)\b.*?"
    r"(?:\brid\b|\breq_id\b|\brequest_id\b)\s*[:=\[]\s*(?P<rid>[A-Za-z0-9_\-\.]+)\s*\]?"
    r".*?(?:\bpct\b|\bpercent\b)?\s*[:=]\s*(?P<pct>\d+)?"
    r".*?(?:\bsm\b|\bactual_sm\b)?\s*[:=]\s*(?P<sm>\d+)?",
    re.IGNORECASE
)


def parse_ts(line: str) -> Tuple[Optional[datetime], str]:
    m = TS_RE.match(line)
    if not m:
        return None, line
    ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
    return ts, m.group("rest")


@dataclass
class GcInit:
    pct: int
    sm: int
    total: int
    align: int
    maxv: int


@dataclass
class GcEvent:
    stage: int
    rid: str
    evt: str  # enter/exit
    ts: datetime
    pct: Optional[int]
    sm: Optional[int]


def load_requests(req_jsonl: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with req_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rid = r.get("rid")
            if isinstance(rid, str):
                out[rid] = r
    return out


def parse_logs(log_paths: List[Path]) -> Tuple[Dict[int, GcInit], List[GcEvent]]:
    stage_init: Dict[int, GcInit] = {}
    events: List[GcEvent] = []

    for p in log_paths:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                raw = raw.rstrip("\n")
                ts, rest = parse_ts(raw)
                if ts is None:
                    continue

                sm = STAGE_RE.search(rest)
                if not sm:
                    continue
                stage = int(sm.group("stage"))

                mi = GC_INIT_RE.search(rest)
                if mi:
                    stage_init[stage] = GcInit(
                        pct=int(mi.group("pct")),
                        sm=int(mi.group("sm")),
                        total=int(mi.group("total")),
                        align=int(mi.group("align")),
                        maxv=int(mi.group("maxv")),
                    )
                    continue

                me = GC_EVT_RE.search(rest)
                if me:
                    evt = me.group("evt").lower()
                    rid = me.group("rid")
                    pct_s = me.group("pct")
                    sm_s = me.group("sm")
                    events.append(
                        GcEvent(
                            stage=stage,
                            rid=rid,
                            evt=evt,
                            ts=ts,
                            pct=int(pct_s) if pct_s and pct_s.isdigit() else None,
                            sm=int(sm_s) if sm_s and sm_s.isdigit() else None,
                        )
                    )
    return stage_init, events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--req-jsonl", required=True)
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    req_map = load_requests(Path(args.req_jsonl))
    stage_init, events = parse_logs([Path(x) for x in args.logs])

    # key: (rid, stage) -> enter_ts/exit_ts
    enter_map: Dict[Tuple[str, int], GcEvent] = {}
    exit_map: Dict[Tuple[str, int], GcEvent] = {}

    for e in events:
        k = (e.rid, e.stage)
        if e.evt == "enter":
            # 若重复 enter，以最早为准
            if k not in enter_map or e.ts < enter_map[k].ts:
                enter_map[k] = e
        elif e.evt == "exit":
            # 若重复 exit，以最晚为准
            if k not in exit_map or e.ts > exit_map[k].ts:
                exit_map[k] = e

    rows = []
    for (rid, stage), en in enter_map.items():
        ex = exit_map.get((rid, stage))
        req = req_map.get(rid, {})
        init = stage_init.get(stage)

        pct = en.pct or (init.pct if init else None)
        sm = en.sm or (init.sm if init else None)
        total_sm = init.total if init else None

        dur_s = None
        exit_ts = None
        if ex is not None:
            exit_ts = ex.ts
            dur_s = (ex.ts - en.ts).total_seconds()

        rows.append({
            "rid": rid,
            "stage_id": stage,
            "pct": pct,
            "actual_sm": sm,
            "total_sm": total_sm,
            "enter_time": en.ts.strftime("%Y-%m-%d %H:%M:%S"),
            "exit_time": exit_ts.strftime("%Y-%m-%d %H:%M:%S") if exit_ts else "",
            "stage_dur_s": f"{dur_s:.6f}" if dur_s is not None else "",
            "client_latency_s": f"{req.get('latency_s', '')}",
            "client_percent": req.get("percent", ""),
            "is_warmup": req.get("is_warmup", ""),
            "ok": req.get("ok", ""),
        })

    # 也把 “只有请求但没抓到 enter/exit” 的记录吐出来（方便排查）
    # （可选：你不想要就注释掉）
    seen_rids = {r["rid"] for r in rows}
    for rid, req in req_map.items():
        if rid in seen_rids:
            continue
        rows.append({
            "rid": rid,
            "stage_id": "",
            "pct": "",
            "actual_sm": "",
            "total_sm": "",
            "enter_time": "",
            "exit_time": "",
            "stage_dur_s": "",
            "client_latency_s": f"{req.get('latency_s', '')}",
            "client_percent": req.get("percent", ""),
            "is_warmup": req.get("is_warmup", ""),
            "ok": req.get("ok", ""),
        })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rid", "stage_id", "pct", "actual_sm", "total_sm",
        "enter_time", "exit_time", "stage_dur_s",
        "client_latency_s", "client_percent", "is_warmup", "ok",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (str(x["rid"]), str(x["stage_id"]))):
            w.writerow(r)

    print(f"[OK] wrote {out_path} rows={len(rows)}")
    if not stage_init:
        print("[WARN] did not find any GreenContext init lines (enabled percent=..., actual_sm=...)")
    if not events:
        print("[WARN] did not find any GreenContext enter/exit events. Check your log format & regex.")


if __name__ == "__main__":
    main()


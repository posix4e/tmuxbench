#!/usr/bin/env python3
"""Aggregate a suite sweep into a leaderboard (pass% + pillar A/B/C + missed-event).

Usage:  python harness/suite_agg.py suite_results.jsonl
"""
import json, sys, collections


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "suite_results.jsonl"
    recs = [json.loads(l) for l in open(path) if l.strip()]
    by = collections.defaultdict(list)
    for r in recs:
        by[r["model"]].append(r)

    def pillar(rows, p):
        rs = [x for x in rows if (x.get("category") or "").startswith(p) and x.get("passed") is not None]
        return (sum(1 for x in rs if x["passed"]) / len(rs)) if rs else None

    def miss(rows):
        m = [x["missed_event_rate"] for x in rows if x.get("missed_event_rate") is not None]
        return (sum(m) / len(m)) if m else None

    rows = []
    for m, rs in by.items():
        g = [x for x in rs if x.get("passed") is not None]
        ov = (sum(1 for x in g if x["passed"]) / len(g)) if g else 0
        rows.append({"m": m, "runs": len(rs), "ov": ov, "A": pillar(rs, "A"),
                     "B": pillar(rs, "B"), "C": pillar(rs, "C"), "miss": miss(rs),
                     "errs": sum(1 for x in rs if x.get("error"))})
    rows.sort(key=lambda r: -r["ov"])

    def pc(x):
        return "  . " if x is None else f"{x*100:>3.0f}%"
    print(f"{'model':32} | runs | pass | A·sync | B·evt | C·orch | miss | err")
    for r in rows:
        mr = "  . " if r["miss"] is None else f"{r['miss']:.2f}"
        print(f"{r['m']:32} | {r['runs']:>4} | {pc(r['ov'])} | {pc(r['A'])}  | "
              f"{pc(r['B'])} | {pc(r['C'])}  | {mr} | {r['errs']}")


if __name__ == "__main__":
    main()

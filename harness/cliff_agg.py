#!/usr/bin/env python3
"""Aggregate a cliff sweep into per-model depth x op-count pass-rate heatmaps,
and report each model's "generalization cliff" (where it falls below 50%).

Usage:  python harness/cliff_agg.py cliff_results.jsonl
"""
import json, sys, collections


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "cliff_results.jsonl"
    recs = load(path)
    by = collections.defaultdict(lambda: collections.defaultdict(list))  # model -> (d,n) -> [pass]
    depths, ns = set(), set()
    for r in recs:
        if r.get("passed") is None:
            continue
        by[r["model"]][(r["depth"], r["n"])].append(1 if r["passed"] else 0)
        depths.add(r["depth"]); ns.add(r["n"])
    depths, ns = sorted(depths), sorted(ns)

    # order models by overall pass rate (proxy for capability)
    def overall(m):
        cells = [v for v in by[m].values()]
        flat = [x for c in cells for x in c]
        return sum(flat) / len(flat) if flat else 0
    models = sorted(by, key=overall, reverse=True)

    for m in models:
        cell = by[m]
        print(f"\n=== {m}  (overall {overall(m)*100:.0f}%) ===")
        print("depth\\n   " + "".join(f"{n:>5}" for n in ns) + "   | cliff(n=1)")
        for d in depths:
            row = ""
            for n in ns:
                vals = cell.get((d, n), [])
                row += f"{'  . ' if not vals else f'{sum(vals)/len(vals)*100:>4.0f}'}"
            # cliff at n=1: pass>=0.5?
            v1 = cell.get((d, 1), [])
            mark = ""
            print(f"  d{d}  {row}")
        # generalization cliff: deepest depth still >=50% at n=1, and max n still >=50% at d0/d1
        passes = {(d, n): (sum(v)/len(v) if v else 0) for (d, n), v in cell.items()}
        deepest = max([d for d in depths if passes.get((d, 1), 0) >= 0.5], default=None)
        # for the simplest mechanism (lowest depth present), how many ops before falloff
        d_lo = depths[0]
        maxn = max([n for n in ns if passes.get((d_lo, n), 0) >= 0.5], default=None)
        print(f"  -> deepest indirection at n=1 with >=50%: "
              f"{'d'+str(deepest) if deepest is not None else 'none'};"
              f"  max op-count at d{d_lo} with >=50%: {maxn if maxn is not None else 'none'}")

    print("\n(cells = pass%, '.' = no data; rows=indirection depth, cols=op-count n)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate SFT warm-start data from tmuxbench reference solutions (free demos).

Each row: {prompt, completion, depth, n} where `prompt` is what the policy sees
(rl/env.make_prompt) and `completion` is the task's reference script wrapped in a
fenced block (the target the model learns to emit). Used to lift a small model
off the 0% floor before GRPO. Verifies every demo actually scores 1.0.

Usage:  python rl/make_sft.py --out rl/sft.jsonl
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "harness"))
import cliff  # noqa: E402
import env as rlenv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=sorted(cliff.DEPTHS))
    ap.add_argument("--ns", type=int, nargs="+", default=cliff.N_DEFAULT)
    ap.add_argument("--repeats", type=int, default=3, help="copies per cell (token RNG varies)")
    ap.add_argument("--out", default=os.path.join(HERE, "sft.jsonl"))
    args = ap.parse_args()

    rows, verified = [], 0
    for d in args.depths:
        for n in args.ns:
            ref = cliff.DEPTHS[d]["reference"].format(n=n)
            completion = f"```bash\n{ref}\n```"
            # verify the demo is actually correct before teaching it
            if rlenv.score(d, n, completion) >= 0.99:
                verified += 1
            for _ in range(args.repeats):
                rows.append({"prompt": rlenv.make_prompt(d, n),
                             "completion": completion, "depth": d, "n": n})
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cells = len(args.depths) * len(args.ns)
    print(f"wrote {len(rows)} SFT demos to {args.out} "
          f"({verified}/{cells} reference cells verified at reward>=0.99)")


if __name__ == "__main__":
    main()

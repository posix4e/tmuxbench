#!/usr/bin/env python3
"""Sweep models over the generalization-cliff grid (depth x op-count).

Infra-agnostic: hits any OpenAI-compatible endpoint. Point it at OpenRouter
(default) or a local vLLM server with --base-url http://HOST:8000/v1 (and
--key-env / a dummy key). Appends one JSON line per run.

Example (local vLLM size ladder, run on the box that can reach the server):
    OPENAI_API_KEY=x python harness/cliff_sweep.py \
        --models Qwen/Qwen2.5-1.5B-Instruct --policy openai \
        --base-url http://192.168.122.73:8000/v1 --seeds 5 --out cliff.jsonl
"""
import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cliff, run as runner  # noqa: E402

_lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--depths", type=int, nargs="+", default=sorted(cliff.DEPTHS))
    ap.add_argument("--ns", type=int, nargs="+", default=cliff.N_DEFAULT)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--policy", default="openrouter",
                    choices=["openrouter", "openai", "anthropic"])
    ap.add_argument("--base-url", help="OpenAI-compatible base URL (sets OPENAI_BASE_URL)")
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--loop", default="tool", choices=["text", "tool"],
                    help="agent loop; use 'text' for small/open models without reliable tool-calling")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="cliff_results.jsonl")
    args = ap.parse_args()
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url

    units = [(m, d, n, s) for m in args.models for d in args.depths
             for n in args.ns for s in range(1, args.seeds + 1)]
    print(f"{len(units)} runs: {len(args.models)} models x {len(args.depths)} depths "
          f"x {len(args.ns)} ns x {args.seeds} seeds", flush=True)

    def run_one(unit):
        m, d, n, s = unit
        task = cliff.build(d, n)
        try:
            r = runner.run_task(task, agent_mode="model", policy_name=args.policy,
                                model_id=m, loop_mode=args.loop, max_turns=args.max_turns,
                                verbose=False)
        except Exception as e:
            r = {"error": str(e)[:200]}
        rec = {"model": m, "depth": d, "n": n, "seed": s,
               "passed": r.get("passed"), "score": r.get("assertion_score"),
               "turns": r.get("phases", {}).get("agent", {}).get("turns"),
               "error": r.get("error")}
        with _lock:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    done = passed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, u) for u in units]
        for fu in as_completed(futs):
            r = fu.result(); done += 1
            if r.get("passed"):
                passed += 1
            if done % 25 == 0:
                print(f"  {done}/{len(units)} done, {passed} passed", flush=True)
    print(f"DONE: {done} runs, {passed} passed -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

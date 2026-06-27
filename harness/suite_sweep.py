#!/usr/bin/env python3
"""Sweep models over the full tmuxbench task suite (the missing-events leaderboard).

Infra-agnostic: hits any OpenAI-compatible endpoint. Point at a local vLLM server
with --base-url http://HOST:8000/v1 (small open models) or OpenRouter (default).
Appends one JSON line per run; aggregate with suite_agg.py.

Example (local vLLM, run on the box that can reach the server):
    OPENAI_API_KEY=x python harness/suite_sweep.py \
        --models Qwen/Qwen2.5-7B-Instruct --policy openai \
        --base-url http://192.168.122.73:8000/v1 --loop text --seeds 5 --out suite.jsonl
"""
import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as runner  # noqa: E402

_lock = threading.Lock()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--policy", default="openrouter",
                    choices=["openrouter", "openai", "anthropic"])
    ap.add_argument("--base-url", help="OpenAI-compatible base URL (sets OPENAI_BASE_URL)")
    ap.add_argument("--loop", default="tool", choices=["text", "tool"])
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="suite_results.jsonl")
    args = ap.parse_args()
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url

    tasks = runner.discover_all()
    units = [(m, t, s) for m in args.models for t in tasks
             for s in range(1, args.seeds + 1)]
    print(f"{len(units)} runs: {len(args.models)} models x {len(tasks)} tasks "
          f"x {args.seeds} seeds ({args.loop} loop)", flush=True)

    def run_one(unit):
        m, task_dir, s = unit
        task = runner.load_task(task_dir)
        try:
            r = runner.run_task(task, agent_mode="model", policy_name=args.policy,
                                model_id=m, loop_mode=args.loop, max_turns=args.max_turns,
                                verbose=False)
        except Exception as e:
            r = {"id": task.get("id"), "error": str(e)[:200]}
        rec = {"model": m, "task": task.get("id"), "category": task.get("category"),
               "seed": s, "passed": r.get("passed"), "score": r.get("assertion_score"),
               "missed_event_rate": r.get("missed_event_rate"),
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
            passed += 1 if r.get("passed") else 0
            if done % 25 == 0:
                print(f"  {done}/{len(units)} done, {passed} passed", flush=True)
    print(f"DONE: {done} runs, {passed} passed -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

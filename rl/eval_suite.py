#!/usr/bin/env python3
"""Transfer eval: prompt a model once per REAL tmuxbench task (held out from RL,
which trained only on the synthetic cliff tasks), score the emitted script with the
harness. Tells us whether cliff-RL generalizes to the actual benchmark or just
memorized the cliff generator.

Run on a box that can reach the vLLM server AND has tmux (e.g. tdx2):
  python rl/eval_suite.py --base-url http://IP:8000/v1 --model Qwen/Qwen2.5-3B-Instruct --out rl/eval_suite_base.jsonl
  python rl/eval_suite.py --base-url http://IP:8000/v1 --model grpo --out rl/eval_suite_grpo.jsonl
"""
import argparse, json, os, sys, tempfile, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "harness"))
import env as rlenv  # noqa: E402  (extract_script, SYS_HINT)
import run as runner  # noqa: E402

SYS_HINT = rlenv.SYS_HINT


def generate(base_url, model, prompt, key="x", temperature=0.0, max_tokens=512):
    body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": temperature,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + key, "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"].get("content") or ""


def score_task(task, script):
    script = rlenv.extract_script(script)
    if not script:
        return None, 0.0
    fd, path = tempfile.mkstemp(prefix="rl_suite_", suffix=".sh")
    try:
        os.write(fd, script.encode()); os.close(fd)
        r = runner.run_task(task, agent_mode="cmdfile", agent_cmdfile=path, verbose=False)
        return r.get("passed"), float(r.get("assertion_score") or 0.0)
    except Exception:
        return None, 0.0
    finally:
        try: os.unlink(path)
        except OSError: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = runner.discover_all()
    n = 0
    with open(args.out, "w") as f:
        for td in tasks:
            task = runner.load_task(td)
            prompt = SYS_HINT + "\n\n# Task\n" + task.get("prompt", "")
            for s in range(1, args.seeds + 1):
                try:
                    comp = generate(args.base_url, args.model, prompt, temperature=args.temperature)
                    passed, sc = score_task(task, comp)
                except Exception as e:
                    passed, sc = None, 0.0
                rec = {"model": args.model, "task": task.get("id"), "category": task.get("category"),
                       "seed": s, "passed": bool(passed), "score": sc}
                f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
    print(f"[eval_suite] {n} runs -> {args.out}  (aggregate: harness/suite_agg.py {args.out})")


if __name__ == "__main__":
    main()

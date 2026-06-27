#!/usr/bin/env python3
"""Single-turn eval: prompt a model (vLLM OpenAI endpoint) once per task, score
the emitted script with the real harness reward (rl/env.score). This matches how
the policy was trained, giving an honest before/after. Output is cliff_agg-compatible
(depth,n,passed) so you can reuse `harness/cliff_agg.py`.

Run on a box that can reach the vLLM server AND has tmux (e.g. tdx2):
  python rl/eval.py --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen2.5-3B-Instruct \
      --out rl/eval_base.jsonl
  # to eval a LoRA served by vLLM (--enable-lora --lora-modules name=path):
  python rl/eval.py --base-url ... --model <lora-name> --out rl/eval_grpo.jsonl
"""
import argparse, json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import env as rlenv  # noqa: E402
import cliff  # noqa: E402


def generate(base_url, model, prompt, key="x", temperature=0.0, max_tokens=256):
    body = json.dumps({"model": model, "max_tokens": max_tokens,
                       "temperature": temperature,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + key,
                                          "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"].get("content") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--depths", type=int, nargs="+", default=sorted(cliff.DEPTHS))
    ap.add_argument("--ns", type=int, nargs="+", default=cliff.N_DEFAULT)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_done = 0
    with open(args.out, "w") as f:
        for d in args.depths:
            for n in args.ns:
                for s in range(1, args.seeds + 1):
                    try:
                        comp = generate(args.base_url, args.model, rlenv.make_prompt(d, n),
                                        temperature=args.temperature)
                        sc = rlenv.score(d, n, comp)
                    except Exception as e:
                        comp, sc = f"[err {e}]"[:80], 0.0
                    rec = {"model": args.model, "depth": d, "n": n, "seed": s,
                           "score": sc, "passed": sc >= 0.99}
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    n_done += 1
    print(f"[eval] {n_done} runs -> {args.out}  (aggregate: harness/cliff_agg.py {args.out})")


if __name__ == "__main__":
    main()

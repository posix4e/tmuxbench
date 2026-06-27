#!/usr/bin/env python3
"""tmuxbench as a single-turn RLVR environment.

The policy writes ONE shell script that solves a task; we run it through the
harness in `cmdfile` mode and return the (dense) assertion score as reward. This
reuses the real grader on real tmux state, so the reward is cheat-resistant and
free of any judge model. Multi-turn ReAct is collapsed to single-shot script
generation (the cliff + most event tasks are scriptable in one go).

Public API:
  make_prompt(depth, n)        -> the instruction shown to the policy
  extract_script(completion)   -> strip prose/markdown fences to a runnable script
  score(depth, n, script)      -> reward in [0,1] (assertion_score), 0.0 on failure
  reward_batch(rows, comps)    -> list[float]  (rows carry depth/n)

`rl/make_sft.py` uses make_prompt + each task's reference solution as expert demos.
Designed to be the `reward_funcs` callable for TRL GRPO.
"""
import os, re, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "harness"))
import cliff, run as runner  # noqa: E402

SYS_HINT = (
    "You are operating tmux via shell commands. Write a single shell script (plain "
    "`tmux ...` and shell commands; the server is already running) that completes "
    "the task. Output ONLY the script, no explanation."
)


def make_prompt(depth, n):
    return SYS_HINT + "\n\n# Task\n" + cliff.DEPTHS[depth]["prompt"].format(n=n)


_FENCE = re.compile(r"```(?:bash|sh)?\s*(.*?)```", re.S)


def extract_script(completion):
    """Pull a runnable script out of a model completion (handles ``` fences)."""
    m = _FENCE.search(completion or "")
    s = m.group(1) if m else (completion or "")
    # drop a leading shebang line if present; keep the rest verbatim
    lines = [ln for ln in s.splitlines() if not ln.strip().startswith("#!")]
    return "\n".join(lines).strip()


def score(depth, n, script):
    """Reward = assertion_score in [0,1] from running `script` as the cmdfile agent."""
    script = extract_script(script)
    if not script:
        return 0.0
    task = cliff.build(depth, n)
    fd, path = tempfile.mkstemp(prefix="rl_agent_", suffix=".sh")
    try:
        os.write(fd, script.encode()); os.close(fd)
        r = runner.run_task(task, agent_mode="cmdfile", agent_cmdfile=path, verbose=False)
        s = r.get("assertion_score")
        return float(s) if s is not None else 0.0
    except Exception:
        return 0.0
    finally:
        try: os.unlink(path)
        except OSError: pass


def reward_batch(depths, ns, completions):
    return [score(d, n, c) for d, n, c in zip(depths, ns, completions)]


if __name__ == "__main__":
    # local sanity: reference -> ~1.0, empty -> 0.0, across the grid
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=sorted(cliff.DEPTHS))
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 4])
    args = ap.parse_args()
    ok = 0; tot = 0
    for d in args.depths:
        for n in args.ns:
            ref = cliff.DEPTHS[d]["reference"].format(n=n)
            r_ref = score(d, n, ref)
            r_empty = score(d, n, "echo nope")
            good = r_ref >= 0.99 and r_empty < 0.5
            ok += good; tot += 1
            print(f"  d{d} n{n}: reward(ref)={r_ref:.2f} reward(junk)={r_empty:.2f} "
                  f"{'ok' if good else 'CHECK'}")
    print(f"env sanity: {ok}/{tot} cells give clean reward separation")

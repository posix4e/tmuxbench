#!/usr/bin/env python3
"""tmuxbench generalization-cliff task generator.

Goal: quantify *when tmux gets in the way* by holding the task semantics trivial
and constant while scaling two axes, then finding where a model falls off:

  - indirection DEPTH (d): how much tmux machinery sits between the agent and the
    value it must route. d0 has no tmux at all (plain files); each higher rung
    swaps in a more involved tmux retrieval path.
  - op COUNT (n): how many independent values must be retrieved and routed.

Every cell (d, n) is the SAME job: read n secret tokens and write them to
$REPORT (one per line). The tokens are random per run and the expected answer is
stashed where the *checker* reads it (`$HOME/.bench_expect`), never named in the
prompt — so a model cannot shortcut the mechanism. Grading is order-insensitive.

`build(d, n)` returns a task dict compatible with run.run_task (no TOML needed).
Run `python harness/cliff.py` to validate the whole grid on reference solutions.
"""

# --- shared bash helpers, injected into every setup ---
_GEN = r'''
gen(){ head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n'; }
: > "$HOME/.bench_expect"
'''

# Each depth defines: PROMPT (how the agent is told to retrieve), SETUP body that
# places n tokens via the mechanism and appends each to $HOME/.bench_expect, a
# REFERENCE that retrieves them to $REPORT, and PERCEIVE (a grep over the
# agent-only cmdlog proving the tmux mechanism was used; None for d0).

DEPTHS = {
    0: {
        "name": "files (no tmux)",
        "prompt": ("{n} secret value(s) are saved in the files $HOME/secrets/s1 .. "
                   "$HOME/secrets/s{n}. Read each file and write all the values, one "
                   "per line, to the file $REPORT."),
        "setup": '''mkdir -p "$HOME/secrets"
for i in $(seq 1 {n}); do t=$(gen); printf '%s\\n' "$t" > "$HOME/secrets/s$i"; printf '%s\\n' "$t" >> "$HOME/.bench_expect"; done''',
        "reference": '''for i in $(seq 1 {n}); do cat "$HOME/secrets/s$i"; done > "$REPORT"''',
        "perceive": None,
    },
    1: {
        "name": "tmux options",
        "prompt": ("{n} secret value(s) are stored in the session-wide tmux user "
                   "options @secret1 .. @secret{n}. Read each option and write all "
                   "the values, one per line, to the file $REPORT."),
        "setup": '''tmux new-session -d -s cliff -x 200 -y 50
for i in $(seq 1 {n}); do t=$(gen); tmux set -g "@secret$i" "$t"; printf '%s\\n' "$t" >> "$HOME/.bench_expect"; done''',
        "reference": '''for i in $(seq 1 {n}); do tmux show -gv "@secret$i"; done > "$REPORT"''',
        "perceive": r'show.*@secret|show-option',
    },
    2: {
        "name": "pane capture",
        "prompt": ("{n} secret value(s) have each been printed into their own tmux "
                   "window (cliff:w1 .. cliff:w{n}). Capture each pane's contents, "
                   "read the value it printed, and write all the values, one per "
                   "line, to the file $REPORT."),
        "setup": '''tmux new-session -d -s cliff -x 200 -y 50
for i in $(seq 1 {n}); do t=$(gen); tmux new-window -t cliff: -n "w$i"; tmux send-keys -t "cliff:w$i" "clear; printf 'secret %s' $t" Enter; printf '%s\\n' "$t" >> "$HOME/.bench_expect"; done
sleep 0.5''',
        "reference": '''for i in $(seq 1 {n}); do tmux capture-pane -t "cliff:w$i" -p | grep -oE '[0-9a-f]{{8}}' | tail -1; done > "$REPORT"''',
        "perceive": r'capture-pane',
    },
    3: {
        "name": "targeted pane (distractors)",
        "prompt": ("Each real secret is printed in a window named real1 .. real{n}; "
                   "decoy windows (named junkN) print bogus values. Capture only the "
                   "real* windows, read their values, and write them one per line to "
                   "$REPORT."),
        "setup": '''tmux new-session -d -s cliff -x 200 -y 50
for i in $(seq 1 {n}); do tmux new-window -t cliff: -n "junk$i"; tmux send-keys -t "cliff:junk$i" "clear; printf 'secret deadbeef'" Enter; done
for i in $(seq 1 {n}); do t=$(gen); tmux new-window -t cliff: -n "real$i"; tmux send-keys -t "cliff:real$i" "clear; printf 'secret %s' $t" Enter; printf '%s\\n' "$t" >> "$HOME/.bench_expect"; done
sleep 0.5''',
        "reference": '''for i in $(seq 1 {n}); do tmux capture-pane -t "cliff:real$i" -p | grep -oE '[0-9a-f]{{8}}' | tail -1; done > "$REPORT"''',
        "perceive": r'capture-pane',
    },
    4: {
        "name": "respawn dead pane",
        "prompt": ("{n} worker window(s) (cliff:w1 .. cliff:w{n}) have died before "
                   "printing their secret. Respawn each so it prints its value, "
                   "capture it, and write all values one per line to $REPORT."),
        "setup": '''tmux new-session -d -s cliff -x 200 -y 50
tmux set-window-option -g remain-on-exit on
for i in $(seq 1 {n}); do t=$(gen); printf '%s\\n' "$t" >> "$HOME/.bench_expect"; tmux new-window -t cliff: -n "w$i" "false"; tmux set-environment -g "TOK$i" "$t"; done
sleep 0.4''',
        # reference: respawn each dead pane with a command that prints its token, then capture
        "reference": '''for i in $(seq 1 {n}); do t=$(tmux show-environment -g "TOK$i" | cut -d= -f2); tmux respawn-pane -t "cliff:w$i" "printf 'secret %s' $t; sleep 30"; done
sleep 0.4
for i in $(seq 1 {n}); do tmux capture-pane -t "cliff:w$i" -p | grep -oE '[0-9a-f]{{8}}' | tail -1; done > "$REPORT"''',
        "perceive": r'respawn-pane',
    },
}

N_DEFAULT = [1, 2, 4, 8, 16]


def build(d, n):
    spec = DEPTHS[d]
    checks = [{
        "id": "values_correct", "required": True, "weight": 1,
        "event": "retrieve", "role": "react", "op": "eq", "expect": "1",
        "query": ('if [ "$(sort -u "$REPORT" 2>/dev/null)" = '
                  '"$(sort -u "$HOME/.bench_expect")" ]; then echo 1; else echo 0; fi'),
    }]
    if spec["perceive"]:
        checks.append({
            "id": "used_mechanism", "required": False, "weight": 0,
            "event": "retrieve", "role": "perceive", "op": "ge", "expect": "1",
            "query": "grep -cE '%s' \"$AGENT_CMDLOG\" 2>/dev/null || true" % spec["perceive"],
        })
    return {
        "id": f"CLIFF-d{d}-n{n}", "category": f"CLIFF-D{d}", "tier": "cliff",
        "title": f"depth {d} ({spec['name']}), {n} ops",
        "prompt": spec["prompt"].format(n=n),
        "setup": _GEN + spec["setup"].format(n=n),
        "reference": spec["reference"].format(n=n),
        "checks": checks,
        "env": {"width": 200, "height": 50},
        "_dir": "<generated>", "_path": "<generated>",
    }


def grid(depths=None, ns=None):
    depths = depths if depths is not None else sorted(DEPTHS)
    ns = ns if ns is not None else N_DEFAULT
    return [build(d, n) for d in depths for n in ns]


if __name__ == "__main__":
    import os, sys, argparse
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import run as runner
    ap = argparse.ArgumentParser(description="validate the cliff grid on reference")
    ap.add_argument("--depths", type=int, nargs="+")
    ap.add_argument("--ns", type=int, nargs="+")
    args = ap.parse_args()
    ok = 0; tot = 0
    for task in grid(args.depths, args.ns):
        r = runner.run_task(task, agent_mode="reference", verbose=False)
        tot += 1; ok += 1 if r.get("passed") else 0
        flag = "ok " if r.get("passed") else "FAIL"
        print(f"  {flag} {task['id']:16} score={r.get('assertion_score')} "
              f"miss={r.get('missed_event_rate')}")
    print(f"reference grid: {ok}/{tot} pass")

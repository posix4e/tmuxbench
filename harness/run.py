#!/usr/bin/env python3
"""tmuxbench runner + grader (M0/M1).

Per-task lifecycle:
    isolate -> setup -> agent -> check -> teardown -> emit result JSON

Isolation: each task gets its own tmux server (`tmux -L bench_<uuid> -f /dev/null`),
a temp $HOME, a temp working dir, and file sinks ($SINK, $REPORT). A PATH shim
(harness/tmux-shim) scopes every `tmux` call to the task socket and logs it.

Grading reads back real server state via the task's `checks` (shell queries
compared to expectations) -- never the model's prose. Verification is independent
of the reference solution: any command sequence producing the right state passes.

Agent modes:
    --agent reference   (default for M0) run the task's `reference` script as the agent
    --agent cmdfile F   run the shell script in file F as the agent
    (a real model loop is M2 / task #3; this file is the substrate it will plug into)

Usage:
    run.py TASK_DIR [TASK_DIR ...]      # run specific tasks
    run.py --all                        # run every tasks/*/task.toml
    run.py --all --json                 # machine-readable JSON to stdout
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, uuid

HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HARNESS_DIR)
SHIM = os.path.join(HARNESS_DIR, "tmux-shim")

try:
    import tomllib
except ModuleNotFoundError:
    sys.exit("tmuxbench: needs Python >= 3.11 (stdlib tomllib).")


def load_task(task_dir):
    path = task_dir if task_dir.endswith(".toml") else os.path.join(task_dir, "task.toml")
    with open(path, "rb") as f:
        t = tomllib.load(f)
    t["_path"] = path
    t["_dir"] = os.path.dirname(path)
    return t


def compare(op, out, expect):
    if op == "eq":
        return out == str(expect)
    if op == "ne":
        return out != str(expect)
    if op in ("lt", "le", "gt", "ge"):
        try:
            a, b = float(out), float(expect)
        except ValueError:
            return False
        return {"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[op]
    if op == "regex":
        return re.search(str(expect), out, re.S) is not None
    if op == "in":
        exp = expect if isinstance(expect, list) else [expect]
        return out in [str(x) for x in exp]
    if op == "contains":
        return str(expect) in out
    if op == "nonempty":
        return len(out) > 0
    raise ValueError(f"unknown op: {op}")


def run_shell(script, env, cwd, timeout):
    """Run a shell snippet; return (exit_code, stdout, stderr, timed_out)."""
    try:
        p = subprocess.run(
            ["bash", "-c", script], env=env, cwd=cwd,
            capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or ""), (e.stderr or ""), True


def run_task(task, agent_mode="reference", agent_cmdfile=None, verbose=True,
             policy_name="mock", model_id="claude-haiku-4-5-20251001", max_turns=12,
             loop_mode="text"):
    real_tmux = shutil.which("tmux")  # PATH not yet shimmed inside this process
    if not real_tmux:
        sys.exit("tmuxbench: real tmux not found on PATH.")

    workdir = tempfile.mkdtemp(prefix="tmuxbench_")
    home = os.path.join(workdir, "home"); os.makedirs(home, exist_ok=True)
    sink = os.path.join(workdir, "sink")
    report = os.path.join(workdir, "report")
    cmdlog = os.path.join(workdir, "cmdlog")
    agent_cmdlog = os.path.join(workdir, "agent_cmdlog")  # agent-only log, frozen for checks
    socket = "bench_" + uuid.uuid4().hex[:12]

    # Expose the shim under the literal name `tmux`, first on PATH, so the model
    # (and task scripts) type plain `tmux` and we transparently scope the socket.
    bindir = os.path.join(workdir, "bin"); os.makedirs(bindir, exist_ok=True)
    os.symlink(SHIM, os.path.join(bindir, "tmux"))

    env = dict(os.environ)
    env.pop("TMUX", None); env.pop("TMUX_PANE", None)  # never nest in a live tmux
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    env["HOME"] = home
    env["BENCH_SOCKET"] = socket
    env["BENCH_REAL_TMUX"] = real_tmux
    env["BENCH_CMDLOG"] = cmdlog
    env["AGENT_CMDLOG"] = agent_cmdlog  # checks grep this to test what the AGENT did
    env["SINK"] = sink
    env["REPORT"] = report

    geom = task.get("env", {})
    env["BENCH_WIDTH"] = str(geom.get("width", 200))
    env["BENCH_HEIGHT"] = str(geom.get("height", 50))

    result = {
        "id": task.get("id"), "category": task.get("category"),
        "tier": task.get("tier"), "title": task.get("title"),
        "socket": socket, "agent_mode": agent_mode,
        "phases": {}, "checks": [], "agent_cmds": 0,
    }

    try:
        # --- setup ---
        rc, so, se, to = run_shell(task.get("setup", "true"), env, workdir, 30)
        result["phases"]["setup"] = {"exit": rc, "timeout": to, "stderr": se.strip()[:500]}

        # --- agent (clear cmdlog so we count only agent commands) ---
        open(cmdlog, "w").close()
        if agent_mode == "model":
            import agent as agentmod
            def run_cmd(cmd):
                return run_shell(cmd, env, workdir, 30)
            prompt = task.get("prompt", "")
            # Tool-calling loop (native function calls) for the model providers that
            # support it; falls through to the text RUN:/DONE loop otherwise.
            if (loop_mode == "tool"
                    and policy_name in ("openai", "openrouter", "anthropic")):
                if policy_name == "anthropic":
                    loop = agentmod.run_anthropic_tool_agent(prompt, run_cmd, model_id,
                                                             max_turns=max_turns)
                else:
                    loop = agentmod.run_openai_tool_agent(prompt, run_cmd, model_id,
                                                          max_turns=max_turns)
            else:
                if policy_name == "anthropic":
                    policy = agentmod.anthropic_policy(model_id)
                elif policy_name in ("openai", "openrouter"):
                    policy = agentmod.openai_policy(model_id)
                elif policy_name == "trustedrouter":
                    # model_id carries the fusion preset: "quality" or "budget"
                    preset = (model_id.rsplit(":", 1)[-1] or "quality")
                    policy = agentmod.trustedrouter_policy(preset=preset)
                else:  # mock: replay a script (cmdfile if given, else the reference) via the loop
                    src = open(agent_cmdfile).read() if agent_cmdfile else task.get("reference", "true")
                    policy = agentmod.mock_policy(agentmod.commands_from_script(src))
                loop = agentmod.run_model_agent(prompt, run_cmd, policy, max_turns=max_turns)
            result["phases"]["agent"] = {"exit": 0, "timeout": False,
                                         "turns": loop["turns"], "done": loop["done"]}
            result["transcript"] = loop["transcript"]
        else:
            if agent_mode == "reference":
                agent_script = task.get("reference", "true")
            elif agent_mode == "cmdfile":
                with open(agent_cmdfile) as f:
                    agent_script = f.read()
            else:
                raise ValueError(f"unknown agent mode: {agent_mode}")
            rc, so, se, to = run_shell(agent_script, env, workdir, task.get("env", {}).get("agent_timeout", 60))
            result["phases"]["agent"] = {"exit": rc, "timeout": to, "stderr": se.strip()[:500]}
        try:
            with open(cmdlog) as f:
                result["agent_cmds"] = sum(1 for _ in f)
        except FileNotFoundError:
            pass
        # Freeze the agent-only command log so checks can grade what the agent did
        # (later check queries also call tmux, which would pollute the live cmdlog).
        try:
            shutil.copyfile(cmdlog, agent_cmdlog)
        except FileNotFoundError:
            open(agent_cmdlog, "w").close()

        # --- checks ---
        total_w = passed_w = 0.0
        req_ok = True
        events = {}  # event name -> {"checks": [...], "handled": bool}
        for chk in task.get("checks", []):
            w = float(chk.get("weight", 1)); required = bool(chk.get("required", True))
            rc, so, se, to = run_shell(chk["query"], env, workdir, chk.get("timeout", 15))
            out = so.strip()
            ok = (not to) and compare(chk["op"], out, chk.get("expect"))
            total_w += w
            if ok:
                passed_w += w
            elif required:
                req_ok = False
            entry = {
                "id": chk.get("id"), "op": chk["op"], "expect": chk.get("expect"),
                "got": out[:200], "required": required, "weight": w,
                "timeout": to, "pass": ok,
            }
            ev = chk.get("event")
            if ev:  # event-handling check: role is "perceive" / "react" / "state"
                entry["event"] = ev; entry["role"] = chk.get("role")
                events.setdefault(ev, {"checks": []})["checks"].append(ok)
            result["checks"].append(entry)
        result["assertion_score"] = round(passed_w / total_w, 4) if total_w else 0.0
        result["passed"] = req_ok and bool(task.get("checks"))

        # Missing-events metric: an event is "handled" iff all its checks pass
        # (perceive AND react). missed_event_rate = missed / fired.
        if events:
            for ev, d in events.items():
                d["handled"] = all(d["checks"]); d["checks"] = len(d["checks"])
            fired = len(events); missed = sum(1 for d in events.values() if not d["handled"])
            result["events"] = {ev: d["handled"] for ev, d in events.items()}
            result["missed_event_rate"] = round(missed / fired, 4)
    finally:
        # --- teardown (always) ---
        run_shell("tmux kill-server >/dev/null 2>&1 || true", env, workdir, 10)
        shutil.rmtree(workdir, ignore_errors=True)

    return result


def discover_all():
    root = os.path.join(REPO_ROOT, "tasks")
    out = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if os.path.isfile(os.path.join(d, "task.toml")):
                out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description="tmuxbench runner + grader")
    ap.add_argument("tasks", nargs="*", help="task dirs or .toml files")
    ap.add_argument("--all", action="store_true", help="run all tasks/*/task.toml")
    ap.add_argument("--agent", default="reference", choices=["reference", "cmdfile", "model"])
    ap.add_argument("--cmdfile", help="shell script to run as the agent (with --agent cmdfile, or as the mock source)")
    ap.add_argument("--policy", default="mock",
                    choices=["mock", "anthropic", "openai", "openrouter", "trustedrouter"],
                    help="model-loop policy (with --agent model)")
    ap.add_argument("--model", default="openai/gpt-4o-mini", help="model id for the chosen policy")
    ap.add_argument("--max-turns", type=int, default=12, help="agent-loop turn budget (with --agent model)")
    ap.add_argument("--loop", default="text", choices=["text", "tool"],
                    help="model loop: text RUN:/DONE protocol, or native tool-calling (openai/anthropic)")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    task_dirs = discover_all() if args.all else args.tasks
    if not task_dirs:
        ap.error("no tasks given (use --all or pass task dirs)")

    results = []
    for td in task_dirs:
        task = load_task(td)
        r = run_task(task, agent_mode=args.agent, agent_cmdfile=args.cmdfile, verbose=not args.json,
                     policy_name=args.policy, model_id=args.model, max_turns=args.max_turns,
                     loop_mode=args.loop)
        results.append(r)
        if not args.json:
            mark = "PASS" if r["passed"] else "FAIL"
            ev = ""
            if "missed_event_rate" in r:
                missed = [e for e, ok in r["events"].items() if not ok]
                ev = f" missed-events={r['missed_event_rate']:.2f}" + (f" {missed}" if missed else "")
            print(f"[{mark}] {r['id']:<16} score={r['assertion_score']:.2f} "
                  f"cmds={r['agent_cmds']:<3}{ev}  {r['title']}")
            for c in r["checks"]:
                if not c["pass"]:
                    print(f"        ✗ {c['id']}: op={c['op']} expect={c['expect']!r} got={c['got']!r}"
                          + ("  [timeout]" if c["timeout"] else ""))

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        n = len(results); p = sum(1 for r in results if r["passed"])
        avg = sum(r["assertion_score"] for r in results) / n if n else 0
        print(f"\n{p}/{n} tasks passed · mean assertion score {avg:.2f}")

    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()

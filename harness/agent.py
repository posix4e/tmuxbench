#!/usr/bin/env python3
"""tmuxbench shell-agent loop (model under test).

A minimal, dependency-free ReAct-style loop: the model is told it has one tool —
a shell — and must drive the task to completion. Protocol (model-agnostic, plain
text so it works with any chat endpoint):

    The model replies with EITHER
        RUN: <a single shell command>
    OR
        DONE
    (optionally preceded by a short THINK: ... line, which we ignore for control).

Each RUN is executed in the task's prepared environment (the PATH shim scopes
`tmux` to the isolated socket and logs it), and the trimmed stdout/stderr/exit is
fed back as the next observation. The loop ends on DONE or the turn budget.

Policies (how the next message is produced):
  - mock(commands):   replays a fixed command list one per turn, then DONE.
                      Lets us validate the loop end-to-end with no model.
  - anthropic(model): calls the Anthropic Messages API via stdlib urllib
                      (needs ANTHROPIC_API_KEY; honors ANTHROPIC_BASE_URL).

run_model_agent() is called by run.py with a `run_cmd` closure bound to the
task's env/workdir, so agent commands share the same isolated server and are
logged exactly like the reference solution.
"""
import json, os, urllib.request, urllib.error

SYSTEM = (
    "You are operating a Unix shell to complete a tmux task. "
    "You have exactly one tool: a shell. Reply with EITHER a single line "
    "`RUN: <command>` to execute one shell command, OR `DONE` when the task is "
    "complete. Use plain `tmux ...` commands (the server is already running). "
    "Do not explain. One command per reply."
)


def parse_action(text):
    """Return ('done', None) or ('run', cmd) or ('noop', None)."""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.upper().startswith("DONE"):
            return ("done", None)
        if s.startswith("RUN:"):
            return ("run", s[4:].strip())
    # tolerate a bare command with no prefix
    s = text.strip()
    if s and not s.upper().startswith("THINK"):
        return ("run", s.splitlines()[0].strip())
    return ("noop", None)


def run_model_agent(prompt, run_cmd, policy, max_turns=12, obs_limit=1500):
    """Drive `policy` over the task. run_cmd(cmd)->(rc,out,err,timed_out).
    Returns {turns, done, transcript:[...]}."""
    history = []  # alternating ("assistant", action_text), ("user", observation)
    transcript = []
    done = False
    turns = 0
    for turns in range(1, max_turns + 1):
        try:
            reply = policy(SYSTEM, prompt, history)
        except Exception as e:  # policy/endpoint failure -> stop, record it
            transcript.append({"turn": turns, "error": str(e)[:300]})
            break
        kind, cmd = parse_action(reply)
        if kind == "done":
            transcript.append({"turn": turns, "action": "DONE"})
            done = True
            break
        if kind == "noop":
            history.append(("assistant", reply))
            history.append(("user", "No command parsed. Reply `RUN: <cmd>` or `DONE`."))
            transcript.append({"turn": turns, "action": "(noop)", "raw": reply[:200]})
            continue
        rc, out, err, to = run_cmd(cmd)
        obs = (out + (("\n[stderr] " + err) if err.strip() else "")).strip()
        obs = obs[:obs_limit] if obs else "(no output)"
        if to:
            obs = "[timed out] " + obs
        history.append(("assistant", "RUN: " + cmd))
        history.append(("user", f"exit={rc}\n{obs}"))
        transcript.append({"turn": turns, "cmd": cmd, "exit": rc, "obs": obs[:300]})
    return {"turns": turns, "done": done, "transcript": transcript}


# ---------------- tool-calling loop ----------------
# A native function-calling loop. Same task, same grading (the shim still logs
# every tmux command), but the model emits commands as structured tool calls
# instead of `RUN:`/`DONE` text — removing the parsing penalty that the text
# protocol imposes on verbose reasoning models.

SYSTEM_TOOL = (
    "You are operating a Unix shell to complete a tmux task. Use the `shell` tool "
    "to run one command at a time; read each result before the next call. The tmux "
    "server is already running — use plain `tmux ...` commands. When the task is "
    "complete, stop calling tools and reply with a short final message."
)

SHELL_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Run a single shell command and return its stdout, stderr, and exit code.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "one shell command"}},
            "required": ["command"],
        },
    },
}
SHELL_TOOL_ANTHROPIC = {
    "name": "shell",
    "description": "Run a single shell command and return its stdout, stderr, and exit code.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "one shell command"}},
        "required": ["command"],
    },
}


def _obs_text(run_cmd, cmd, obs_limit):
    rc, out, err, to = run_cmd(cmd)
    obs = (out + (("\n[stderr] " + err) if err.strip() else "")).strip()
    obs = obs[:obs_limit] if obs else "(no output)"
    if to:
        obs = "[timed out] " + obs
    return rc, obs


def _http_json(url, headers, body, timeout=120, retries=3):
    """POST JSON; retry transient failures (429 / 5xx / network) with backoff."""
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data,
                                     headers={**headers, "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (408, 409, 429, 500, 502, 503, 504) and attempt < retries:
                continue
            raise
        except Exception as e:  # URLError, timeout, transient socket errors
            last = e
            if attempt < retries:
                continue
            raise
    raise last


def run_openai_tool_agent(prompt, run_cmd, model, base_url=None,
                          key_env="OPENROUTER_API_KEY", max_turns=12, obs_limit=1500):
    """Tool-calling loop over an OpenAI-compatible chat-completions endpoint."""
    key = os.environ.get(key_env) or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(f"{key_env} not set")
    base = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
    headers = {"Authorization": "Bearer " + key,
               "HTTP-Referer": "https://github.com/posix4e/tmuxbench", "X-Title": "tmuxbench"}
    messages = [{"role": "system", "content": SYSTEM_TOOL},
                {"role": "user", "content": prompt}]
    transcript = []
    done = False
    turns = 0
    for turns in range(1, max_turns + 1):
        try:
            data = _http_json(base + "/chat/completions", headers,
                              {"model": model, "max_tokens": 700, "messages": messages,
                               "tools": [SHELL_TOOL_OPENAI], "tool_choice": "auto"})
            msg = data["choices"][0]["message"]
        except Exception as e:
            transcript.append({"turn": turns, "error": str(e)[:300]})
            break
        tcs = msg.get("tool_calls") or []
        # keep the assistant turn in history (content may be null)
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         "tool_calls": tcs} if tcs else
                        {"role": "assistant", "content": msg.get("content") or ""})
        if not tcs:  # no tool call -> the model considers the task done
            transcript.append({"turn": turns, "action": "DONE", "text": (msg.get("content") or "")[:200]})
            done = True
            break
        for tc in tcs:
            fn = tc.get("function", {})
            try:
                cmd = json.loads(fn.get("arguments") or "{}").get("command", "")
            except Exception:
                cmd = ""
            if not cmd:
                messages.append({"role": "tool", "tool_call_id": tc.get("id"),
                                 "content": "error: no command provided"})
                transcript.append({"turn": turns, "action": "(noop)", "raw": str(fn)[:200]})
                continue
            rc, obs = _obs_text(run_cmd, cmd, obs_limit)
            messages.append({"role": "tool", "tool_call_id": tc.get("id"),
                             "content": f"exit={rc}\n{obs}"})
            transcript.append({"turn": turns, "cmd": cmd, "exit": rc, "obs": obs[:300]})
    return {"turns": turns, "done": done, "transcript": transcript}


def run_anthropic_tool_agent(prompt, run_cmd, model, max_turns=12, obs_limit=1500):
    """Tool-calling loop over the Anthropic Messages API."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    messages = [{"role": "user", "content": prompt}]
    transcript = []
    done = False
    turns = 0
    for turns in range(1, max_turns + 1):
        try:
            data = _http_json(base + "/v1/messages", headers,
                              {"model": model, "max_tokens": 700, "system": SYSTEM_TOOL,
                               "messages": messages, "tools": [SHELL_TOOL_ANTHROPIC]})
        except Exception as e:
            transcript.append({"turn": turns, "error": str(e)[:300]})
            break
        blocks = data.get("content", [])
        messages.append({"role": "assistant", "content": blocks})
        tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
        if not tool_uses or data.get("stop_reason") != "tool_use":
            txt = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            transcript.append({"turn": turns, "action": "DONE", "text": txt[:200]})
            done = True
            break
        results = []
        for tu in tool_uses:
            cmd = (tu.get("input") or {}).get("command", "")
            if not cmd:
                results.append({"type": "tool_result", "tool_use_id": tu.get("id"),
                                "content": "error: no command provided"})
                transcript.append({"turn": turns, "action": "(noop)"})
                continue
            rc, obs = _obs_text(run_cmd, cmd, obs_limit)
            results.append({"type": "tool_result", "tool_use_id": tu.get("id"),
                            "content": f"exit={rc}\n{obs}"})
            transcript.append({"turn": turns, "cmd": cmd, "exit": rc, "obs": obs[:300]})
        messages.append({"role": "user", "content": results})
    return {"turns": turns, "done": done, "transcript": transcript}


# ---------------- policies ----------------

def mock_policy(commands):
    """Replay a fixed list of commands, one per turn, then DONE."""
    state = {"i": 0}
    def policy(system, prompt, history):
        i = state["i"]; state["i"] += 1
        if i < len(commands):
            return "RUN: " + commands[i]
        return "DONE"
    return policy


def anthropic_policy(model="claude-haiku-4-5-20251001"):
    """Anthropic Messages API via stdlib urllib (no SDK). Needs ANTHROPIC_API_KEY."""
    def policy(system, prompt, history):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        msgs = [{"role": "user", "content": prompt}]
        for role, text in history:
            msgs.append({"role": "assistant" if role == "assistant" else "user",
                         "content": text})
        body = json.dumps({
            "model": model, "max_tokens": 400, "system": system, "messages": msgs,
        }).encode()
        req = urllib.request.Request(
            base + "/v1/messages", data=body,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.load(r)
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return policy


def openai_policy(model="openai/gpt-4o-mini", base_url=None, key_env="OPENROUTER_API_KEY"):
    """OpenAI-compatible chat completions via stdlib urllib. Works with OpenRouter.
    Reads the key from key_env (or OPENAI_API_KEY); base defaults to OpenRouter."""
    def policy(system, prompt, history):
        key = os.environ.get(key_env) or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(f"{key_env} not set")
        base = (base_url or os.environ.get("OPENAI_BASE_URL")
                or "https://openrouter.ai/api/v1").rstrip("/")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": prompt}]
        for role, text in history:
            messages.append({"role": "assistant" if role == "assistant" else "user",
                             "content": text})
        body = json.dumps({"model": model, "max_tokens": 400, "messages": messages}).encode()
        req = urllib.request.Request(
            base + "/chat/completions", data=body,
            headers={"Authorization": "Bearer " + key, "content-type": "application/json",
                     "HTTP-Referer": "https://github.com/posix4e/tmuxbench",
                     "X-Title": "tmuxbench"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
        return (data["choices"][0]["message"].get("content") or "")
    return policy


def trustedrouter_policy(preset="quality", max_turns_tokens=400):
    """TrustedRouter Synth *fusion* policy (a panel of models + judge + synthesizer).

    Unlike a plain model, fusion is invoked on `trustedrouter/synth` with a
    `trustedrouter:synth` tool carrying the preset (`quality` or `budget`). Calls
    are slow (~30-90s) and occasionally return a transient `fusion failed`, so we
    retry a few times with backoff. Needs TRUSTEDROUTER_API_KEY."""
    def policy(system, prompt, history):
        key = os.environ.get("TRUSTEDROUTER_API_KEY")
        if not key:
            raise RuntimeError("TRUSTEDROUTER_API_KEY not set")
        base = (os.environ.get("TRUSTEDROUTER_BASE_URL")
                or "https://api.trustedrouter.com/v1").rstrip("/")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": prompt}]
        for role, text in history:
            messages.append({"role": "assistant" if role == "assistant" else "user",
                             "content": text})
        body = json.dumps({
            "model": "trustedrouter/synth", "messages": messages,
            "tools": [{"type": "trustedrouter:synth",
                       "parameters": {"preset": preset,
                                      "max_completion_tokens": max_turns_tokens}}],
        }).encode()
        last = None
        for attempt in range(4):  # retry transient 'fusion failed'
            req = urllib.request.Request(
                base + "/chat/completions", data=body,
                headers={"Authorization": "Bearer " + key,
                         "content-type": "application/json", "X-Title": "tmuxbench"})
            try:
                with urllib.request.urlopen(req, timeout=240) as r:
                    data = json.load(r)
                return (data["choices"][0]["message"].get("content") or "")
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:200]
                last = RuntimeError(f"HTTP {e.code}: {detail}")
                if e.code in (400, 429, 502, 503, 504) and attempt < 3:
                    continue  # transient fusion/panel failure -> retry
                raise last
            except Exception as e:
                last = e
                if attempt < 3:
                    continue
                raise
        raise last
    return policy


def commands_from_script(script):
    """Split a reference/agent shell script into one-line commands for the mock."""
    cmds = []
    for line in script.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            cmds.append(s)
    return cmds

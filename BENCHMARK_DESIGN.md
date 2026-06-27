# tmuxbench — Design

A benchmark for evaluating how well LLMs **use** tmux: not "can it recite the
keybinding" but "give it a goal, a shell, and a real tmux server — does it reach
the goal?"

tmux is an unusually good benchmark target because **its entire state is
queryable as format strings** (`tmux display-message -p '#{...}'`). That means
grading is *objective and automated* — we assert on real server state, never on
the wording of the model's answer and never via an LLM judge.

Status: design spec. No harness code yet. All tmux mechanics below were verified
against **tmux 3.6a** on macOS.

---

## 1. Goals & non-goals

**Goals**
- Measure an LLM's ability to drive tmux toward a concrete goal through a shell.
- Cover two pillars (per project decision):
  1. **Agentic task completion** — multi-step session/window/pane/config work.
  2. **Events / hooks / control mode** — the harder, more distinguishing tier.
- Be fully auto-gradable, deterministic, and reproducible.
- Be cheap to extend: adding a task = one self-contained file.

**Non-goals (v1)**
- Knowledge Q&A and one-shot command generation (explicitly out per decision;
  the schema leaves room to add them later as a degenerate 1-turn case).
- Testing tmux itself. We assume a fixed, pinned tmux version.
- GUI/visual rendering correctness beyond what `capture-pane` exposes.

---

## 2. Design principles

1. **State is truth.** A task passes iff the resulting tmux server state matches
   assertions expressed as format strings / observable side-effects. The model's
   prose is irrelevant to scoring.
2. **Isolation per task.** Every task runs on its own tmux server with a unique
   socket and an empty config, so tasks never see the user's tmux, each other,
   or `~/.tmux.conf`. (Verified: `tmux -L <socket> -f /dev/null`.)
3. **Determinism.** No network, no wall-clock-dependent assertions, fixed
   terminal geometry, fixed tmux version. Same model behavior → same score.
4. **No hidden oracles.** Grading uses only tmux's own introspection + files the
   task explicitly writes. No second LLM in the grading path.
5. **Cheap to author, hard to game.** Tasks are declarative; verification is
   independent of the reference solution (any command sequence that produces the
   right state passes).

---

## 3. Architecture

### 3.1 The shell-agent loop (runner)

The model under test is driven like a coding agent: it is given a task prompt and
a single tool — **run a shell command** — and iterates until it declares done or
hits a turn budget.

```
            ┌──────────────────────── per task ────────────────────────┐
 setup.sh ──▶ isolated tmux server (tmux -L bench_<uuid> -f /dev/null)  │
            │                                                           │
 prompt ───▶ ┌─────────────── agent loop (≤ max_turns) ──────────────┐ │
            │ │  model ──emit──▶ bash command                         │ │
            │ │     ▲                │ exec (cwd=sandbox, env scrubbed)│ │
            │ │     └──stdout/stderr/exit─┘                            │ │
            │ │  model emits DONE  ──or──  turn budget exhausted       │ │
            │ └───────────────────────────────────────────────────────┘ │
            │                                                           │
 check.sh ──▶ query final state ──▶ assertions ──▶ score + transcript  │
 teardown ──▶ tmux -L bench_<uuid> kill-server                         │
            └───────────────────────────────────────────────────────────┘
```

Key contract: **the model never sees the socket name or the verification.** It is
told only that "a tmux server is running; use `tmux` normally." The runner
injects the correct `-L <socket>` itself by exposing tmux to the model through a
wrapper on `PATH` (see §3.3) so the model types plain `tmux ...`.

### 3.2 Isolation model (verified)

| Concern | Mechanism |
|---|---|
| Separate server | `tmux -L bench_<uuid>` (unique socket name) |
| No user config | `-f /dev/null` (clean defaults) |
| Fixed geometry | `new-session -x 200 -y 50` (deterministic pane sizes/layouts) |
| No env bleed | run agent commands with scrubbed env + a temp `$HOME`, temp cwd |
| No network | run with network disabled / blocked at the harness level |
| Clean teardown | `tmux -L bench_<uuid> kill-server` in a `trap`/`finally` |

Verified that two servers on different `-L` sockets are fully independent and that
`-f /dev/null` yields stock keybindings/options.

### 3.3 The tmux wrapper (so the model types plain `tmux`)

The runner puts a shim early on `PATH`:

```sh
#!/bin/sh
exec /opt/homebrew/bin/tmux -L "$BENCH_SOCKET" -f /dev/null "$@"
```

So the model's `tmux new-window -t web` is transparently scoped to the task's
server. This keeps prompts natural and prevents the model from accidentally (or
deliberately) touching other servers. The shim also lets us **log every tmux
invocation** the model makes — the basis for the efficiency metric (§6).

### 3.4 Task lifecycle

1. Allocate `BENCH_SOCKET=bench_<uuid>`, temp `$HOME`, temp cwd.
2. Run `setup.sh` (establishes initial state; may create sessions/windows/files).
3. Snapshot initial state (for diff-based checks and transcript).
4. Run the agent loop with `prompt` and the shell tool.
5. Run `check.sh` → emits per-assertion pass/fail + numeric score to JSON.
6. `kill-server`, delete temp dirs, write the run record.

---

## 4. Task taxonomy

Two pillars, each with categories. Every task is tagged with exactly one
category and a difficulty tier (§5).

### Pillar A — Agentic task completion

| Code | Category | Examples |
|---|---|---|
| `A-SESS` | Sessions | create/rename/kill, grouped sessions, detach handling |
| `A-WIN`  | Windows | create/move/swap/renumber, `renumber-windows`, naming |
| `A-PANE` | Panes | split (h/v), resize to exact dims, zoom, break/join-pane |
| `A-LAYOUT` | Layouts | apply named layouts, achieve a target geometry |
| `A-TARGET` | Targeting | `sess:win.pane`, `{last}`/`{next}`/`+`/`-`, `-t` precision |
| `A-SEND` | Input/automation | `send-keys`, run a program in a pane, type & submit |
| `A-CAP` | Output capture | `capture-pane` to find a value, scrollback, `-S`/`-E` ranges |
| `A-BUF` | Buffers | `set-buffer`/`save-buffer`/`paste-buffer`, named buffers |
| `A-OPT` | Options/config | `set-option`/`setw`, prefix change, `bind-key`, key tables |
| `A-INFO` | Introspection | answer a question by querying state (report via a sink) |
| `A-FMT` | Format strings | build a `display-message`/status format with conditionals |
| `A-WORKFLOW` | Multi-step | "set up this dev layout": composite of the above |

### Pillar B — Events / hooks / control mode

| Code | Category | Examples |
|---|---|---|
| `B-HOOK` | Hooks | `set-hook` for `after-*`, `pane-died`, `window-linked`, etc. |
| `B-NOTIFY` | Alerts/monitor | `monitor-activity`, `monitor-silence`, bell, `#{window_flags}` |
| `B-CTRL` | Control mode | drive `-C`/`-CC`, parse `%begin/%end/%output/%layout-change` |
| `B-WAIT` | Synchronization | `wait-for -L/-S/-U` channels as event signals |
| `B-RESPawn` | Lifecycle | `respawn-pane`, `remain-on-exit`, react to exit codes |

**Why B is harder and more distinguishing:** events are stateful and timing
sensitive, the docs are terse, and naive solutions silently don't fire (see §7).

---

## 5. Difficulty tiers

| Tier | Meaning | Typical turns | Example |
|---|---|---|---|
| T1 | Single command, obvious | 1–2 | "Create a detached session named `db`." |
| T2 | A few commands, some targeting | 3–6 | "In session `web`, split window 2 vertically; run `top` in the new pane." |
| T3 | Multi-step or requires introspection/recovery | 6–15 | "Renumber windows to be contiguous from 1, preserving order and the active window." |
| T4 | Events/control-mode/format-heavy; non-obvious mechanics | 10–25 | "Configure activity monitoring so window `logs` flags on output; demonstrate the flag." |

Difficulty drives turn budgets and weighting in the aggregate score.

---

## 6. Metrics

Primary and secondary metrics per task, aggregated per category and overall.

- **Pass rate (primary):** fraction of tasks where *all required* assertions pass.
- **Assertion score:** weighted fraction of assertions passed (gives partial
  credit; some assertions marked `required`, others `bonus`).
- **Efficiency:** number of tmux invocations vs. the reference solution's count
  (`model_cmds / ref_cmds`). Logged via the wrapper shim. Reported, not gating.
- **Turns to completion:** agent-loop iterations used vs. budget.
- **Error recovery:** did the model recover after a command returned nonzero?
  (Counts tasks where ≥1 command errored but the task still passed.)
- **Safety:** flag any attempt to touch a non-bench socket, the network, or
  paths outside the sandbox (should be impossible, but we detect and zero-score).

Aggregate score = mean assertion score weighted by tier (e.g. T1=1, T2=2,
T3=3, T4=4), reported overall and per category, with pass-rate alongside.

---

## 7. Events / hooks / control mode — special handling

This pillar needs care; verified gotchas:

### 7.1 Hooks may not *fire their action* in a scripted/detached server
Observed: `set-hook pane-died 'rename-window dead'` + `remain-on-exit on` +
killing a pane → `#{pane_dead}` correctly became `1`, **but the window was not
renamed** (the hook action didn't run without the normal client event loop).

**Design rule:** grade hooks with a *layered* strategy:
1. **Registration check (always):** the hook/binding is correctly configured —
   `tmux show-hooks -g` / `show-options` / `list-keys` confirms the exact action.
2. **Behavioral check (when feasible):** trigger the condition and observe a
   side-effect that *does* survive scripting — preferably a **file/env sink**:
   ```
   set-hook -t s1 pane-died 'run-shell "echo fired >> $SINK"'
   ```
   then assert the sink. Where the action must mutate tmux state, run the trigger
   **with a client attached** (`tmux -C attach` in the background under
   `timeout`) so the event loop runs, then assert state.
3. Fall back to **state polling** of the raw condition (`#{pane_dead}`,
   `#{window_activity}`, `#{window_flags}`) to confirm the event *happened*,
   independent of whether the model's action ran.

Tasks should prefer hook actions that write to a sink — it's the most robust,
fully deterministic signal.

### 7.2 Control-mode `attach` holds the connection open
Observed: `printf 'list-sessions\n' | tmux -C attach | head` **hung for 2
minutes** — `attach` doesn't exit on EOF.

**Design rule for `B-CTRL`:** always wrap control-mode interaction in `timeout`
and send an explicit `detach`/`kill-session`:
```sh
timeout 5 sh -c "printf 'list-windows\ndetach\n' | tmux -C attach -t s1"
```
Grading parses the control-mode protocol stream (`%begin/%end`, `%output`,
`%window-add`, `%layout-change`, `%session-changed`) the model captured and
reports. Verified this emits e.g. `%begin … %end` and `%session-changed $0 s1`.

### 7.3 `wait-for` as a deterministic event signal (verified)
`wait-for -L <chan>` blocks until `wait-for -S <chan>`; works across processes.
Use it to make event tasks deterministic instead of `sleep`:
- Setup arms a hook: `... 'wait-for -S done'`.
- Trigger the event, then `wait-for -L done` (under `timeout`) to synchronize
  before asserting — no race, no arbitrary sleeps.

### 7.4 `remain-on-exit` to observe pane death
A pane with `remain-on-exit on` stays as a dead pane after its process exits, so
`#{pane_dead}`, `#{pane_dead_status}` (exit code), and `respawn-pane` are all
observable/gradable.

---

## 8. Task schema

Each task is a self-contained directory `tasks/<id>/` (or a single YAML for
simple ones). Fields:

```yaml
id: A-PANE-003                 # unique, category-prefixed
category: A-PANE               # taxonomy code (§4)
tier: T2                       # difficulty (§5)
title: "Split and run top in the right pane"
prompt: |                      # exactly what the model is told (natural language)
  A tmux server is running. In session "web", select window "editor",
  split it into two side-by-side panes, and run `top` in the right-hand pane.
  Leave the left pane active.

env:                           # optional runtime knobs
  width: 200
  height: 50
  max_turns: 8

setup: |                       # shell; establishes initial state. $SINK, $HOME temp.
  tmux new-session -d -s web -x 200 -y 50
  tmux rename-window -t web:0 editor

# Verification: ordered list of assertions. Each is a tiny, independent check.
# `query` runs and its trimmed stdout is compared to `expect` per `op`.
checks:
  - id: two_panes
    required: true
    weight: 1
    query: "tmux display-message -p -t web:editor '#{window_panes}'"
    op: eq
    expect: "2"
  - id: side_by_side          # both panes same height, different x → horizontal split
    required: true
    weight: 1
    query: "tmux list-panes -t web:editor -F '#{pane_height}' | sort -u | wc -l | tr -d ' '"
    op: eq
    expect: "1"
  - id: top_running_right
    required: true
    weight: 2
    # right pane = the one with the larger pane_left; assert its command is top
    query: |
      tmux list-panes -t web:editor -F '#{pane_left} #{pane_current_command}' \
        | sort -n | tail -1 | awk '{print $2}'
    op: in           # top may appear as "top"
    expect: ["top"]
  - id: left_active
    required: false        # bonus
    weight: 1
    query: "tmux display-message -p -t web:editor '#{pane_at_left}#{?pane_active,A,_}'"
    op: regex
    expect: "1A"           # active pane is the leftmost

reference: |              # canonical solution; used only for efficiency baseline + docs
  tmux split-window -h -t web:editor 'top'
  tmux select-pane -t web:editor -L

teardown: |              # optional; runner always kill-servers regardless
  true

tags: [split, send-keys, targeting]
```

**Assertion `op`s:** `eq`, `ne`, `lt`/`le`/`gt`/`ge` (numeric), `regex`,
`in` (stdout ∈ list), `contains`, `nonempty`. All compare against trimmed stdout
of `query`, which is just shell (so any tmux/awk/grep pipeline is allowed).

This design keeps **verification independent of the reference solution** — any
sequence of commands that yields the right state passes.

---

## 9. Worked examples

### 9.1 `A-SESS-001` (T1) — create a detached session
```yaml
prompt: "Create a new detached tmux session named `db`. Do not attach to it."
setup: |
  true
checks:
  - id: exists
    required: true
    query: "tmux has-session -t db 2>/dev/null && echo yes || echo no"
    op: eq
    expect: "yes"
reference: "tmux new-session -d -s db"
```

### 9.2 `A-WIN-004` (T3) — renumber windows contiguously, keep order & active
```yaml
prompt: |
  Session "proj" has windows at indices 0, 3, 7 (in that order). Renumber them
  to 0, 1, 2 keeping their order and names, and keep whatever window was active
  still active afterward.
setup: |
  tmux new-session -d -s proj -x 200 -y 50
  tmux rename-window -t proj:0 a
  tmux new-window -t proj:3 -n b
  tmux new-window -t proj:7 -n c
  tmux select-window -t proj:3        # b is active
checks:
  - id: indices
    required: true
    query: "tmux list-windows -t proj -F '#{window_index}' | paste -sd, -"
    op: eq
    expect: "0,1,2"
  - id: order_preserved
    required: true
    query: "tmux list-windows -t proj -F '#{window_name}' | paste -sd, -"
    op: eq
    expect: "a,b,c"
  - id: active_preserved
    required: true
    query: "tmux display-message -p -t proj '#{window_name}'"  # active window
    op: eq
    expect: "b"
reference: |
  tmux set-option -t proj renumber-windows on
  tmux move-window -r -t proj            # renumber in place
```
*(Note the trap: a naive `move-window` per window can collide indices or change
the active window — this is what T3 is probing.)*

### 9.3 `A-CAP-002` (T2) — extract a value from pane output
```yaml
prompt: |
  In session "calc", a shell is running in the only pane. Compute 6 * 7 in that
  shell and tell me the result by writing it (digits only) to the file at
  $REPORT.
env: { report_var: REPORT }
setup: |
  tmux new-session -d -s calc -x 200 -y 50
checks:
  - id: answer
    required: true
    query: "cat \"$REPORT\""
    op: eq
    expect: "42"
reference: |
  tmux send-keys -t calc 'expr 6 \* 7' Enter
  sleep 0.3
  tmux capture-pane -p -t calc | grep -E '^[0-9]+$' | tail -1 > "$REPORT"
```
*(Probes `send-keys` + `capture-pane` + parsing. The sink file decouples grading
from how the model reports.)*

### 9.4 `A-FMT-001` (T3) — author a format string with a conditional
```yaml
prompt: |
  Using a single `tmux display-message -p` call, print a line in the form
  "<session>:<windows>[<attached?>]" where the bracket contains "A" if the
  session has any attached client, else "-". Write the output to $REPORT.
  The current session is "fmt".
setup: |
  tmux new-session -d -s fmt -x 200 -y 50
checks:
  - id: format
    required: true
    query: "cat \"$REPORT\""
    op: regex
    expect: "^fmt:1\\[-\\]$"        # detached server → "-"
reference: |
  tmux display-message -p -t fmt '#{session_name}:#{session_windows}[#{?session_attached,A,-}]' > "$REPORT"
```

### 9.5 `B-HOOK-001` (T4) — hook writes to a sink on pane death (robust grading)
```yaml
prompt: |
  In session "svc", configure tmux so that whenever a pane in that session dies,
  the text "PANE_DIED" is appended to the file at $SINK. Then prove it works by
  causing a pane to die.
env: { sink_var: SINK }
setup: |
  tmux new-session -d -s svc -x 200 -y 50
  tmux set-window-option -g remain-on-exit on
checks:
  - id: hook_registered
    required: true
    query: "tmux show-hooks -t svc 2>/dev/null | grep -c pane-died"
    op: ge
    expect: "1"
  - id: hook_fired
    required: true
    query: "grep -c PANE_DIED \"$SINK\" 2>/dev/null || echo 0"
    op: ge
    expect: "1"
reference: |
  tmux set-hook -t svc pane-died "run-shell 'echo PANE_DIED >> $SINK'"
  tmux new-window -t svc 'false'   # process exits → pane dies → hook fires
```
*(Robust because the sink survives scripting per §7.1; layered registration +
behavioral checks.)*

### 9.6 `B-CTRL-001` (T4) — drive control mode and report a notification
```yaml
prompt: |
  Using tmux control mode (`tmux -C`), attach to session "cm", create a new
  window named "fromctrl", then detach. Capture the control-mode output stream
  and write it to $REPORT.
setup: |
  tmux new-session -d -s cm -x 200 -y 50
checks:
  - id: window_created
    required: true
    query: "tmux list-windows -t cm -F '#{window_name}' | grep -c '^fromctrl$'"
    op: eq
    expect: "1"
  - id: protocol_captured
    required: false
    query: "grep -c '%window-add\\|%begin' \"$REPORT\" 2>/dev/null || echo 0"
    op: ge
    expect: "1"
reference: |
  timeout 5 sh -c "printf 'new-window -n fromctrl\ndetach\n' | tmux -C attach -t cm" > "$REPORT" 2>&1
```
*(Probes the §7.2 gotcha: the model must bound the control-mode session or it
hangs.)*

### 9.7 `B-WAIT-001` (T3) — use a wait-for channel to synchronize
```yaml
prompt: |
  Run the command `sleep 0.2; echo done` in a pane of session "sync". Use a
  tmux wait-for channel named "ch" so that when the command finishes it signals
  the channel, and confirm by blocking on the channel and then writing "OK" to
  $REPORT.
setup: |
  tmux new-session -d -s sync -x 200 -y 50
checks:
  - id: signalled
    required: true
    query: "cat \"$REPORT\""
    op: eq
    expect: "OK"
reference: |
  tmux send-keys -t sync 'sleep 0.2; echo done; tmux wait-for -S ch' Enter
  tmux wait-for -L ch && echo OK > "$REPORT"
```

---

## 10. Anti-cheating & robustness

- **Socket scoping:** the wrapper forces `-L bench_<uuid>`; commands targeting
  another socket (`tmux -L something_else`) are blocked/flagged by the wrapper.
- **Verification independence:** checks never read the model's prose and never
  pattern-match the reference solution — only real state / sinks. So "looks
  right" can't pass; only "is right" passes.
- **No-op detection:** snapshot initial state; if final == initial for a task
  that requires change, auto-fail with a clear reason.
- **Network/file sandbox:** scrubbed env, temp `$HOME`, temp cwd, network off.
  Attempts to escape are detected and zero-scored.
- **Idempotent setup/teardown:** every run starts from `kill-server` clean; the
  `trap` guarantees teardown even on harness error.
- **Pinned tmux:** record `#{version}` (verified `3.6a`) in every run record;
  refuse to score across versions unless explicitly allowed.

---

## 11. Reporting

Per run, emit `results/<model>/<timestamp>/`:
- `summary.json` — overall + per-category aggregate score, pass rate, efficiency.
- `tasks/<id>.json` — per-assertion results, turns used, commands issued,
  errors encountered + recovered, final-state snapshot.
- `transcript/<id>.txt` — full agent loop (model commands + outputs) for audit.

```jsonc
// summary.json (shape)
{
  "model": "…", "tmux_version": "3.6a", "tasks": 42,
  "overall": { "pass_rate": 0.71, "assertion_score": 0.83, "weighted": 0.78 },
  "by_category": { "A-PANE": {"pass_rate": 0.9, ...}, "B-HOOK": {...} },
  "efficiency_median": 1.4   // model cmds / reference cmds
}
```

---

## 12. Repository layout

```
tmuxbench/
  BENCHMARK_DESIGN.md      ← this file
  harness/
    run.sh                 ← per-task lifecycle (isolate→setup→agent→check→teardown)
    tmux-shim              ← PATH wrapper that injects -L/-f and logs commands
    grade.py               ← evaluates checks[] against state, writes JSON
    agent/                 ← model clients (shell-agent loop); pluggable per model
  tasks/
    A-SESS-001/task.yaml
    A-PANE-003/task.yaml
    B-HOOK-001/task.yaml
    ...
  schema/task.schema.json  ← JSON Schema for task.yaml (CI-validated)
  results/<model>/<ts>/...
```

---

## 13. Build roadmap

1. **M0 — harness skeleton:** isolation + shim + a trivial single-turn "runner"
   that executes a hardcoded command list, proving setup→check→teardown end to
   end on `A-SESS-001`.
2. **M1 — grader:** `grade.py` implementing all `op`s + JSON output + schema.
3. **M2 — shell-agent loop:** real model client (bash tool, turn budget,
   DONE detection, transcript). One reference model wired in.
4. **M3 — seed task set:** ~8–12 tasks spanning A-* and the §7 B-* gotchas
   (the worked examples in §9 are the starting set).
5. **M4 — events hardening:** wait-for synchronization, control-mode timeout
   wrapping, hook sink pattern as reusable helpers.
6. **M5 — scale & report:** 30+ tasks, aggregate reporting, efficiency baseline,
   multi-model comparison.

---

## 14. Open questions

- **Send-keys timing:** capture tasks need a settle delay; standardize on
  `wait-for` where possible, else a fixed small sleep encoded in `check.sh`
  (kept out of the model's path so it can't be the failure cause).
- **Multiple valid layouts:** for geometry tasks, assert *invariants*
  (pane count, relative position, sizes) rather than exact `window_layout`
  checksums, which over-constrain.
- **Model "done" signal:** explicit sentinel (model prints `BENCHMARK_DONE`) vs.
  letting it run to budget. Recommend explicit sentinel + budget backstop.
- **Partial credit policy:** which assertions are `required` vs `bonus` per task
  — needs a per-category convention.
- **tmux version matrix:** pin 3.6a for v1; later, optionally run a 3.3a/3.4
  matrix since format vars and hooks differ across versions.
```

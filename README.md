# tmuxbench

**Why LLM agents underperform on interactive control planes.** They miss
asynchronous events, measured on real tmux state:

- **Missing events** — the agent fails to perceive/act on asynchronous signals
  (a sub-agent pane dies, a job finishes, a bell fires, an interrupt arrives)
  because it polls at the wrong time, never arms monitoring, or is busy in
  another pane. tmuxbench grades this directly as a per-type **missed-event rate**.

The leaderboard is observational — it shows *that* agents miss events, not *why*.
Candidate causes (scarce async training data, long-horizon degradation, model
capability, and the agent scaffold) are discussed but not claimed; §6.3 shows the
deficit is jointly governed by capability and scaffolding. See `paper/PAPER.md`.

### → Live explainer: https://posix4e.github.io/tmuxbench/

An animated, single-page walkthrough (styled as a tmux session) of the failure,
the benchmark, the 21-model result, and the reading list — with a "missing events"
demo.

## Why tmux

tmux's entire state is queryable (`tmux display-message -p '#{...}'`), so grading
is objective and automated — we assert on the real control plane, never on what
the model says. It is natively asynchronous (hooks, `wait-for`, background
processes) and its panes are a substrate for an agent to orchestrate sub-agents.
That makes it a near-ideal instrument for the **missed-event rate** — did the agent
perceive and act on each asynchronous signal?

## Contents

| Path | What |
|---|---|
| [`index.html`](index.html) | The animated explainer (the Pages site). |
| [`BENCHMARK_DESIGN.md`](BENCHMARK_DESIGN.md) | Instrument design (tmux 3.6a): isolation, command-logging shim, shell-agent loop, state-based grading, task pillars A/B/C, verified gotchas. |
| [`harness/`](harness/) | `tmux-shim`, `run.py` (runner + grader, text **and** tool loops), `cliff.py` (generalization-cliff task generator), `cliff_sweep.py`/`cliff_agg.py`/`cliff_h100.sh` (cliff sweep + size-ladder driver). |
| [`tasks/`](tasks/) | Seed tasks (A solo, B events, C orchestration). |
| [`data/`](data/) | Raw cliff run records (size-ladder + frontier) + the result table. |
| [`paper/PAPER.md`](paper/PAPER.md) | The write-up: framework, methodology, results. |
| [`paper/refs.bib`](paper/refs.bib) | Bibliography (arXiv). |

### The generalization cliff

`harness/cliff.py` holds a trivial task constant (retrieve N tokens → a file) and
scales tmux **indirection depth** (D0 files → D1 option → D2 capture-pane → D3
targeted → D4 respawn) × **op-count**, then sweeps a small-open-model size ladder
(Qwen2.5 0.5–14B on an H100) + frontier models. Result: the cliff marches right
with capability — a single tmux indirection (D0→D1) collapses every model ≤3B
("tmux tax"), `capture-pane` (D2) is a wall for every open model, and respawn/event
handling (D4) walls everyone including gpt-4.1. See [`data/README.md`](data/README.md).

## Run the benchmark

```sh
python3 harness/run.py --all          # run all tasks/*/task.toml, grade on real state
python3 harness/run.py tasks/B-WAIT-001 --json

# Drive a real model — native tool-calling loop (recommended) or text RUN:/DONE loop:
OPENROUTER_API_KEY=sk-... python3 harness/run.py --all --agent model --policy openrouter --model <id> --loop tool
ANTHROPIC_API_KEY=sk-... python3 harness/run.py --all --agent model --policy anthropic --model <id> --loop tool

# Validate the loop with no model (replays the reference one command per turn):
python3 harness/run.py tasks/B-DIE-001 --agent model --policy mock
```

Requires Python ≥ 3.11 (stdlib `tomllib`) and tmux on `PATH`. Each task runs on
an isolated server (`tmux -L <uuid> -f /dev/null`); a shim exposed as `tmux`
scopes the socket and logs every command. The agent loop
([`harness/agent.py`](harness/agent.py)) is dependency-free (stdlib `urllib`);
event tasks report a per-type **missed-event rate** (did the agent perceive and
act on the asynchronous signal?).

## Status

Framework, explainer, harness (isolation + shim + grader + text **and**
tool-calling model loops), 11 seed tasks, and the **missed-event metric** are
done. A **21-model pilot across two independent agent protocols** shows the
headline result holds either way: the native **tool-calling** loop (5,313 runs,
11 tasks) gives **20/21 models worse on events than synchronous tasks** (mean
64% → 28%), and the **text** `RUN:`/`DONE` loop (9,334 runs) gives **21/21** (mean
71% → 21%, missed-event 0.88). The loop you use shifts cross-model *sync* ranking
(gpt-5 41%→52%, gemini-2.5-pro 60%→90% from text→tool) but **not** the within-model
sync→events gap — that's the load-bearing result. A **multi-model fusion system**
(TrustedRouter Synth, 180 runs via `--policy trustedrouter`) also fails to escape
it (same drop; pane death `B-DIE` 5/20). But a **mature scaffold (mini-swe-agent) +
frontier models** (sonnet-4.5, gpt-5.2, glm-5.2) largely closes it (34/36) — so the
deficit is jointly governed by capability and scaffolding, with pane death the
hard residue. Disentangling the two (frontier models in the minimal loop; the
mature scaffold over weak models) is the open follow-up.

---

*Graded on `#{real_state}`, not on prose.*

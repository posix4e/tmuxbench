---
title: "Missing Events: Why LLM Agents Underperform on Interactive Control Planes"
author:
  - Anonymous
date: 2026
abstract: |
  Interactive control planes — terminal multiplexers (tmux), remote shells
  (ssh), and an agent orchestrating sub-agents — are persistent, stateful, and
  event-driven. We introduce **tmuxbench**, a hermetic, fully auto-gradable
  benchmark whose state is entirely queryable (`tmux display-message -p
  '#{...}'`), and use it to study a behavioral failure we call **missing
  events**: the agent fails to perceive or act on asynchronous signals (a
  sub-agent pane dies, a long job finishes, a bell fires, an interrupt arrives)
  because it polls at the wrong time, never arms monitoring, or is busy
  elsewhere. tmuxbench measures this directly via a per-event-type **missed-event
  rate**, graded on real tmux state rather than model prose. Across **two
  independent agent protocols** — a native tool-calling loop (5,313 runs) and a
  text loop (9,334 runs), 21 models — models are **systematically worse on events
  than on synchronous tasks** (20/21 and 21/21; mean gaps of 36 and 50 points;
  missed-event rates up to 0.88); a multi-model **fusion** system does not escape
  it. The failure is near-universal under a minimal agent harness and survives both
  a protocol change and ensembling — yet a stronger **native** terminal-control
  harness with the newest frontier models (claude-sonnet-4.5, gpt-5.2, glm-5.2)
  largely closes it (**34/36**), leaving pane-death detection as the robust
  residue. The deficit is thus a function of both capability and scaffolding, and
  tmuxbench is sensitive enough to separate them and to localize the events that
  stay hard.
---

# 1. Introduction

Reinforcement learning is now the dominant post-training recipe for LLM agents,
and there is strong evidence it reshapes *tool use* in synchronous, single-turn
settings — which tools are called, when, and how often [@agentrl_scaling_law_2025;
@agenticrl_survey_2025]. But real agentic work increasingly runs on **interactive
control planes**: environments that persist across turns, hold state the agent
must track, emit asynchronous events, and support delegation to other workers. A
terminal multiplexer is the canonical example — tmux keeps sessions alive, fires
hooks and alerts, runs background processes, and lets one agent drive sub-agents
across panes. ssh extends this across machines; "an agent controlling another
agent" makes the delegation explicit.

Three findings say this regime is important and unsolved. **Asynchrony breaks
agents:** ReAct with GPT-4o scores 47% on synchronous tasks but **11%** on
asynchronous ones [@robotouille_2025], and mid-task interruptions defeat frontier
models [@interruptbench_2026]. **Stateful delegation collapses:** subtask
accuracy of 78% drops to **50%** at the task level, with agents leaving
"persistent wrong state" [@entcollabbench_2026]. **Curated terminal benchmarks
under-sample real use:** TerminalWorld correlates only *r* = 0.20 with
Terminal-Bench [@terminalworld_2026], and Terminal-Bench itself uses a single
headless shell with no multiplexing, async, or sub-agents [@terminalbench_2026].

Our object of study is **missing events**: a behavioral failure we can measure
end-to-end on real state. The control plane emits asynchronous signals the agent
must notice and act on, and agents miss them — they poll at the wrong moment,
never arm monitoring (`set-hook`, `monitor-activity`, `wait-for`), or are occupied
in another pane. tmux makes every such event queryable (`#{pane_dead}`,
`#{window_activity}`, `#{window_flags}`, wait-for channels), so a **missed-event
rate** is cleanly gradable.

**What we claim, and what we don't.** The tmuxbench leaderboard (§6.1) is
*observational*: it shows *that* agents miss events, and that the deficit is
near-universal and survives ensembling, but it does not pin down *why* — scarce
async/event training data, long-horizon context degradation, model capability, and
the agent scaffold itself are all live candidate explanations. We do not adjudicate
among them here; §6.3 and §7 show the deficit is jointly governed by capability and
scaffolding, and we make **no causal claim** about its origin.

**Contributions.**

1. A **framing** (§3) of missing events as a distinct, gradable failure of the
   perceive/synchronize loop on interactive control planes, and why these
   environments provoke it.
2. **tmuxbench** (§4): a hermetic, fully auto-gradable instrument with solo (A),
   event-driven (B), and orchestration (C) pillars, an explicit **event
   taxonomy**, and a **missed-event rate** — grading real tmux state, never model
   prose.
3. A **multi-thousand-run, 21-model** result across two agent protocols, plus a
   multi-model **fusion** comparison and a frontier-model native-harness sweep
   (§6): the missing-events deficit is near-universal under a minimal scaffold,
   survives a protocol change and ensembling, and is largely closed by the
   strongest models given a mature scaffold — so the metric separates
   capability-and-scaffolding from the genuinely hard cases.

# 2. Related Work

## 2.1 Missing / asynchronous events
Robotouille shows the sync→async collapse (47%→11%) and frames the demands —
"manage overlapping tasks and interruptions" [@robotouille_2025]. InterruptBench
isolates mid-task interruption [@interruptbench_2026]. AsyncLM engineers an
interrupt mechanism for in-flight tool returns and notes that *fine-tuning is
needed even to handle interrupt semantics* [@asynclm_2024] — evidence that
post-training, not just scaffolding, governs async handling. **Gap:** these probe
*whether* agents fail asynchronously; none use a control plane whose every event
is queryable to attribute failures to specific missed event types.

## 2.2 Supporting: terminal benchmarks and stateful delegation
Terminal-Bench [@terminalbench_2026], TerminalWorld [@terminalworld_2026], and
the enterprise multi-agent collapse [@entcollabbench_2026] establish that
single-shell, synchronous, stateless-delegation evaluation misses the regime we
target. Agentic-RL reshapes tool use in synchronous domains
[@agentrl_scaling_law_2025; @verltool_2025; @agenticrl_survey_2025], which is the
behavior whose transfer to async/stateful control we question.

# 3. Framework: missing events

A control plane requires four things over a long horizon: **perceive** state and
events, **decide**, **act**, and **synchronize** with concurrent processes.
**Missing events** is a failure of the *perceive/synchronize* links: the signal is
present and queryable, but the agent does not look, looks at the wrong time, or
never arms a mechanism (hook / monitor / channel) that would surface it. It is a
*behavioral* property of the deployed policy, directly gradable on real tmux state,
and we measure it across **all** models — closed and open (§6.1).

Difficulty is governed by three **amplifiers** we vary in the design (§5.2):
trajectory **length**, **asynchrony** (number/spacing of events), and **delegation
depth** (sub-agents in panes). A longer, more asynchronous, more delegated
trajectory both presents more events to miss and places the agent further from the
synchronous, single-turn regime that dominates agent post-training — a plausible
reason event competence is fragile, though we do not attempt to attribute the
deficit to a single cause (§7).

Why tmux is the right probe. (i) *Ground-truth observability* — complete state
and every event are queryable, so we grade the real control plane, not prose, and
can pinpoint *which* event was missed. (ii) *Native asynchrony/statefulness* —
hooks, alerts, background processes, and `wait-for` make it a genuine event-driven
plane. (iii) *Hermetic isolation* — `tmux -L <uuid> -f /dev/null` gives a
per-task server. (iv) *Separable amplifiers* — length, asynchrony, and delegation
toggle independently. (v) *Auditable trajectories* — the command-logging shim
yields an exact, replayable token stream (useful for any later mechanistic probe).

# 4. The tmuxbench Instrument

tmuxbench drives a model like a coding agent — one shell tool — against an
isolated tmux server, then grades the resulting state. Full mechanics are in the
companion `BENCHMARK_DESIGN.md`; key points:

## 4.1 Isolation, shim, loop, grading
Each task runs on its own server (`tmux -L bench_<uuid> -f /dev/null`) with a
temp `$HOME`, temp cwd, and network off. A shim exposed as `tmux` on `PATH`
scopes the socket and **logs every invocation** (the basis for the efficiency
metric and an auditable, replayable trajectory). The model loops emit-command → observe →
continue until a `DONE` sentinel or turn budget. Grading runs each task's
`checks` (shell `query` compared to an expectation under an operator: `eq`,
numeric, `regex`, `in`, `contains`, `nonempty`) against real state via
`display-message -p '#{...}'`, `capture-pane`, and file sinks. **Verification is
independent of the reference solution** — only correct state passes.

## 4.2 Verified pitfalls
Probing tmux 3.6a surfaced three traps an honest events benchmark must handle:
hooks often do **not** fire their action in a scripted server (grade via file
sink + registration check + `#{pane_dead}` polling); control-mode `attach` holds
the pipe open (wrap in `timeout` + explicit `detach`); `wait-for` gives race-free
synchronization (no brittle `sleep`s).

## 4.3 Pillars and the event taxonomy
- **A — solo (synchronous):** sessions/windows/panes/formats; doubles as a
  tmux-fluency baseline to subtract (§5.4).
- **B — events (asynchronous):** the event taxonomy graded for a **missed-event
  rate** — pane/sub-agent **death** (`#{pane_dead}` + exit status),
  **activity/silence** (`monitor-activity`, `monitor-silence`, `#{window_flags}`),
  **bell/alert**, **job completion** (`wait-for`), and **interrupt/refocus** (a
  higher-priority signal mid-task).
- **C — orchestration:** an agent spawns sub-agent loops in panes, coordinates,
  and aggregates — fan-out/fan-in, producer/consumer, failure-injection, context
  handoff.

# 5. Methodology

## 5.1 Hypotheses
- **H1 (missing events scale with asynchrony).** Missed-event rate rises with the
  number and density of concurrent events, and is higher when the agent is
  occupied in another pane (orchestration).
- **H2 (events are an armed-monitoring competence).** Agents that proactively arm
  monitoring (hooks/`wait-for`) have far lower missed-event rates than agents that
  poll; the gap widens with trajectory length.

## 5.2 Conditions
Three amplifiers crossed with task pillar: **length** (short/long), **asynchrony**
(none/few/many events), **delegation depth** (solo / 1 level / nested). Each
isolates one driver of the missing-events failure.

## 5.3 The measurement
**Missed-event rate (behavioral).** For each fired event, grade whether the agent
*perceived and acted on it* — armed the right monitoring, read the signal via a
valid path, and reacted before proceeding — not merely whether the end state is
right. Reported per event type, plus the aggregate event-handling correctness.

## 5.4 Metrics and controls
Metrics: pass rate, weighted assertion score, efficiency, turns, **missed-event
rate** (per type), and event-handling correctness. Controls: subtract the pillar-A
fluency baseline so event effects are not raw tmux proficiency; ≥2 prompt
paraphrases, no per-model tuning; ≥3 seeds with flake gating; deterministic event
firing (`wait-for`, `remain-on-exit`); turn-budget exhaustion reported as a
distinct failure mode.

# 6. Results

## 6.1 Pilot: events are systematically harder than synchronous tasks

We ran the shell-agent loop (§4) over the 11-task suite — 5 synchronous (pillar A:
session, pane-split, format-string, capture, kill-window), 5 event-driven (pillar
B: sub-agent death+respawn, background-activity, `wait-for` sync, bell, interrupt)
and 1 orchestration (pillar C: fan-out/aggregate) — across **21 models** spanning
$0.02–$3.00/M, served via OpenRouter, graded on real tmux state. To guard against
an agent-protocol artifact we run **two independent loops**: a native
**tool-calling** loop (primary) and a minimal `RUN:`/`DONE` **text** loop
(cross-check).

**Primary (tool-calling, 11 tasks, 5,313 runs, ~253/model).** Mean **64%**
synchronous vs **28%** event pass — a **36-point** within-model gap — and
**20/21** models score lower on events than on synchronous tasks (the lone
exception, `gpt-5`, reaches parity at 52%/50%). Per-model pass rate with bootstrap
95% CIs:

| Model | runs | Pass% (95% CI) | A·sync | B·events | C·orch | missed-event |
|---|--:|--:|--:|--:|--:|--:|
| deepseek/deepseek-chat-v3.1 | 253 | 85% [80–89] | 97% | 75% | 78% | 0.25 |
| anthropic/claude-sonnet-4 | 253 | 78% [73–83] | 100% | 51% | 100% | 0.60 |
| openai/gpt-4o | 253 | 77% [71–82] | 92% | 57% | 96% | 0.43 |
| google/gemini-2.5-pro | 253 | 67% [61–73] | 90% | 41% | 87% | 0.51 |
| deepseek/deepseek-chat | 253 | 65% [59–71] | 80% | 44% | 96% | 0.63 |
| openai/gpt-4.1 | 253 | 62% [57–68] | 76% | 44% | 87% | 0.54 |
| openai/gpt-5-mini | 253 | 60% [54–66] | 77% | 55% | 4% | 0.32 |
| qwen/qwen-2.5-72b-instruct | 253 | 60% [55–66] | 76% | 44% | 61% | 0.67 |
| mistralai/mistral-small-3.2-24b | 253 | 50% [43–56] | 73% | 25% | 57% | 0.78 |
| google/gemini-2.5-flash | 253 | 48% [42–54] | 64% | 22% | 96% | 0.86 |
| openai/gpt-5 | 253 | 46% [40–53] | 52% | 50% | 0% | 0.38 |
| openai/gpt-4o-mini | 253 | 42% [36–49] | 76% | 10% | 35% | 0.89 |
| openai/o4-mini | 253 | 42% [36–47] | 72% | 19% | 0% | 0.76 |
| meta-llama/llama-3.3-70b-instruct | 253 | 39% [33–45] | 63% | 22% | 0% | 0.97 |
| deepseek/deepseek-r1 | 253 | 36% [30–42] | 62% | 17% | 4% | 0.80 |
| qwen/qwen3-32b | 253 | 27% [22–33] | 57% | 2% | 4% | 0.98 |
| google/gemini-2.5-flash-lite | 253 | 26% [21–31] | 49% | 7% | 4% | 0.92 |
| mistralai/mistral-nemo | 253 | 23% [18–28] | 46% | 3% | 0% | 0.99 |
| meta-llama/llama-3.1-8b-instruct | 253 | 16% [12–21] | 25% | 11% | 0% | 1.00 |
| qwen/qwen-2.5-7b-instruct | 253 | 8% [5–12] | 18% | 0% | 0% | 1.00 |
| meta-llama/llama-3.2-3b-instruct | 253 | 0% [0–0] | 0% | 0% | 0% | 1.00 |

**Cross-check (text loop, 9 tasks, 9,334 runs).** The same shape, stronger:
**21/21** models worse on events, mean **71% → 21%**, missed-event **0.88**. That
the gap survives a completely different agent protocol is the point — it is not an
artifact of how the agent emits commands.

**The protocol does move cross-model ranking, not the gap.** Switching loops
changes individual synchronous scores substantially, which is exactly why we did
not trust the text-loop ranking:

| Model | A·sync (text) | A·sync (tool) |
|---|--:|--:|
| openai/gpt-5 | 41% | 52% |
| openai/gpt-5-mini | 66% | 77% |
| google/gemini-2.5-pro | 60% | 90% |
| openai/o4-mini | 88% | 72% |
| qwen/qwen3-32b | 72% | 57% |
| anthropic/claude-sonnet-4 | 99% | 100% |

The text loop penalized verbose reasoning models (gpt-5, gpt-5-mini) and flattered
others; the tool loop removes that parsing penalty. But under *both*, every model
(20/21 and 21/21) drops from sync to events — the **within-model sync→events gap**
is the load-bearing, protocol-independent result.

Failures are concrete, not grader artifacts (reference solutions pass every task):
models call `respawn-pane` without first reading `#{pane_dead}` and reuse the
dying command (the pane re-dies); they use `wait-for` with no `-S`/`-L` flags and
time out; they never read `#{window_activity_flag}`/`#{window_bell_flag}` and guess
the active window; they finish a task while a sub-agent pane sits blocked on input.

## 6.2 A multi-model fusion system does not escape it

To test whether the deficit is a single-model limitation, we ran **TrustedRouter
Synth** — a fusion system that polls a *panel* of models, has a judge weigh them,
and a synthesizer write the final answer — through the text loop on the 9-task
suite (**180 runs**, both live presets, `budget` and `quality`). Fusion scores
among the strongest configurations we tested (overall 82–86%), yet reproduces the
same signature: a sync→events drop (sync 98% → events 68%) and a collapse on the
same event, pane-death (`B-DIE` passes 5/20, missed-event 0.75). A committee of
models still does not read `#{pane_dead}`.

| Fusion preset | runs | Pass% | A·sync | B·events | C·orch | missed-event |
|---|--:|--:|--:|--:|--:|--:|
| trustedrouter/synth · budget | 90 | 86% | 96% | 73% | 70% | 0.40 |
| trustedrouter/synth · quality | 90 | 82% | 100% | 63% | 50% | 0.55 |

That the missing-events deficit survives both scale (21 models) and ensembling
(fusion) is the core finding under a *minimal* agent harness: it is not a quirk of
one model or a small model, and it is not fixed by polling more models and voting.
What *does* move it is capability and scaffolding together (§6.3).

## 6.3 A mature scaffold with frontier models largely closes the gap

The deficit is not immutable. Run the **same tasks and verifier** through a mature
agent scaffold — `mini-swe-agent`, a widely-used shell-command agent (same
interaction model as our loop, but better-engineered prompting and control flow) —
with the *newest* frontier models, and it nearly vanishes. An independent sweep
(orchestrated via the Oddish/Harbor runner on Modal; 36 trials over the 9-task
suite, schema `remote-control-tbench/v1`) scores **34/36 (94%)**:

| Model (mini-swe-agent) | passed | cost |
|---|--:|--:|
| anthropic/claude-sonnet-4.5 | 9/9 | $0.33 |
| openai/gpt-5.2 | 9/9 | $0.23 |
| z-ai/glm-5.2 | 9/9 | $0.00 |
| qwen/qwen3-coder | 7/9 | $0.02 |

Both failures are `qwen3-coder`: one format-string slip (`A-FMT`) and — tellingly —
the same pane-death miss (`B-DIE`, `perceived_death` = 0) that sinks weaker setups.
The lesson is that the missing-events deficit is a function of **both capability
and scaffolding**: near-universal for a minimal scaffold across 21 models (§6.1),
yet largely solved by the strongest current models given a mature one. The robust
residue, even at the frontier, is **pane death**. This reframes the contribution:
tmuxbench is sensitive enough to separate a capable agent stack from a weak one,
and to localize the one event that remains hard. *Caveat:* this sweep varies
scaffold **and** model set simultaneously (and n=36), so it establishes that the
gap is closable, not which factor closes it; the clean cross — frontier models in
the minimal loop, and `mini-swe-agent` over weaker models — is future work.

## 6.4 Small open models on the real suite

The 21-model leaderboard (§6.1) covered open models only through OpenRouter, where
the smallest reliable options are ~7B. To get clean numbers at the small end —
where the missing-events question is sharpest — we ran the full 11-task suite over a
Qwen2.5 **size ladder** (0.5/1.5/3/7/14/32B) served by vLLM on a local H100 (text
loop, 5 seeds, 314 runs):

| Qwen2.5 | A·sync | B·events | C·orch | missed-event |
|---|--:|--:|--:|--:|
| 0.5B | 16% | 0% | 0% | 1.00 |
| 1.5B\* | 48% | 0% | — | 1.00 |
| 3B | 20% | 0% | 0% | 1.00 |
| 7B | 60% | 0% | 0% | 1.00 |
| 14B | 80% | 0% | 60% | 1.00 |
| 32B | 80% | 28% | 60% | 0.70 |

(\*1.5B under-sampled, 39 runs; small-model variance is high at the bottom.) Sync
competence climbs cleanly with size (16% → 80%), but **events stay at a flat 0% all
the way to 14B** — only **32B** cracks them at all (28%, missed-event 0.70). For
open models the missing-events wall is near-total below ~32B, and the within-model
sync→events gap is total (e.g. 14B: 80% → 0%). This is the §6.1 deficit at its most
extreme, on introspectable models. The next section isolates *which* tmux operation
causes the collapse.

## 6.5 The generalization cliff: isolating the tmux tax

To separate *tmux overhead* from *task difficulty*, we hold the goal trivial and
constant — retrieve `n` random tokens and write them to a file — and scale two
orthogonal axes (`harness/cliff.py`): **indirection depth** (D0 plain files / no
tmux, D1 read a tmux user-option, D2 capture a pane, D3 target the correct pane
among decoys, D4 respawn a dead pane) and **op-count** `n ∈ {1,2,4,8,16}`. Tokens
are random per run and the expected answer is stashed where only the checker reads
it, so a model cannot bypass the mechanism. We sweep a small-open-model **size
ladder** (Qwen2.5 0.5/1.5/3/7/14B, served with vLLM on an H100, text loop) plus
three frontier models (OpenRouter, tool loop); pass rate averaged over op-counts:

| model | D0 files | D1 option | D2 capture | D3 targeted | D4 event |
|---|--:|--:|--:|--:|--:|
| Qwen2.5-0.5B | 7% | 0% | 0% | 0% | 0% |
| Qwen2.5-1.5B | 20% | 0% | 0% | 0% | 0% |
| Qwen2.5-3B | 40% | 0% | 0% | 0% | 0% |
| Qwen2.5-7B | 93% | 33% | 0% | 0% | 0% |
| Qwen2.5-14B | 87% | 40% | 0% | 0% | 0% |
| gpt-4o-mini | 100% | 93% | 7% | 0% | 0% |
| gpt-4o | 100% | 67% | 47% | 20% | 0% |
| gpt-4.1 | 100% | 67% | 40% | 47% | 0% |

The cliff is sharp and **marches right with capability**. Three reads: (1) the
**tmux tax** — a single indirection (D0→D1, reading a tmux option instead of a
file) collapses every model ≤3B and halves 7B/14B, with the *task unchanged*; (2)
**`capture-pane` (D2) is a wall for every open model** in the ladder (0% across all
five Qwen sizes) — reading pane output is the universal break point, and only the
closed frontier models get traction; (3) **respawn/event handling (D4) walls
everyone, including gpt-4.1** — consistent with pane death being the robust residue
of §6.1–6.3. Op-count compounds independently of depth (e.g. Qwen2.5-3B clears D0
at `n ≤ 4` but not `n ≥ 8`). Because the semantic task is fixed, the gap from D0 to
each model's cliff is a clean, capability-resolved measure of how much tmux
machinery an agent can absorb before it stops generalizing. Raw runs and the
generator are in `data/` and `harness/cliff.py`.

## 6.6 RL moves the cliff by exactly one rung — a compositional generalization

Can RL move the cliff for a small model? A first, naive attempt (GRPO on *all*
depths, after SFT on every reference) only memorized: it saturated the generator
(0→100% in-distribution) but showed no transfer to the real suite. So we ran the
controlled version: SFT + GRPO on Qwen2.5-3B over **depths D0–D2 only**, with a
**KL penalty (β = 0.04)** to the base policy, holding **D3 and D4 entirely out of
training** (no reference answers, no rollouts), single-turn. We then evaluate the
base and the trained policy on every depth and on the real suite:

| depth | base | RL'd (trained D0–D2) | |
|---|--:|--:|---|
| D0–D2 | 0/45 | **45/45** | trained (memorization expected) |
| **D3** targeted pane | 0/15 | **15/15 (100%)** | **held out** |
| **D4** respawn dead pane | 0/15 | **0/15 (0%)** | **held out** |
| real 11-task suite | 3/33 (9%) | 5/33 (15%) | capability checkpoint |

The result is sharp and honest. On **D3** — a *recombination* of trained skills
(it reuses the `capture-pane` idiom from D2 plus a pane-targeting flag) — the policy
generalizes **completely** (0→100%), despite never seeing D3 in training (base 0/15;
Fisher's exact p ≈ 3×10⁻⁶ on pooled held-out, non-overlapping CIs). On **D4** — a
*genuinely novel operation* (respawning a dead pane, which appears in no trained
task) — transfer is **exactly zero** (0/15 → 0/15). The cliff moves precisely one
rung: RL extends an agent to **compositions of what it learned, but not to
operations it never saw**. Meanwhile the real-suite checkpoint did **not** regress
(9% → 15%, Fisher p ≈ 1.0, n.s.) — the KL penalty avoided the catastrophic
forgetting the naive run risked, though it brought no significant real-task gain.

Scope, stated plainly: this is **one model, one training seed** (the p-values cover
eval, not seed variance), **single-turn**, and D3/D4 are *harder same-family
variants*, not a different control plane — so this is **in-family compositional
generalization on synthetic tasks**, not a claim about general tmux competence or
cross-tool (e.g. ssh) transfer. The clean next controls are: ablate the
`capture-pane` idiom from SFT (train D0–D1, test D2–D4) to confirm it is skill-reuse
not idiom-copying; multiple seeds; and a true cross-control-plane transfer test.

# 7. Discussion and Limitations

The result stands on its own: the missing-events deficit is real, near-universal
across 21 models, and survives both a change of agent protocol (tool-calling vs
text) and a fusion ensemble. What it does **not** do is explain the cause, and we
are deliberate about not over-reading it.

**Candidate mechanisms (open).** Several explanations are consistent with the data
and not yet separated: (a) **scarce async/event training data** — most agentic
post-training is synchronous and single-turn; (b) **long-horizon/context
degradation** independent of asynchrony; (c) **model capability** — the deficit
tracks model strength within a fixed scaffold; and (d) the **agent scaffold** —
which shifts cross-model ranking (§6.1) and, with the newest models, largely closes
the gap (§6.3). §6.3 shows (c) and (d) interact: the strongest models given a
mature scaffold solve nearly everything. The cleanest next step is to *disentangle*
them — run frontier models through the minimal loop, and the mature scaffold over
weaker models — which the harness supports directly.

Limitations. **(1)** The leaderboard is **observational** — it establishes the
failure, not its cause (see candidates above). **(2)** Cross-model ranking is
harness-sensitive: the tool-calling and text loops disagree on individual sync
scores (§6.1), so we lead with the *within-model* sync→events drop, which holds in
both (20/21 and 21/21). **(3)** Single tmux version (3.6a) and platform.
**(4)** ssh / multi-host control deferred. **(5)** Sub-agent workers are
standardized for internal validity at some cost to realism.

# 8. Conclusion

Interactive control planes expose a robust behavioral failure — **missing
events** — that current evaluation under-samples. tmuxbench measures it directly on
real tmux state, never on prose: 21 models across two independent agent protocols
(**5,313** tool-calling + **9,334** text runs), with **20/21** and **21/21** worse
on events than synchronous tasks (mean gaps 36 and 50 points; missed-event up to
0.88), and a multi-model fusion ensemble that collapses the same way. Under a
*minimal* harness the deficit is near-universal; under a *native* terminal-control
harness the newest frontier models largely close it (34/36), with pane death the
robust residue — so the metric cleanly separates capability-and-scaffolding from
the genuinely hard cases. We deliberately stop short of asserting a single cause —
scarce async training data, long-horizon degradation, model capability, and the
agent scaffold are all consistent and not yet disentangled (§7). The contribution
is the instrument and the finding: a vague "agents are bad at terminals" becomes a
named, gradable, near-universal failure with a clear path to its mechanism.

# References

(Generated from `refs.bib`.)

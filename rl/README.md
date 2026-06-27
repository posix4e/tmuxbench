# rl/ — RLVR on the tmuxbench reward

Can a small model be **trained** over the generalization cliff? This trains a small
open model with the tmuxbench grader as the reward (RLVR), and re-runs the cliff to
see if it moves.

**Single-turn reduction:** the policy writes one shell script; we run it through the
harness in `--agent cmdfile` mode and use the dense `assertion_score` as reward
(`env.py`). This turns multi-turn ReAct into standard RLVR — no agentic-RL framework
needed. The reward is the real grader on real tmux state (cheat-resistant).

## Pipeline
- `env.py` — `score(depth,n,script)` = harness reward in [0,1]; `make_prompt`; `reward_batch` (TRL reward_func). Validate locally: `python rl/env.py` (reference→1.0, junk→0.0).
- `make_sft.py` — SFT warm-start demos from the reference solutions (free experts).
- `sft.py` — LoRA SFT warm-start (TRL `SFTTrainer`).
- `grpo.py` — GRPO (TRL, LoRA), reward = `env`. `--no-vllm` uses transformers-native
  rollouts (avoids vLLM↔TRL version coupling on bleeding-edge stacks).
- `eval.py` — single-turn before/after eval against a vLLM endpoint; aggregate with
  `harness/cliff_agg.py`.
- `run_h100.sh` — runs **inside the vllm-h100 guest**, in an **isolated `~/rl-env`
  venv** (never touches the guest's `vllm-env`). `SMOKE=1 bash rl/run_h100.sh` first.

## Status — RL moves the cliff by exactly one rung (compositional generalization)

Two experiments on an H100 (TRL 1.7, Qwen2.5-3B):

**1. Naive (train on ALL depths) → memorization.** In-dist 0→100%, but the real
11-task suite stayed at baseline (no transfer). Don't train on what you evaluate.

**2. Controlled hold-out (`run_heldout.sh`) → real, narrow positive.** SFT + GRPO on
**D0–D2 only**, KL penalty β=0.04, **D3/D4 held entirely out** of training:

| depth | base | RL'd (trained D0–D2) | |
|---|--:|--:|---|
| D0–D2 | 0/45 | 45/45 | trained |
| **D3** targeted pane | 0/15 | **15/15 (100%)** | held out |
| **D4** respawn dead pane | 0/15 | **0/15 (0%)** | held out |
| real 11-task suite | 3/33 (9%) | 5/33 (15%) | capability |

**D3** (recombination: capture-pane + targeting flag) generalizes completely
(Fisher p≈3e-6); **D4** (a genuinely novel op, respawn) transfers zero. So RL
extends to **compositions of trained skills, not to operations never seen**. The
real suite didn't regress (KL avoided forgetting; gain n.s.). **Scope:** one model,
one seed, single-turn; D3/D4 are harder same-family variants, not a new control
plane — in-family compositional generalization, not general tmux competence. Next
controls: ablate the capture-pane idiom from SFT (train D0–D1, test D2–D4); multiple
seeds; ssh transfer. Drivers: `run_heldout.sh` (train), `run_eval.sh` (eval);
data in `../data/rl/`.

## Upgrade path: multi-turn via TRL Harbor (future)
The single-turn reduction trades away "perceive at the right moment." The principled
multi-turn version is TRL's [Harbor integration](https://huggingface.co/docs/trl/en/harbor)
(`trl.experimental.harbor`): the policy drives each turn in a sandbox and TRL captures
tokens/log-probs/mask. tmuxbench tasks are already Harbor-shaped (instruction +
environment + verifier; we even use `task.toml`), so porting is natural — it needs a
sandbox image per rollout (docker/daytona/modal) and the multi-turn loop. Deferred:
the single-turn path gets a first result faster.

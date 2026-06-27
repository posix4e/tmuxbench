#!/usr/bin/env bash
# The rigorous version: SFT + GRPO on D0-D2 ONLY (D3-D4 held out), with a KL
# penalty to the base policy (beta>0) to guard against forgetting. Run INSIDE the
# vllm-h100 guest (reuses the isolated ~/rl-env from run_h100.sh).
#   bash rl/run_heldout.sh        # STEPS overridable
set -uo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
STEPS="${STEPS:-150}"
BETA="${BETA:-0.04}"

cd ~ && { [ -d tmuxbench ] && (cd tmuxbench && git fetch -q origin && git reset -q --hard origin/main) || git clone -q https://github.com/posix4e/tmuxbench; }
cd ~/tmuxbench
[ -x ~/rl-env/bin/pip ] || { echo "FATAL: build ~/rl-env first via rl/run_h100.sh"; exit 1; }
PY=~/rl-env/bin/python

# SFT demos restricted to D0-D2 so the held-out depths' answers never leak via SFT
$PY rl/make_sft.py --depths 0 1 2 --out rl/sft_heldout.jsonl
echo "=== HELD-OUT: SFT(D0-D2) -> GRPO(D0-D2, beta=$BETA, $STEPS steps) ==="
$PY rl/sft.py  --model "$MODEL" --data rl/sft_heldout.jsonl --out rl/out/sft_heldout --epochs 3 \
  || { echo "SFT-FAILED"; exit 1; }
$PY rl/grpo.py --model "$MODEL" --sft-adapter rl/out/sft_heldout --depths 0 1 2 --beta "$BETA" \
  --steps "$STEPS" --num-generations 8 --no-vllm --out rl/out/grpo_heldout \
  || { echo "GRPO-FAILED"; exit 1; }
echo "HELDOUT-TRAIN-OK -> rl/out/grpo_heldout"

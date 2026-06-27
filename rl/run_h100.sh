#!/usr/bin/env bash
# Run INSIDE the vllm-h100 guest. Trains in an ISOLATED venv (~/rl-env) so the
# guest's working vllm-env is never touched. Rollouts use transformers-native
# generation (--no-vllm) to avoid vLLM<->TRL version coupling on the bleeding-edge
# guest stack (transformers 5.x / vllm 0.23 / torch 2.11).
#
#   SMOKE=1 bash rl/run_h100.sh     # tiny end-to-end validation
#   bash rl/run_h100.sh             # full SFT warm-start -> GRPO
set -uo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
SMOKE="${SMOKE:-0}"
STEPS="${STEPS:-300}"

cd ~ && { [ -d tmuxbench ] && (cd tmuxbench && git fetch -q origin && git reset -q --hard origin/main) || git clone -q https://github.com/posix4e/tmuxbench; }
cd ~/tmuxbench
# the cloud image's python3-venv often lacks ensurepip -> venv has no pip; ensure it
if [ ! -x ~/rl-env/bin/pip ]; then
  rm -rf ~/rl-env
  sudo apt-get update -q >/dev/null 2>&1 || true
  sudo apt-get install -y -q python3.12-venv python3-venv >/dev/null 2>&1 || true
  python3 -m venv ~/rl-env || python3 -m venv --without-pip ~/rl-env
  [ -x ~/rl-env/bin/pip ] || ~/rl-env/bin/python -m ensurepip --upgrade
fi
[ -x ~/rl-env/bin/pip ] || { echo "FATAL: could not bootstrap pip in ~/rl-env"; exit 1; }
~/rl-env/bin/pip install -q -U pip
~/rl-env/bin/pip install -q trl peft datasets accelerate
[ -d ~/rl-env/lib ] && ~/rl-env/bin/python -c "import trl" 2>/dev/null || { echo "FATAL: trl install failed"; exit 1; }
PY=~/rl-env/bin/python
$PY -c "import trl,transformers,torch,peft,datasets;print('VERS trl',trl.__version__,'tf',transformers.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())"
$PY rl/make_sft.py --out rl/sft.jsonl

if [ "$SMOKE" = 1 ]; then
  echo "=== SMOKE: SFT 1 epoch + GRPO 3 steps ==="
  $PY rl/sft.py  --model "$MODEL" --data rl/sft.jsonl --out rl/out/sft_smoke --epochs 1 || { echo "SFT-FAILED"; exit 1; }
  $PY rl/grpo.py --model "$MODEL" --sft-adapter rl/out/sft_smoke --steps 3 --num-generations 4 --no-vllm --out rl/out/grpo_smoke || { echo "GRPO-FAILED"; exit 1; }
  echo "SMOKE-OK"
else
  echo "=== FULL: SFT warm-start -> GRPO ($STEPS steps) ==="
  $PY rl/sft.py  --model "$MODEL" --data rl/sft.jsonl --out rl/out/sft --epochs 3 || { echo "SFT-FAILED"; exit 1; }
  $PY rl/grpo.py --model "$MODEL" --sft-adapter rl/out/sft --steps "$STEPS" --num-generations 8 --no-vllm --out rl/out/grpo || { echo "GRPO-FAILED"; exit 1; }
  echo "TRAIN-OK -> rl/out/grpo"
fi

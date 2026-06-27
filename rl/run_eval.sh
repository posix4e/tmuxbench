#!/usr/bin/env bash
# Before/after eval. Serves base + the GRPO LoRA from ONE vLLM in the vllm-h100
# guest, then evals both on the cliff (in-distribution) and the real 11-task suite
# (held out from RL). Run ON tdx2.
set -uo pipefail
VM="vllm-h100"; SSH_KEY="$HOME/.ssh/id_ed25519"; GU="ubuntu"; REPO="$HOME/tmuxbench"
BASE="Qwen/Qwen2.5-3B-Instruct"
ADAPTER="${ADAPTER:-/home/$GU/tmuxbench/rl/out/grpo}"   # override for held-out: rl/out/grpo_heldout
TAG="${TAG:-}"                                          # output suffix, e.g. "_heldout"
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o LogLevel=ERROR)
V(){ virsh "$@" 2>/dev/null || sudo -n virsh "$@"; }
get_ip(){ V domifaddr "$VM" --source lease 2>/dev/null | awk '/ipv4/{print $4}' | cut -d/ -f1 | head -1; }
G(){ ssh "${SSH_OPTS[@]}" "$GU@$IP" "$@"; }

[ "$(V domstate "$VM")" != running ] && V start "$VM" >/dev/null
IP=""; for _ in $(seq 1 60); do IP=$(get_ip); [ -n "$IP" ] && break; sleep 2; done
echo "guest=$IP"
for _ in $(seq 1 40); do G true 2>/dev/null && break; sleep 3; done

echo "=== serving base + grpo LoRA ==="
G "sudo systemctl stop vllm 2>/dev/null; sudo systemctl reset-failed vllm 2>/dev/null; true"
G "bash -s" <<EOF
cat > /home/$GU/serve_eval.sh <<'SH'
#!/usr/bin/env bash
export HOME=/home/$GU HF_HOME=/home/$GU/.cache/huggingface CUDA_HOME=/usr/local/cuda
export PATH="\$CUDA_HOME/bin:/home/$GU/.local/bin:\$PATH" VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
source /home/$GU/vllm-env/bin/activate
exec vllm serve $BASE --enable-lora --lora-modules grpo=$ADAPTER --max-lora-rank 16 \
  --host 0.0.0.0 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.9
SH
chmod +x /home/$GU/serve_eval.sh
sudo systemd-run --unit=vllm --collect --uid=1000 --gid=1000 --setenv=HOME=/home/$GU -p WorkingDirectory=/home/$GU /home/$GU/serve_eval.sh
EOF
ok=0; for _ in $(seq 1 120); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://$IP:8000/health || true)" = 200 ] && { ok=1; break; }
  G 'systemctl is-active --quiet vllm' || { echo "serve died"; G 'journalctl -u vllm -n 20 --no-pager'; break; }
  sleep 8
done
[ "$ok" != 1 ] && { echo "no endpoint"; exit 1; }

cd "$REPO" && git fetch -q origin && git reset -q --hard origin/main
URL="http://$IP:8000/v1"
echo "=== cliff (all depths: D0-D2 in-dist, D3-D4 held-out) ==="
python3 rl/eval.py       --base-url "$URL" --model "$BASE" --out rl/eval_cliff${TAG}_base.jsonl
python3 rl/eval.py       --base-url "$URL" --model grpo    --out rl/eval_cliff${TAG}_grpo.jsonl
echo "=== suite (capability checkpoint) ==="
python3 rl/eval_suite.py --base-url "$URL" --model "$BASE" --out rl/eval_suite${TAG}_base.jsonl
python3 rl/eval_suite.py --base-url "$URL" --model grpo    --out rl/eval_suite${TAG}_grpo.jsonl
G 'sudo systemctl stop vllm 2>/dev/null; true'
echo "=== CLIFF base ==="; python3 harness/cliff_agg.py rl/eval_cliff${TAG}_base.jsonl | tail -20
echo "=== CLIFF grpo ==="; python3 harness/cliff_agg.py rl/eval_cliff${TAG}_grpo.jsonl | tail -20
echo "=== SUITE base ==="; python3 harness/suite_agg.py rl/eval_suite${TAG}_base.jsonl
echo "=== SUITE grpo ==="; python3 harness/suite_agg.py rl/eval_suite${TAG}_grpo.jsonl
echo "EVAL-DONE"

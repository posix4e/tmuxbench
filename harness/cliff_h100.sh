#!/usr/bin/env bash
# Run the generalization-cliff grid over a small-open-model SIZE LADDER on a
# local H100, served by vLLM inside the vllm-h100 guest. Run this ON tdx2 (the
# host that can reach the guest network); it drives the guest's vLLM and runs
# the tmux harness locally on tdx2.
#
#   git clone https://github.com/posix4e/tmuxbench ~/tmuxbench   # once
#   bash ~/tmuxbench/harness/cliff_h100.sh
set -uo pipefail

VM="vllm-h100"
SSH_KEY="$HOME/.ssh/id_ed25519"
GU="ubuntu"
REPO="$HOME/tmuxbench"
OUT="$REPO/cliff_h100.jsonl"
# Same-family size ladder (ungated, all fit one H100 in bf16):
MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
)
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
          -o ConnectTimeout=10 -o LogLevel=ERROR)
say(){ printf '\033[1;36m==>\033[0m %s\n' "$*"; }
V(){ virsh "$@" 2>/dev/null || sudo -n virsh "$@"; }
get_ip(){ V domifaddr "$VM" --source lease 2>/dev/null | awk '/ipv4/{print $4}' | cut -d/ -f1 | head -1; }

[ "$(V domstate "$VM" 2>/dev/null)" != running ] && { say "starting $VM"; V start "$VM" >/dev/null; }
say "resolving guest IP"
IP=""; for _ in $(seq 1 60); do IP=$(get_ip); [ -n "$IP" ] && break; sleep 2; done
[ -z "$IP" ] && { echo "no guest IP"; exit 1; }
say "guest IP: $IP"
for _ in $(seq 1 60); do ssh "${SSH_OPTS[@]}" "$GU@$IP" true 2>/dev/null && break; sleep 3; done

for M in "${MODELS[@]}"; do
  say "serving $M"
  ssh "${SSH_OPTS[@]}" "$GU@$IP" "sudo systemctl stop vllm 2>/dev/null; sudo systemctl reset-failed vllm 2>/dev/null; true"
  ssh "${SSH_OPTS[@]}" "$GU@$IP" "bash -s" <<EOF
cat > /home/$GU/serve.sh <<'SH'
#!/usr/bin/env bash
export HOME=/home/$GU HF_HOME=/home/$GU/.cache/huggingface CUDA_HOME=/usr/local/cuda
export PATH="\$CUDA_HOME/bin:/home/$GU/.local/bin:\$PATH"
source /home/$GU/vllm-env/bin/activate
exec vllm serve $M --host 0.0.0.0 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.9
SH
chmod +x /home/$GU/serve.sh
sudo systemd-run --unit=vllm --collect --uid=1000 --gid=1000 --setenv=HOME=/home/$GU -p WorkingDirectory=/home/$GU /home/$GU/serve.sh
EOF
  ok=0
  for _ in $(seq 1 150); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://$IP:8000/health" || true)
    [ "$code" = 200 ] && { ok=1; break; }
    ssh "${SSH_OPTS[@]}" "$GU@$IP" 'systemctl is-active --quiet vllm' || { echo "vllm died for $M"; ssh "${SSH_OPTS[@]}" "$GU@$IP" 'journalctl -u vllm -n 15 --no-pager'; break; }
    sleep 10
  done
  [ "$ok" != 1 ] && { echo "SKIP $M (endpoint not healthy)"; continue; }
  say "running cliff sweep for $M"
  OPENAI_API_KEY=x OPENAI_BASE_URL="http://$IP:8000/v1" \
    python3 "$REPO/harness/cliff_sweep.py" --models "$M" --policy openai --loop text \
      --depths 0 1 2 3 4 --ns 1 2 4 8 16 --seeds 3 --workers 4 --out "$OUT"
done
ssh "${SSH_OPTS[@]}" "$GU@$IP" 'sudo systemctl stop vllm 2>/dev/null; true'
say "ALL DONE -> $OUT"

#!/usr/bin/env python3
"""GRPO: reinforce a small model on the tmuxbench reward (rl/env.score), LoRA +
vLLM rollouts. Reward is the real harness assertion score on real tmux state.
Run on the H100 guest (needs a GPU + tmux installed for the reward env).

  python rl/grpo.py --model Qwen/Qwen2.5-3B-Instruct \
      [--sft-adapter rl/out/sft] --steps 300 --out rl/out/grpo
"""
import argparse, os, sys, random
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env as rlenv  # noqa: E402
import cliff  # noqa: E402

# Curriculum weighting: oversample depths where a small model has signal to
# bootstrap from; still expose harder rungs so GRPO can climb when ready.
DEPTH_WEIGHTS = {0: 4, 1: 4, 2: 3, 3: 2, 4: 2}
NS = [1, 2, 4]


def build_dataset(size, seed=0, depths=None):
    rng = random.Random(seed)
    allow = set(depths) if depths is not None else set(DEPTH_WEIGHTS)
    pool = [d for d, w in DEPTH_WEIGHTS.items() if d in allow for _ in range(w)]
    rows = []
    for _ in range(size):
        d = rng.choice(pool); n = rng.choice(NS)
        rows.append({"prompt": rlenv.make_prompt(d, n), "depth": d, "n": n})
    return Dataset.from_list(rows)


def tmux_reward(prompts, completions, depth, n, **kwargs):
    """TRL GRPO reward_func: dense assertion score per (depth,n,completion)."""
    return rlenv.reward_batch(depth, n, completions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--sft-adapter", default=None, help="LoRA adapter from sft.py to continue from")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--dataset-size", type=int, default=2048)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--depths", type=int, nargs="+", default=sorted(DEPTH_WEIGHTS),
                    help="restrict TRAINING depths (e.g. 0 1 2 to hold out D3/D4)")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="KL penalty to the base policy (>0 guards against forgetting)")
    ap.add_argument("--no-vllm", action="store_true",
                    help="use transformers-native generation for rollouts (avoids vLLM/TRL version coupling)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out", "grpo"))
    args = ap.parse_args()

    ds = build_dataset(args.dataset_size, depths=args.depths)
    peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    vllm_kw = ({} if args.no_vllm else
               {"use_vllm": True, "vllm_mode": "colocate", "vllm_gpu_memory_utilization": 0.35})
    cfg = GRPOConfig(
        output_dir=args.out, max_steps=args.steps,
        per_device_train_batch_size=args.num_generations,
        num_generations=args.num_generations, gradient_accumulation_steps=4,
        learning_rate=1e-5, logging_steps=2, save_steps=100, bf16=True,
        max_completion_length=512,                 # room for the script to finish + stop (TRL 1.7)
        temperature=0.9, beta=args.beta,           # KL penalty to base (>0 guards against forgetting)
        report_to=[], **vllm_kw,
    )
    model = args.model
    trainer = GRPOTrainer(model=model, reward_funcs=tmux_reward, args=cfg,
                          train_dataset=ds, peft_config=peft)
    if args.sft_adapter:
        # continue from the SFT LoRA by loading its weights into the policy
        try:
            trainer.model.load_adapter(args.sft_adapter, adapter_name="default")
            print(f"[grpo] loaded SFT adapter {args.sft_adapter}")
        except Exception as e:
            print(f"[grpo] WARN could not load SFT adapter ({e}); training from base")
    trainer.train()
    trainer.save_model(args.out)
    print(f"[grpo] saved adapter -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SFT warm-start: teach a small model to emit tmux scripts from the reference
demos (rl/make_sft.py output), LoRA. Run on the H100 guest. Saves an adapter that
grpo.py can continue from.

  python rl/sft.py --model Qwen/Qwen2.5-3B-Instruct --data rl/sft.jsonl --out rl/out/sft
"""
import argparse, os
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "sft.jsonl"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out", "sft"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    ds = load_dataset("json", data_files=args.data, split="train")

    # render prompt+completion as a chat example; train on the assistant turn
    def to_chat(ex):
        return {"messages": [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": ex["completion"]},
        ]}
    ds = ds.map(to_chat, remove_columns=ds.column_names)

    peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs,
                    per_device_train_batch_size=4, gradient_accumulation_steps=4,
                    learning_rate=args.lr, logging_steps=5, save_strategy="epoch",
                    bf16=True, max_length=1024, packing=False,
                    assistant_only_loss=True, report_to=[])
    trainer = SFTTrainer(model=args.model, args=cfg, train_dataset=ds, peft_config=peft)
    trainer.train()
    trainer.save_model(args.out)
    print(f"[sft] saved adapter -> {args.out}")


if __name__ == "__main__":
    main()

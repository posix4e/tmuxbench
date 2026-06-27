# cliff data — the generalization-cliff sweep

Raw per-run records behind the generalization-cliff result (`harness/cliff.py`).
Each line: `{model, depth, n, seed, passed, score, turns}`. Reproduce the tables
with `python harness/cliff_agg.py data/cliff_h100.jsonl`.

- **`cliff_h100.jsonl`** — small-open **size ladder**, Qwen2.5 0.5/1.5/3/7/14B,
  served by vLLM on a local H100, driven by `harness/cliff_h100.sh`. Text loop.
  5 depths × 5 op-counts × 3 seeds × 5 models = 375 runs.
- **`cliff_openrouter.jsonl`** — the 3 frontier OpenAI models (gpt-4o, gpt-4o-mini,
  gpt-4.1) via OpenRouter, tool loop. 225 runs. (The same sweep also tried 5 open
  models on OpenRouter but the budget hit $0 mid-run and those were 402'd — omitted.)
- **`suite_h100.jsonl`** — the **real 11-task suite** over the Qwen2.5 size ladder
  (0.5/1.5/3/7/14/32B) served by vLLM on the H100, text loop, 5 seeds (314 runs).
  `python harness/suite_agg.py data/suite_h100.jsonl`.

### Small-open size ladder on the real suite (`suite_h100.jsonl`)

| Qwen2.5 | A·sync | B·events | C·orch | missed-event |
|---|--:|--:|--:|--:|
| 0.5B | 16% | 0% | 0% | 1.00 |
| 1.5B* | 48% | 0% | — | 1.00 |
| 3B | 20% | 0% | 0% | 1.00 |
| 7B | 60% | 0% | 0% | 1.00 |
| 14B | 80% | 0% | 60% | 1.00 |
| 32B | 80% | 28% | 60% | 0.70 |

(*1.5B under-sampled, 39 runs.) Sync competence climbs with size, but **events are a
flat 0% up to 14B** — only 32B cracks them at all (28%, missed-event 0.70). For open
models the missing-events wall is near-total below ~32B.

## Result (pass %, averaged over op-counts 1..16)

| model | D0 files | D1 options | D2 capture-pane | D3 targeted | D4 respawn/event |
|---|--:|--:|--:|--:|--:|
| Qwen2.5-0.5B | 7% | 0% | 0% | 0% | 0% |
| Qwen2.5-1.5B | 20% | 0% | 0% | 0% | 0% |
| Qwen2.5-3B | 40% | 0% | 0% | 0% | 0% |
| Qwen2.5-7B | 93% | 33% | 0% | 0% | 0% |
| Qwen2.5-14B | 87% | 40% | 0% | 0% | 0% |
| gpt-4o-mini | 100% | 93% | 7% | 0% | 0% |
| gpt-4o | 100% | 67% | 47% | 20% | 0% |
| gpt-4.1 | 100% | 67% | 40% | 47% | 0% |

The cliff marches right with capability: ≤3B models can't reliably do even the
no-tmux baseline (D0); a single tmux indirection (D0→D1) is the "tmux tax" that
collapses small open models; `capture-pane` (D2) is a wall for every open model;
respawn/event handling (D4) is a wall for everyone, including gpt-4.1. Op-count
compounds independently (e.g. Qwen2.5-3B passes D0 at n≤4 but not n=8+).

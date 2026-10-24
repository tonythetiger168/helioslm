# Route B: beat every MediaTek model by continuing an open base (v5.127)

On 2026-10-23 the owner chose route B ("B路線") to beat every MediaTek
Research text model with the least training. The plan is to continue the
pretraining of an open Apache-2.0 3–4B base on Taiwanese data, then run
the post-training stack the repo already has. Training a 4B model from
scratch on 5T+ tokens (route A, about US$260–450k) is dropped. The
from-scratch 0.5B (#40) stays the research line and keeps its v1.0 gate.

Nothing here spends money. Every GPU stage below needs the owner's
approval first.

## The bar

| MediaTek model | TMMLU+ | MMLU | MT-Bench-tw | Note |
|---|---|---|---|---|
| Breexe-8x7B | 48.9 | 69.9 | 7.2 | hardest; Mixtral-based MoE, 47B total |
| Breeze 2 8B | 46.6 | 67.0 | – | Llama 3.1 base |
| Breeze 2 3B | 42.7 | 59.9 | 4.26 | Llama 3.2 base |
| Breeze-7B | – | – | – | Mistral base, 2024 |

The numbers are the vendors' papers, for orientation only. The gate in
`docs/prereg/b1.0.json` reruns every rival under our own scorer. Breeze 3
(2026-02) has only speech and guard models, so there is no newer text
model to beat.

## Base model

A plan may only name a base from the approved list
(`cpt_plan.BASES`). The distillation teachers are a separate list
(`eval/teacher_license_audit.py`) and are unchanged.

| Candidate | Licence | Size | Vocab | llama.cpp | Verdict |
|---|---|---|---|---|---|
| **google/gemma-4-E4B** | Apache-2.0 | 4.5B effective, 8B with per-layer embeddings | 262K | `gemma4` | **chosen** |
| google/gemma-4-E2B | Apache-2.0 | 2.3B effective | 262K | `gemma4` | pilot only |
| mistralai/Ministral-3-3B-Base-2512 | Apache-2.0 | 3.4B | 131K | `mistral3` | fallback; weaker Chinese |

Why Gemma 4 E4B:

- **Scores.** It is among the strongest open models in the size class.
  The instruct model reports MMLU-Pro 69.4 and multilingual MMMLU 76.6
  (vendor numbers). For comparison, Breexe-8x7B's MMLU is 69.9, and
  MMLU-Pro is the harder test.
- **Licence.** Gemma 4 moved from the Gemma Terms of Use to plain
  Apache-2.0, which allows commercial use, derivatives and renaming.
  Ship the base's LICENSE and NOTICE.
- **Size.** About 3 GB at Q4, so it fits the L8 (8 GB GPU) tier. The
  per-layer embedding tables can stay in CPU memory.
- **llama.cpp.** Supported natively: Google publishes GGUF builds. That
  covers the L8 and L16 local tiers from the product-line review without
  a custom GGUF architecture.
- **Architecture.** It uses the per-layer embeddings that LB8 added to
  Evastar, so the team already knows the design.

The main risk is the base's Chinese. Gemma 4 is weaker in Traditional
Chinese than some other open 4B models, so continued pretraining carries
more of the load. That is why the 2B-class pilot and an evaluation of the untouched base
come first.

Licences were read from the vendors' model cards on 2026-10-23.
huggingface.co is blocked from the cloud container, so the revision of
each base and baseline is pinned on the pod, where the `PENDING` entries
in the pre-registration are filled in.

## Fit with the roadmap (#40–#45)

- **#41 (3B dense).** Route B replaces it as the 3B-class model. E4B is
  3B class in memory: about 3 GB at Q4, which fits the L8 (8 GB GPU)
  tier. The product-line review rejected adding new sizes or
  an 8B dense model, so B1 stays a single 4B model and does not add a
  second, larger model to chase Breeze 2 8B. The 4B is gated against
  every MediaTek model, Breexe-8x7B included.
- **The v2.0 plan.** `docs/prereg/v2.0.json`, which gates the
  from-scratch 3B against Breeze 2 3B, is left as it is in case route A
  is revived. B1.0 is a separate release line.
- **#40 (0.5B).** Unchanged. It stays a research line with its v1.0 gate.
- **#42–#45 (MoE).** Unchanged. They are later and would also start
  from open bases if route B works.
- **Evastar components.** Architecture features such as MLA, KDA and
  LatentMoE do not apply to a continued Gemma model. The data,
  distillation and RL work does carry over: LB2 WSD schedule, LB3
  mixtures and quality filter, LB5 decay traces, LB6 thinking switch and
  length stages, LB7 merging and top-k-union MOPD, and the D1/D2
  DeepSeek distillation sets.

## Recipe

1. **Continued pretraining, 40B tokens on a WSD schedule.**
   - 32B stable tokens and 8B decay tokens, from
     `data/mixtures/b1_gemma4_e4b_cpt_v1.json`.
   - About 59% of the tokens are Taiwanese Traditional Chinese, from
     commercially licensed sources only. The rest replays general
     English, maths, code and Chinese data so the base does not forget
     what it knows.
   - The decay phase adds the D1 reasoning traces (LB5).
   - The base keeps its own tokenizer. Its 262K vocabulary already
     covers Traditional Chinese, and the shards are uint32 (v5.127).
   - Use a low peak learning rate (about 1e-5 to 3e-5) with a short
     warm-up.
2. **SFT**, about 1B tokens, on DeepSeek-V4.1-Flash traces (D1/D2, an
   allowed teacher) with the `/think` and `/no_think` switch (LB6).
3. **On-policy distillation and RL.** MOPD with top-k-union KL (LB7),
   then GRPO length stages (LB6). Average the best checkpoints (LB7
   linear or TIES merge).
4. **Pre-registered evaluation** against `docs/prereg/b1.0.json`:
   - B1 must beat all four MediaTek models on TMMLU+, TW-JusticeBench,
     C-Eval and CMMLU.
   - It must also beat its own base on the two Taiwanese tasks, so the
     release adds something beyond the base.
   - It may lose at most 2 points to the base on C-Eval and CMMLU.

Before training, run `eval/decontam.py` over the mixture against the
four eval sets. Non-commercial Taiwanese corpora are left out, such as
Taiwan-Text-Excellence, tw-news and TaiwanChat.

## Cost

From `python3 examples/cpt_plan.py data/cpt/b1_gemma4_e4b.json`, at
H100 US$2.7/GPU-h and MFU 0.30. The MFU is a guess, because of the
large 262K-vocabulary head: measure it in the first hour and re-cost.
The 6ND estimate uses the 4.5B effective parameters, since the
per-layer embedding tables are lookups.

| Stage | GPU-h | US$ |
|---|---|---|
| E2B pilot (5B tokens + eval) | 73 | 196 |
| B1 continued pretraining, stable phase (32B tokens) | 809 | 2,184 |
| B1 decay (8B tokens) | 202 | 546 |
| DeepSeek D2 teacher traces (4xH200, ~1 day) | 96 | 346 |
| SFT (1B tokens) | 25 | 68 |
| MOPD / RL (8xH100, ~1 day) | 192 | 518 |
| Eval (B1 x 3 seeds + 5 baselines) | 40 | 108 |
| **B1 total** | **1,364** | **3,770** |
| B1 with 100% contingency | | **7,541** |

Order of spend, each step approved separately:

1. **An evaluation of the untouched E4B, about US$20.** It shows how far
   behind Breexe the base is on TMMLU+ and TW-JusticeBench.
2. **The E2B pilot, about US$200–300.** It checks the mixture and the
   learning rate. It passes if TMMLU+ and TW-JusticeBench go up and
   C-Eval and CMMLU stay within 2 points of the base.
3. **B1 continued pretraining.**
4. **Post-training.**

If the base evaluation shows a large gap on the Taiwanese tasks, raise
the continued-pretraining budget, for example to 80B tokens (about
US$5.5k without contingency), before the B1 run rather than after it.

## What the CPU prep PR contains (v5.127)

- `evastar/src/training/cpt_plan.py`: the approved bases, the plan schema and the cost model.
- `examples/cpt_plan.py`: prints the cost table and, with `--bases`, the
  candidate list.
- `data/cpt/b1_gemma4_e4b.json` and `data/cpt/b0_gemma4_e2b_pilot.json`:
  the run plans.
- `data/mixtures/b1_gemma4_e4b_cpt_v1.json`: the continued-pretraining
  mixture.
- `docs/prereg/b1.0.json`: the pre-registered gates.
- uint32 token shards for vocabularies over 65,536:
  `pretrain.write_shards(dtype=...)`, `TokenShard(path, dtype=...)` and
  `build_midtrain.py --dtype`.
- T130 `test_cpt_plan.py`.

Still to do, for the pod:

- A Hugging Face transformers training entry point for the continued
  pretraining. transformers is not installed in CI, and adding it needs
  the owner's consent.
- uint32 ids in the SFT shards (`distill_sft`) and the logit-KD target
  arrays.
- Pin the revisions marked `PENDING`.

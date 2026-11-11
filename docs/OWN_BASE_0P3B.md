# Own base model, 0.3B, from scratch on TPU (v5.266)

A small base model trained from random initialisation with our own
tokenizer and our own data mixture: a technology demonstration of the
full pretraining pipeline, not a product model. Product tiers keep
building on open base checkpoints.

## Model

| | |
|---|---|
| Shape | dense decoder, 24 layers, hidden 1024, 16 query / 4 KV heads (GQA), SwiGLU 2816, RMSNorm, RoPE, tied embeddings |
| Parameters | 304M (`PRESETS["0.3b"]` in `helioslm_v5/src/training/jax_pretrain.py`) |
| Tokenizer | own byte-level BPE, 32,768 tokens, Traditional + Simplified Chinese characters seeded as single tokens (`examples/pilot_prepare_data.py`) |
| Context | 2,048 |
| Export | standard `LlamaForCausalLM` (`export_hf`), so transformers, vLLM, lm-eval and llama.cpp load it without custom code |

## Data

`data/mixtures/own_base_0p3b_v1.json`: main phase about 40% Traditional
Chinese web + Wikipedia, 42% English educational web, 8% Simplified
Chinese web, 10% maths; the last 20% of steps anneal on the
higher-quality subsets (filtered zh-TW web, FineWeb-Edu, FineMath 4+).
Allow-listed card licences only (MIT, ODC-BY-1.0, CC-BY-SA-4.0); code
and unreviewed sources stay out. `fetch_mixture.py` re-reads every card
at the downloaded revision and records it in `manifest.json`.

## Compute

FLOPs per token ≈ 6 × 304M + attention ≈ 2.4 GFLOP. Kaggle's free TPU
v5e-8 (8 chips × 197 bf16 TFLOP/s) at 25–40% MFU gives about 160k–260k
tokens/s:

| Tokens | Wall time on v5e-8 | Kaggle sessions (9 h each, ~20 h/week) |
|---|---|---|
| 6B (≈ 20 tokens / parameter) | 6.5–10 h | 1–2 |
| 20B | 21–35 h | 3–4 (about 2 weeks) |

Data preparation (tokenizer + tokenisation + sharding) runs once on a
small CPU machine; the shards live in a private Hugging Face dataset and
the TPU downloads one shard per source at a time.

## Steps

1. **Tokenizer** (CPU): `pilot_prepare_data.py --hf ... --max-docs 100000 --vocab 32768 --out tok/` on a sample of each main source.
2. **Tokenise** (CPU): `fetch_mixture.py --mixture data/mixtures/own_base_0p3b_v1.json --tokenizer tok/tokenizer.json --tokens 6e9 --allow-blocked --out own_data/`.
3. **Shard + upload** (CPU): `own_base_shards.py --data own_data --out own_shards --upload <user>/helioslm-own-0p3b-tokens`.
4. **Train** (TPU): `examples/notebooks/kaggle_tpu_own_base.ipynb`, *Run all* once per session until the last step; it resumes from the private checkpoint repo and stops cleanly before the session limit.
5. **Evaluate** the exported checkpoint with the usual Taiwanese benchmark scripts (`examples/base_gap_eval.py`) and report validation loss per byte.

## Expectations

At this size and token count a model learns fluent text but little
exam knowledge: multiple-choice benchmarks such as TMMLU+ are expected
near the 25% chance level. The value is the pipeline (own tokenizer,
own data, own training code, standard export) and a baseline for
larger runs on more compute.

## Smoke tests

```bash
pip install jax            # CPU is enough
python -m pytest -q helioslm_v5/tests/test_jax_pretrain.py
python examples/jax_pretrain_tpu.py --preset tiny --synthetic --steps 50 \
    --batch 4 --seq 64 --ce-chunk 32 --out runs/jax_smoke
```

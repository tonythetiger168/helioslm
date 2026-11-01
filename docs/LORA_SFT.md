# LoRA SFT for small domain data (v5.217)

`examples/sft_hf.py train --lora` trains LoRA adapters instead of the full
model. It is meant for small domain sets (a few hundred to a few thousand
prompt / response pairs, e.g. 300–1,000 de-identified JSON → report pairs),
where full-parameter SFT costs more memory and forgets more.

```bash
# 1. tokenise as usual
python examples/sft_hf.py prepare --records pairs.jsonl \
    --tokenizer google/gemma-4-E4B-it --out /workspace/sft_pairs --seq 2048
# 2. LoRA train; base weights frozen in bf16
python examples/sft_hf.py train --model google/gemma-4-E4B-it \
    --data /workspace/sft_pairs --out /workspace/runs/lora \
    --lora --lora-r 16 --lora-alpha 32 --frozen-bf16 \
    --lr 2e-4 --epochs 3 --grad-accum 8
# 3. package the merged model like any other checkpoint
python examples/package_hf.py --model /workspace/runs/lora/final ...
```

Medical data also needs `--data-manifest` (and `--medical`) so the PHI and
licence guard runs (v5.215, `helioslm_v5/src/training/data_guard.py`).

Outputs:

- `<out>/adapter/adapter_config.json` + `adapter_model.safetensors`: the
  adapter alone, with peft key names (`peft.PeftModel.from_pretrained` loads
  it), a few tens of MB;
- `<out>/final`: the base with the adapter merged in (bf16), so
  `package_hf.py` builds GGUF tiers from it unchanged. `--lora-no-merge`
  skips it.

Options: `--lora-r` (16), `--lora-alpha` (32, scale = alpha / r),
`--lora-dropout` (0.05), `--lora-targets` (default `q_proj k_proj v_proj
o_proj gate_proj up_proj down_proj`, the names Gemma, Llama and Qwen use;
a wrapper such as `q_proj.linear` matches too), `--lora-exclude` (default
`vision audio`: the encoders stay frozen). `--freeze` is ignored with
`--lora`: everything but the adapters is frozen. LoRA usually wants a
higher learning rate than full SFT (1e-4 to 3e-4 instead of 1e-5).

The implementation is plain torch (`helioslm_v5/src/training/lora.py`),
so neither CI nor the pods need `peft`.

## Memory estimate

Rough numbers for one GPU, `--frozen-bf16`, micro-batch 1, seq 2048,
gradient checkpointing, r = 16 on the seven projections. Frozen bf16
weights take 2 bytes per stored parameter; the adapters plus AdamW state
are under 1 GB for these sizes; activations with checkpointing add a few
GB and grow with `--seq` and `--micro-batch`. These are estimates, not
measurements: check `nvidia-smi` on the first steps.

| Base | Stored params | Estimate | GPU |
|---|---|---|---|
| Gemma 4 E4B (B1 base) | ~8B (per-layer embeddings included) | 18–22 GB | 24 GB (L4, RTX 3090 / 4090) |
| Qwen3 8B / Llama 3.1 8B | ~8B | 18–22 GB | 24 GB |
| Qwen3 4B | ~4B | 10–13 GB | 16 GB |
| 1–3B models | 1–3B | 4–9 GB | RTX 4060 8 GB (3B only with short `--seq`) |

Without `--frozen-bf16` the frozen base stays fp32 and the weight term
doubles.

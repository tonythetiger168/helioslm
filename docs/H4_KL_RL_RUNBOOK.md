# H4 KL-anchored RL gate — pod runbook (v5.272+)

Runbook for the KL-anchored RL gate from `docs/HDL_BENCH.md`
("Pod-side H4 follow-up gates", item 3). The H4 loop now implements the
frozen-reference k3-KL anchor behind `--kl-coef` (zero retains the original
no-reference control). The first run regressed OpenTitan bug-fix editing
from the stock E4B-it baseline 59/179 to 6/179. This gate uses the anchor
(the pattern in `helioslm_v5/src/training/grpo.py` and
`examples/swe_agent_rl_train.py`), sweeps the coefficient once, and only
then considers scaling RL.

Everything below runs on the pod. HDL-generated code is untrusted:
keep it inside the pod/container, as in `docs/HDL_BENCH.md`.

## Gate 0 — reward audit must be clean before any RL

```bash
python examples/hdl_rl_hf.py audit --tasks /workspace/rl/tasks.jsonl \
    --seeds 1 17 101 --mutants 8 --formal \
    --out /workspace/rl/reward_audit.json
```

Proceed only when the summary has `golden_ok == tasks`,
`false_accept == 0`. Nonzero `unknown` must be resolved with more
directed seeds or a longer `--mutants` pass first (`unknown` is not a
proof of reward safety).

## Step 1 — KL anchor implementation (CPU-verified; pod sweep pending)

The update path is `pg_loss` (token-level `-A * log p`). The implementation uses a frozen
reference pass and the non-negative k3 estimator, exactly the estimator
used in `grpo.py` (`kl = coef * (expm1(logr) - logr)`,
`logr = ref_logp - policy_logp`, clamped):

```python
def ref_logprobs(model, prompt_ids, resp_ids, device):
    """sum-free per-token log p(response) under the frozen reference."""
    import torch
    import torch.nn.functional as F
    ids = torch.tensor([prompt_ids + resp_ids], device=device)
    k = len(resp_ids)
    with torch.no_grad():
        out = model(input_ids=ids[:, :-1], logits_to_keep=k)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return -F.cross_entropy(logits[0, -k:].float(), ids[0, -k:],
                                reduction="none")

def kl_loss(ref_lp, policy_lp, coef):
    """k3 KL penalty (DeepSeekMath estimator), non-negative by construction."""
    import torch
    logr = (ref_lp - policy_lp).clamp(max=60.0)
    return coef * (torch.expm1(logr) - logr)
```

Wiring (mirrors `swe_agent_rl_train.py`):

1. `cmd_train` loads the reference once from `--model` (the SFT
   starting checkpoint) with `AutoModelForCausalLM.from_pretrained(...,
   dtype=torch.bfloat16)`, `eval()`, `requires_grad_(False)`, and never
   reloads it — the anchor stays the SFT model for the whole run.
2. New flags: `--kl-coef` (default 0.0 = current behaviour, so the
   no-KL run is the in-sweep control), `--ref` (default: `--model`).
3. Per kept sample: compute `ref_lp` under `no_grad`, add
   `kl_loss(ref_lp, policy_lp, a.kl_coef).sum() / n_tokens` to the
   `pg_loss` term before `backward`.
4. Log `kl` (mean per round) in `log.jsonl` next to `reward`/`pass`.

Memory note: the frozen bf16 reference of a 4B model costs ~8 GB; on
one A40 with `--frozen-bf16` policy weights this fits alongside the
vLLM rollout subprocess only if `--gpu-mem` is lowered (0.35 worked in
H3 SFT; measure, don't assume).

Implementation acceptance (CPU, before the pod): a new pure-function
test that `kl_loss` is zero when `ref_lp == policy_lp`, positive
otherwise, and monotone in `|logr|`; plus a tiny-random-model test that
one optimizer step with a large `--kl-coef` moves the policy less than
the same step with `--kl-coef 0` (same shape as
`test_pg_loss_raises_logprob_of_positive_answer`). Because k3 has zero
gradient when policy and reference are exactly equal, the tiny-model test
first applies an identical warm step to both arms, then measures the next
step from that shared off-anchor state.

## Step 2 — one coefficient sweep, fixed budget

Same tasks, same seed, same `--max-hours` per arm; only `--kl-coef`
varies:

| arm | `--kl-coef` | purpose |
|---|---|---|
| control | 0.0 | reproduces the no-KL regression baseline |
| low | 0.005 | below the `grpo.py` default 0.01 |
| mid | 0.01 | `grpo.py` / `config_v5.py` default |
| high | 0.04 | strong anchor |

```bash
for KL in 0.0 0.005 0.01 0.04; do
  python examples/hdl_rl_hf.py train --model /workspace/sft \
      --tasks /workspace/rl/tasks.jsonl --out /workspace/runs/rl_kl$KL \
      --kl-coef $KL --frozen-bf16 --max-hours 5 --seed 0
done
```

Watch `log.jsonl`: a working anchor shows `kl` rising early then
flattening, with `pass` not collapsing the way the un-anchored run did.
Abort an arm early if `kl` diverges (order-of-magnitude growth across
rounds) — that is a sign the reward is being hacked faster than the
anchor can hold, and the arm's result is not interpretable.

## Step 3 — acceptance evals

For each arm's `<out>/final`, on the held-out sets:

```bash
python examples/ot_bugfix_run.py generate --tasks ot_tasks.jsonl \
    --model /workspace/runs/rl_kl<KL>/final --out ot_kl<KL>.samples.json
python examples/ot_bugfix_run.py score --tasks ot_tasks.jsonl \
    --ot /workspace/opentitan --samples ot_kl<KL>.samples.json \
    --out ot_kl<KL>.json
python examples/hdl_eval_all.py --model /workspace/runs/rl_kl<KL>/final \
    --rtllm-exclude <list from self-check> \
    --out results/hdl/rl_kl<KL>.json
```

An arm **passes the gate** only when all of:

- reward audit of the training tasks was clean (Gate 0);
- OpenTitan bug-fix pass count is at least the stock E4B-it baseline
  (59/179) — the regression the anchor exists to prevent;
- VerilogEval v2 and RTLLM v2.1 pass@1 within 0.02 absolute of the SFT
  starting checkpoint (same seeds, same n);
- CVDP pass@1 within 0.01 absolute of the SFT starting checkpoint.

Report every arm in one table (pass/fail per criterion, `kl` curve
endpoint, wall hours). If no arm passes, the conclusion is "the H4
reward needs directed-seed or formal-strength work before RL", not
"try more coefficients" — go back to Gate 0 with `--mutants 16`.

## What this gate does not claim

- It validates the anchor on the H4 reward only; it says nothing about
  MCQ rewards (`--reward mcq`) or other domains.
- All thresholds are absolute deltas against the SFT start on our own
  Icarus-based scorers — not comparable to vendor-reported numbers.
- The 6/179 and 59/179 OpenTitan figures are the v5.265 baseline
  (`fail_types` breakdown in that CHANGELOG entry); re-measure the
  E4B-it row if the task file or scorer has changed since.

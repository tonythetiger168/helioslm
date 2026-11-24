# AGENTIC_MID_CANDIDATE — the trained mid_sft_v5.33 checkpoint on the
# five agentic tracks

Status: measured, 2026-11-23. Review-plan item B2-3 follow-up. This is the
second measured five-component candidate run (the first was the toy
checkpoint pipeline smoke). Companion doc: `docs/AGENTIC_BENCH.md`.

## Candidate

`checkpoints/mid_sft_v5.33.pt` — 359,992,352 params, BPE tokenizer
(vocab 32768), mid preset (hidden 1024, 8 layers, 16 heads / 4 KV heads,
kv_latent 256, q_lora 512, 4 experts / 2 active, max_position 8192,
MTP num_modules 1). Checkpoint sha256
`fb0cc813421832912ab47d7832b47735a7ec698417574e14744c20e876b554d6`
(1,440,052,622 bytes, `*.pt` is gitignored; the blob lives in the
project's Hugging Face repo `chienhsinlin/helioslm`). Wrapper:
`examples/mid_candidate.py` (`--candidate mid` in
`examples/agentic_candidate_run.py`). Greedy decoding, every component
on the same checkpoint bytes.

## Five-track results

| component | score | n | evidence |
|---|---|---|---|
| terminal | 0.0000 | 50 | model_run / measured |
| swe | 0.0000 | 20 | model_run / measured |
| long_horizon | 0.0000 | 5 | model_run / measured |
| tool_recovery | 0.0000 | 70 | model_run / measured |
| hdl | 0.0000 | 156 | model_run / measured |
| **Agentic Index** | **0.000000** | | `is_model_score: true` |

Artifacts, descriptors, `task_sets.json` and the index are under
`results/agentic/2026-11-23_mid-v5.33/` (git-ignored, like the toy run);
the index verifies byte-for-byte with `--check`.

The index equals the toy floor, but the failure mode is different and
more informative: the toy model babbles code-shaped text; the mid model
**recognizes none of the five protocols** — its SFT distribution is the
chat tool protocol (`##user##` / `##assistant##` / `##tool##` with
`@@tool@@{"calls":[...]}@@end@@` blocks and the
calc/str_op/file_read/file_write/finish tools). Concretely, measured on
this run's artifacts:

- **terminal / swe** (bash-in-```bash-fences protocol): the model emits
  one or two tokens (typically `'con'` — the start of "conf") and hits
  EOS immediately; all 50 terminal episodes end at `max_steps` with
  **500/500 replies invalid** and zero commands executed; SWE likewise
  0/20.
- **long_horizon** (single-tool-call policy protocol): the model emits
  `@@tool@@`-flavoured text that does not parse against the scenario
  tool set; all five scenarios exhaust their step budgets
  (steps 8/6/12/6/6) with zero guard violations and zero false
  completions.
- **tool_recovery** (repair-a-malformed-block protocol): replies are
  partial regurgitations of the corrupted block, never the exact
  repaired calls; **70/70** fail strict equality.
- **hdl** (Verilog spec-to-RTL): 156 problems x 1 sample, pass@1 0.0;
  failure codes **121 syntax / 35 unknown-module** (the model writes
  natural-language text, not RTL — Icarus rejects it).

None of this is a harness or `generate()` bug — the same checkpoint,
driven **in-distribution**, behaves exactly as trained (next section).
The 0.000000 is an honest measured score on a candidate whose training
distribution does not cover the harness prompts; lifting it requires
SFT data in the target protocols, not harness concessions.

## Control: in-distribution grounded chat

Same checkpoint, same greedy decoding, driven through the chat protocol
it was trained on (`ChatSession`, tools calc/str_op/file_read/finish,
task seed 20260928, 12 tasks across CalcEnv/StrEnv/ComposeEnv):

| gate | outcome | note |
|---|---|---|
| `FixedGate(Route.DIRECT)` | 0/12 | the model **confabulates tool arguments** — e.g. rewrites "41 + -28 - 77 + 94" as "94 + -28 + 97 + 94" — and confidently answers for the hallucinated expression; reproduces deterministically (v4: `checkpoints/mid_agent_eval_v4.json`) |
| `GroundingGate(FixedGate(Route.DIRECT))` | **9/12 correct + 3/12 abstained** (v5: `checkpoints/mid_agent_eval_v5.json`) | the gate intercepts the hallucinated args and feeds back the real observations; the 3 remaining tasks fire the premature-finish refusal (`write_read` family — the model tries to finish before any evidence is gathered) and the oracle returns `ABSTAINED (insufficient evidence)`; **no hallucination leaks** |

The pair (0/12 ungrounded vs 9/12 + 3 safe abstentions grounded) is the
sharpest evidence so far for the project's calibration/know-when-to-abstain
main line: the mid model's failure is not incapacity, it is **unverified
self-trust** — it invents a different problem and solves that one.
Grounding the tool calls against the real environment recovers full
success on every task the model actually engages, and turns its remaining
premature-finish habit into a clean refusal instead of a wrong answer.

> Session note (2026-11-24, sandbox re-verification): re-running
> `examples/eval_mid_agent_v5.py` on this checkout reproduces the recorded
> `mid_agent_eval_v5.json` numbers exactly (9/12 + 3 abstentions). A
> separate ad-hoc probe in an earlier sandbox run produced 12/12 by
> re-prompting after each refusal (treating the abstention observation as
> a normal tool observation instead of a terminal one); that is a harness
> policy choice, not a contradiction — the committed eval's stricter
> accounting is the number of record.

## Reproduce

```bash
# needs the checkpoint blob (gitignored) from HF chienhsinlin/helioslm
python3 examples/agentic_candidate_run.py \
  --candidate mid \
  --checkpoint checkpoints/mid_sft_v5.33.pt \
  --results results/agentic/2026-11-23_mid-v5.33

python3 -m helioslm_v5.eval.agentic_index \
  --results results/agentic/2026-11-23_mid-v5.33 \
  --output results/agentic/2026-11-23_mid-v5.33/agentic_index.json --check
```

(HDL additionally needs `iverilog` + `vvp` on PATH; in this sandbox the
restored tools bundle at `/mnt/agents/output/tools_bundle/` provides
Icarus 13.0.)

## What this means for the roadmap

- The Agentic Index floor is now measured twice, on two very different
  checkpoints, and the index stays honest: it does not move until a
  candidate actually covers the five protocols.
- Next trained-candidate runs should add SFT data in the harness
  protocols (term_bench bash replies, long-horizon single-call policy,
  tool-recovery repair format, Verilog spec-to-RTL) rather than scaling
  the existing chat distribution.
- The GroundingGate result argues for keeping the grounded-chat eval as
  a standing regression gate for every future candidate (it is the one
  in-distribution probe we have).

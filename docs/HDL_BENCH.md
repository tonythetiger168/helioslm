# HDL bench (v5.147)

Chip-design track, stage H0: measure Verilog generation before any HDL
training. First benchmark: **VerilogEval v2 spec-to-rtl** (156 problems,
NVlabs/verilog-eval at commit `c498220d`, MIT).

## On the pod

```bash
apt-get install -y iverilog yosys      # yosys optional
pip install pyslang==12.0.0            # optional lint
python -m helioslm_v5.eval.hdl_bench fetch
python -m helioslm_v5.eval.hdl_bench self-check          # must print 156/156
python examples/hdl_bench_run.py --model <hf-id> --out results/hdl/<name>.json
# upstream-style pass@1: --n 20 --temperature 0.8
```

## What is scored

| Field | Meaning |
| --- | --- |
| `pass@k` | share of problems solved (unbiased estimator over `n` samples) |
| `failure_codes` | upstream sv-iv-analyze codes: `.` pass, `S` syntax, `C` compile, `T` timeout, `R` runtime mismatch, `r` reset in sensitivity list, `p` unbound signal, `m` unknown module, `w` reg declared as wire, `n` no sensitivities, `e` cast, `0` zero-width constant, `c` clk not bound |
| `lint` / `synth` | with `--extra`: pyslang errors and yosys cell count, not part of pass@k |

## RTLLM v2.1 (v5.176)

50 designs (hkust-zhiyao/RTLLM at `51ed553d`, MIT). Upstream scores with
Synopsys VCS; we use Icarus with the same two rules (compiles = syntax,
simulation output contains "pass" = function), so the numbers are
"RTLLM v2.1 (Icarus)" and are compared only with our own runs.

```bash
python -m helioslm_v5.eval.rtllm_bench fetch       # git clone at the pinned commit
python -m helioslm_v5.eval.rtllm_bench self-check  # prints designs Icarus cannot run
python examples/hdl_bench_run.py --bench rtllm --model <hf-id> \
    --exclude <list from self-check> --out results/hdl/<name>_rtllm.json
```

Output has `syntax@k` and `pass@k` (plain mean over designs; upstream
divides by 51) and the excluded designs by name.

Samples are untrusted code; run them only inside the pod or another
container. Next in H0: CVDP (CC-BY-4.0 + Apache-2.0).

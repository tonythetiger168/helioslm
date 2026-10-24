# HDL bench (v5.147)

Chip-design track, stage H0: measure Verilog generation before any HDL
training. First benchmark: **VerilogEval v2 spec-to-rtl** (156 problems,
NVlabs/verilog-eval at commit `c498220d`, MIT).

## On the pod

```bash
apt-get install -y iverilog yosys      # yosys optional
pip install pyslang==12.0.0            # optional lint
python -m evastar.eval.hdl_bench fetch
python -m evastar.eval.hdl_bench self-check          # must print 156/156
python examples/hdl_bench_run.py --model <hf-id> --out results/hdl/<name>.json
# upstream-style pass@1: --n 20 --temperature 0.8
```

## What is scored

| Field | Meaning |
| --- | --- |
| `pass@k` | share of problems solved (unbiased estimator over `n` samples) |
| `failure_codes` | upstream sv-iv-analyze codes: `.` pass, `S` syntax, `C` compile, `T` timeout, `R` runtime mismatch, `r` reset in sensitivity list, `p` unbound signal, `m` unknown module, `w` reg declared as wire, `n` no sensitivities, `e` cast, `0` zero-width constant, `c` clk not bound |
| `lint` / `synth` | with `--extra`: pyslang errors and yosys cell count, not part of pass@k |

Samples are untrusted code; run them only inside the pod or another
container. Next in H0: CVDP (CC-BY-4.0 + Apache-2.0) and RTLLM.

## CVDP v1.1.0, non-agentic code generation (v5.151)

NVIDIA's CVDP (`nvidia/cvdp-benchmark-dataset`, text CC BY 4.0, code
Apache-2.0). Only the open-source flow: the `no_commercial` file and
datapoints whose harness uses the stock simulator image; `list` prints
every skipped id with its reason. The pod has no docker, so each harness
runs directly with its paths pointed at a temp dir; a datapoint passes
when every harness command exits 0, as upstream.

```bash
pip install cocotb==2.0.1 pytest==8.3.2 cocotb-bus==0.3.0 cocotbext-axi==0.1.28
python -m evastar.eval.cvdp_bench fetch          # needs HF access
python -m evastar.eval.cvdp_bench list           # runnable vs skipped
# golden check needs the *_with_solutions file, if you have it:
python -m evastar.eval.cvdp_bench self-check --file <with_solutions.jsonl>
```

Generation and scoring are separate (`cvdp_bench.score(rows, texts)`),
so simulations can run on CPU after the GPU time ends.

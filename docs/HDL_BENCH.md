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
container.

## CVDP v1.1.0, non-agentic code generation (v5.177)

NVIDIA's CVDP (`nvidia/cvdp-benchmark-dataset`, text CC BY 4.0, code
Apache-2.0). Only the open-source flow: the `no_commercial` file and
datapoints whose harness uses the stock simulator image; `list` prints
every skipped id with its reason. The pod has no docker, so each harness
runs directly with its paths pointed at a temp dir; a datapoint passes
when every harness command exits 0, as upstream.

```bash
pip install cocotb==2.0.1 pytest==8.3.2 cocotb-bus==0.3.0 cocotbext-axi==0.1.28
python -m helioslm_v5.eval.cvdp_bench fetch          # needs HF access
python -m helioslm_v5.eval.cvdp_bench list           # runnable vs skipped
# golden check needs the *_with_solutions file, if you have it:
python -m helioslm_v5.eval.cvdp_bench self-check --file <with_solutions.jsonl>
```

Generation and scoring are separate (`cvdp_bench.score(rows, texts)`),
so simulations can run on CPU after the GPU time ends.

## All three in one run (v5.178)

```bash
python examples/hdl_eval_all.py --model <hf-id or checkpoint dir> \
    --rtllm-exclude <list from self-check> --out results/hdl/<name>.json
```

Generates every prompt of the three benchmarks in one vLLM batch, then
simulates on all CPU cores. CVDP has no public golden outputs for the
`no_commercial` file, so its harness failures are not separated from
model failures; compare models only against each other.

## UVM testbenches, SystemVerilog split and gate count (v5.182)

`helioslm_v5/eval/uvm_bench.py` asks the model for a self-checking UVM
testbench (top module `tb`) for a VerilogEval design, given its spec and
working RTL. Each testbench is built once with Verilator and upstream
UVM 2017-1.0 (no DPI) and run on the golden design and on up to three
mutants (one operator or constant changed). `prepare` keeps a mutant
only when VerilogEval's own testbench tells it apart from the
reference, so no mutant is equivalent: 129 of the 156 designs have at
least one. `calibrate` then keeps only the mutants that a random-stimulus
reference UVM testbench catches on Verilator, which is two-state (a
mutant that differs only in X or Z values cannot be caught here). Scores: `build@k`, `golden@k` (UVM_ERROR and UVM_FATAL are
0 on the golden design), `pass@k` (golden clean and every mutant
caught) and `kill_rate`.

```bash
# Verilator >= 5.052 (built from source; Ubuntu's package is too old),
# Icarus for prepare; ccache speeds up the per-testbench C++ builds
python -m helioslm_v5.eval.uvm_bench fetch      # UVM package from Verilator's tree
python -m helioslm_v5.eval.uvm_bench prepare    # mutants, needs the VerilogEval cache
python -m helioslm_v5.eval.uvm_bench calibrate  # drop mutants no testbench can catch here
python -m helioslm_v5.eval.uvm_bench self-check # reference testbench must pass all
python examples/hdl_eval_all.py --model <id> --benches verilog-eval,rtllm,cvdp,uvm \
    --extra --out results/hdl/<name>.json
```

A build takes about 80 s of two CPU cores (UVM is compiled with every
testbench) and about 3.3 GB of memory; the number of parallel builds
follows the container's CPU and memory limits (a pod's `nproc` shows the
host's cores).

The testbench is taken from the answer after `[BEGIN]`, up to `[DONE]`,
`[END]` or `[END_OF_CODE]` (or the end of a cut-off answer). Every build
also gets empty `uvm_pkg.sv` / `uvm_pkg.svh` files, for testbenches that
include the package source as well as importing it, and maps the UVM 1.2
sequence macros (`uvm_do_with`, `uvm_do_on`, `uvm_rand_send_with`, ...)
onto the 1800.2-2017 ones (v5.184).

First baseline (2026-10-27, n=1, 128 tasks): no testbench from
Gemma 4 E4B-it or CodeV-R1-RL-Qwen-7B passes. Most do not parse (an
interface declared inside a package, include files that do not exist,
declarations after statements, missing forward declarations); one
CodeV-R1 testbench builds and then ends with `$finish` before the UVM
report. UVM testbench writing has to be trained.

CVDP results now also carry `pass@k_by_language` (`sv` when an expected
output file is `.sv`/`.svh`, `v` for Verilog). `--extra` runs yosys
`synth` on VerilogEval samples and the reference; each passing sample
gets `area_ratio` = reference cells / sample cells (above 1 is smaller
than the reference), averaged as `area_ratio_mean`.

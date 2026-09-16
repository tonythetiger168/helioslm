# Contributing to HeliosLM

Thanks for your interest in contributing! This project values **correctness you can verify** over feature velocity. If that sounds like your kind of codebase, you're in the right place.

## Ground rules

1. **Every change needs a test.** Unit tests live in `helioslm_v5/tests/`; end-to-end checks in `integration_test_v51.py`. Run both before opening a PR:
   ```bash
   python -m helioslm_v5.tests.test_v5
   python integration_test_v51.py
   ```
2. **Correctness claims need evidence.** If you optimize or refactor a core path (MLA absorption, hybrid decode, DualPipe, speculative sampling), include an equivalence check (bitwise or tight tolerance, matching existing style) — not just "loss looks fine."
3. **Keep it CPU-runnable.** The whole point is that anyone can iterate without a GPU. If your change needs CUDA, guard it and keep the CPU path default.
4. **Small PRs > big PRs.** A focused change with tests beats a sweeping refactor with none.

## Getting started (good first issues)

Open issues labeled [`good first issue`](https://github.com/tonythetiger168/helioslm/labels/good%20first%20issue) — comment on one before starting:

- **#1 CUDA end-to-end verification** — run the suites on a GPU machine and report (hardware access is the only requirement)
- **#2 Fused vs unfused quant speedup benchmark** — `benchmarks/bench_quant.py` + BENCHMARKS.md entry
- **#3 Notebook: add a new attention variant in 30 lines** — guided tour of the hackable-stack pitch
- **#4 HF Hub integration** — `push_to_hub` / `from_hub` with round-trip test
- **#5 Real BPE tokenizer for the toy checkpoint** — replace char-level with a 1024-vocab BPE

More starters:
- Add missing edge-case tests (empty batch, single-token sequence, max-length boundary)
- Improve error messages and docstrings in a module you just read
- Fix a `TODO`/`FIXME` you find in the source

## Development workflow

1. Fork and clone the repo
2. Create a branch: `git checkout -b feat/short-description` (or `fix/...`)
3. Make your change + add tests
4. Run the full test suite (commands above)
5. Open a PR with:
   - What changed and why
   - Test evidence (paste the test run output)
   - If applicable: before/after numbers (memory, tokens/s)

## Code style

- Pure PyTorch; avoid heavyweight dependencies (the only install is `torch`)
- Match the existing module layout under `helioslm_v5/src/`
- Comment *why*, not *what* — the code should read like a paper's appendix

## Roadmap (where help is most wanted)

Maintainer priorities — comment on the issue (or open one) before starting:

| Area | Task | Difficulty |
|---|---|---|
| Pre-trained toy checkpoint | Train a small model on an open corpus so users can `load` and generate out of the box | Medium — high impact |
| CUDA verification | Run the static-checked CUDA paths end-to-end on real hardware | Medium |
| Fused quantization kernels | Speed up GPTQ/AWQ packing paths | Hard |
| HF Hub integration | Load/save configs and checkpoints from Hugging Face | Easy–medium |
| Example notebooks | "Train a tiny HeliosLM on your laptop", "Add an attention variant in 30 lines" | Easy |
| Benchmarks | Standardized CPU benchmark suite for engine + quantization | Easy–medium |
| GGUF / llama.cpp export | Export MLA-variant weights to GGUF for the LocalLLaMA community | Hard |

## Questions?

Open a [Discussion](https://github.com/tonythetiger168/helioslm/discussions) or comment on an issue. Maintainer aim: respond within 48 hours.

## Code of conduct

Be kind, be technical, be honest about limitations. Review comments target code, never people. New contributors get extra patience — everyone was new once.

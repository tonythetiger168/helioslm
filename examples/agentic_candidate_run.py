"""agentic_candidate_run.py — roadmap Phase 3 (P1): the first measured
five-component candidate run for the HeliosLM Agentic Index.

Runs ONE checkpoint through all five agentic components and writes the
native artifacts the adapters accept:

    terminal       run_term_bench() rows            artifacts/terminal.json
    swe            run_swe_bench() rows             artifacts/swe.json
    long_horizon   evaluate_policy() dict           artifacts/long_horizon.json
    tool_recovery  tool-recovery-run/1 object       artifacts/tool_recovery.json
    hdl            hdl-bench/1 dict                 artifacts/hdl.json

then generates the five ``agentic-component/1`` descriptors
(``evidence_type=model_run``, ``verification_tier=measured``) and the
aggregate ``agentic_index.json`` in the results directory.

Usage (from the repo root):

    python3 examples/agentic_candidate_run.py \
        --checkpoint checkpoints/toy_v5.13.pt \
        --results results/agentic/<run-name>

    python3 examples/agentic_candidate_run.py \
        --candidate mid \
        --checkpoint checkpoints/mid_sft_v5.33.pt \
        --results results/agentic/<run-name>

Determinism: greedy decoding (temperature=0), fixed task order, fixed
seeds for the tool-recovery corpus — a rerun on the same checkpoint and
task sets reproduces every artifact byte-for-byte (HDL wall-clock fields
are not part of the schema).

Candidate notes: the committed toy checkpoint is a char-ascii model with
vocab 1024 and 2048 positions; prompts are truncated head+tail to fit,
and characters outside the vocab are mapped to '?'. Both are candidate
limitations, recorded here — they are not harness modifications.

``--candidate mid`` wraps a BPE chat-SFT checkpoint (360M params, vocab
32768, 8192 positions — e.g. mid_sft_v5.33, fetched from the project's
Hugging Face repo; ``*.pt`` is gitignored) via examples/mid_candidate.py.
The mid model was SFT-trained on the chat tool protocol (``##user##`` /
``##assistant##`` / ``##tool##`` with ``@@tool@@`` blocks and the
calc/str_op/file_read/file_write/finish tools), so the bash / policy /
Verilog prompts of these five tracks are out-of-distribution for it —
see docs/AGENTIC_MID_CANDIDATE.md. That is a candidate limitation, not a
harness modification.

HDL data: fetched from the pinned upstream commit. ``--hdl-cache`` may
point at an existing ``hdl_bench.fetch`` cache; otherwise the runner
fetches via one codeload tarball (raw.githubusercontent is intermittent
in some sandboxes) and verifies every sha256 through the normal
``hdl_bench.load`` path. Untrusted generated samples are simulated by
hdl_bench in per-sample temporary directories under timeouts; run this
inside a container, as the hdl_bench safety note requires.
"""
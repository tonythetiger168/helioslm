# HeliosLM v5 Changelog

## v5.43 (2026-10-08) - Vanguard 2 Result + Glimmer Script
- Gated hybrid (soft convex weight per benchmark, cross-val learned):
  median 0.655 -- BELOW pure TF-IDF 0.774. Finding: oracle per-bench
  selection (Level-1 0.796) is an OPTIMISTIC upper bound (test-set leg
  choice), not a reproducible system; Qwen-enc drags down soft voting
  on 35/41 benchmarks (w=1.0 chosen). Honest feature-engineering
  ceiling is ~0.77, not 0.796
- benchmarks/gated_hybrid_results.json
- examples/alignbench_glimmer_ft.py: Vanguard 1 -- RLCD fine-tune of
  Muse Glimmer-30B (agent-tuned Apache-2.0, 4-bit 24GB, single 4060).
  Tests the instruct inductive-bias hypothesis that the Qwen3-0.6B
  base-model failure (10/05) left open. If Glimmer has alignment-
  detection inductive bias, 0.796 -> 0.85+ is the path past Jev 0.911



## v5.42 (2026-10-08) - RSI Routes 2 + 6 Prototypes
- agent/self_gen_env.py: AI-generated curriculum (Absolute Zero minimal
  version) -- model generates expr/string tasks, shared grammar verifies
  well-formedness, independent verify checks if the model can solve its
  own task (training-value check, survey route 2)
- agent/meta_prefetcher.py: meta-level improvement (survey route 6) --
  CalibratedPrefetcher's own hi/lo thresholds adapt from prefetch
  hit-rate outcome. Verified: converges stably, no false triggering
- T41 3/3



## v5.44 (2026-10-08) - Prefill Stability + Quant-Calib Pipeline + Async EnvGRPO
- `eval/sparse_stability.py`: `prefill_stability` — the v5.43 decode-step
  top-k self-consistency probe extended to EVERY query position (batch 0).
  For each position: reference top-k set + n_trials noisy sets (Gaussian
  std on the query side, cache untouched) -> per-position mean pairwise
  Jaccard; overall score = unweighted mean over positions. Documented
  simplification: every position is scored against the SAME shared cache,
  not the causal prefix (a pure function of (query, cache) matches the
  deployed DSA call). Shared loud-input guards refactored into
  `_validate_probe_inputs` / `_pairwise_jaccards`.
  Test: `test_prefill_stability` (zero-noise and full-selection oracles
  return exactly 1.0; same-seed determinism; mean == mean(per_position);
  loud guards on top_k/n_trials/noise_std)
- `examples/train_indexer_distill.py`: `indexer_prefill_stability(model,
  ids, top_k, ...)` runs the probe per MLA layer against the layer's own
  latent cache (same projection path as `teacher_overlap`);
  `distill_indexer(..., stability_probe=)` records per-layer
  `stability_initial`/`stability_final` (mean Jaccard) around training;
  `--stability-probe` CLI flag. Probe is mechanical stability only — a
  stable-but-wrong indexer scores 1.0; quality remains the distill loss's
  job (honest docstring, no implied quality claim)
- `eval/quant_calib.py`: `ece_drift_from_report(report)` recomputes
  nvfp4_ece - bf16_ece from the two probe arms, never trusting the
  report's embedded "drift.ece" copy (a hand-edited / lossy-serialized
  report could carry a stale drift, and TrustGate's widen-only policy
  rides on this number). Missing arm or missing "ece" -> loud ValueError
  (no silent default-drift manufacturing).
  Test: `test_trust_calibration_from_report`
- `agent/trust_gate_v2.py`: `trust_calibration_from_report(report, n=None)`
  — the quant-calib probe -> TrustGate v2 pipeline. Anchors on the bf16
  arm and delegates the merge to `apply_quant_drift` (widen-only:
  max(bf16, nvfp4) ECE; a quantization that improved calibration never
  makes the gate more aggressive than its bf16 evidence). Returns a
  record {"ece", "n", "quant_drift", "policy", "source"} that passes
  TrustGateV2's own validation; verified end-to-end that a +0.03 drift
  widens the band so p=0.84 against p*=0.8 ESCALATEs.
  Test: `test_trust_calibration_from_report` (widen-only on both signs,
  stale embedded drift ignored, n override, partial reports loud).
  Note: 0.02 - 0.05 under IEEE754 double is -0.030000000000000002 —
  the test asserts that side with a 1e-12 tolerance (quantified float
  noise, threshold not relaxed semantically)
- `src/training/env_grpo.py`: `AsyncEnvGRPO(AsyncGRPO)` — the env-routed
  reward path on the v5.31 producer/consumer skeleton. Workers produce
  (task, samples) groups (each sample env-tagged, back-pressure-aware
  put, cooperative stop); the learner scores every group with
  `MultiEnvBatch.reward` and calls the SHARED `_learn_from_samples(
  samples, None, rewards=...)` — env rewards bypass `compute_rewards`,
  the update math is byte-identical to the synchronous EnvGRPO loop.
  Asynchrony changes WHO produces samples and WHEN, never the
  mathematics (AsyncGRPO's documented toy-scale caveats carry over: no
  weight-version sync, thread-level not process fleet). Every metrics
  dict carries its task's "env" for per-env audit.
  Test: `test_async_env_grpo` (3 tasks consumed exactly once under
  G=2, env tags on every metrics dict, finite losses, non-negative k3
  KL, compute_rewards bypass proven by monkeypatched raise, same-seed
  run reproduces the env sequence, loud construction guards)

## v5.43 (2026-10-06) - Env-Wired GRPO + Sparse Stability + Quant-Drift Gate
- `src/training/env_grpo.py`: end-to-end multi-env GRPO loop — the wiring
  the v5.42 report explicitly left open. `MultiEnvBatch` tasks ->
  `GRPOTrainer._sample_response` (G per task) -> env-routed
  `MultiEnvBatch.reward` -> the SHARED update math via a new
  `rewards=` injection point on `_learn_from_samples`
  (`compute_rewards` bypassed, never reimplemented; math byte-identical).
  Per-env advantage normalization stays caller policy (same disclaimer as
  `MultiEnvBatch`). M-T1 mode discipline replicated for sampling.
  Test: `test_env_grpo_loop` in tests/test_v5.py (rollout routing checked
  against independent env scoring; injected rewards verified to bypass
  `compute_rewards`; wrong-length reward tensor fails loudly)
- `eval/sparse_stability.py`: learned-sparse top-k self-consistency probe
  — mean pairwise Jaccard of the lightning indexer's top-k sets under
  repeated query-side Gaussian perturbation (deterministic via seeded
  generator). Query-side only (decode-time cache is frozen by contract);
  mechanical stability, NOT selection quality. Oracles: zero-noise ->
  Jaccard exactly 1.0; top_k == kv_len stable under any noise; loud
  errors on bad inputs. Test: `test_topk_self_consistency`
- `agent/trust_gate_v2.py`: `apply_quant_drift` — merge an NVFP4 ECE
  drift (`ece_quant - ece_bf16` from the v5.42 quant-calib probe) into a
  gate's calibration record under a declared WIDEN-ONLY policy: positive
  drift widens the abstain band by exactly the drift, a negative drift
  never sharpens it, clamp at the gate's 0.5 validation ceiling, full
  provenance fields kept (drift, policy, source). Test:
  `test_quant_drift_trust_gate` (end-to-end: same p=0.84 DIRECTs on a
  tight bf16 band, ESCALATEs after a +0.10 quant drift widens the band)

## v5.42 (2026-10-06) - Decision-Engine Tooling Wave (Phase 1.3 + Phase 2 + Phase 3)
All pieces oracle-tested; claims scoped to what the oracles prove (see
each module's docstring honesty notes).
- Phase 1.3: `tests/test_decision_head_integration.py` (3/3) — DecisionHead
  wired onto real HeliosLMv5 hidden states; end-to-end gradient flow to
  trunk verified; recorded finding: the decision path BYPASSES lm_head
  (zero/None grad by design); aux-loss-free route_bias carries no backprop
  gradient (heuristic-only, design); synthetic routing learned through the
  full model graph to acc 1.000
- Phase 2.1: `src/training/envs.py` — multi-env GRPO registry: MathEnv
  (constructed numeric truth), CodeEnv (RESULT-line literal compare,
  ast.literal_eval only, never exec), AlignmentAuditEnv (RLCDAlignBench
  label convention, refuses synthetic data); MultiEnvBatch mixing and
  routing (5/5 oracles)
- Phase 2.2: `eval/longctx.py` — passkey-by-loglikelihood long-context
  harness (no generation, deterministic); refuses rotary extrapolation
  beyond max_position_embeddings without explicit rope_scaling; records
  model/config provenance with every number (5/5 oracles; perfect stub
  100%, uniform stub ~chance, untrained lite measured as harness smoke)
- Phase 2.3: `eval/quant_calib.py` — calibration x quantization cross
  probe: ECE + accuracy for bf16 vs NVFP4 fake-quant twins of the same
  weights (caller model never mutated; deepcopy inside); ECE math exact
  on calibrated stubs (0.0000) and overconfidence stubs (0.1900) (4/4)
- Phase 3.1: `src/inference/pool_ledger.py` — prefix-pool accounting
  ledger via stats() deltas (zero invasive change to the verified pool);
  effective prompt-token discount + per-request entries; decode cost
  explicitly out of scope (3/3)
- Phase 3.2: `agent/trust_gate_v2.py` — TrustGate v2 calibrated
  probabilistic abstention: cost-sensitive threshold p* = 1 - ce/cw,
  abstain band widened by the head's measured ECE, uncalibrated operation
  loud-tagged with conservative margin (5/5)
- Phase 3.3: `eval/mini_bench.py` — task-level JSONL benchmark harness
  (choice-loglikelihood scoring, per-benchmark aggregation, load-time
  schema loud-errors) (3/3); choice_confidence generalized to
  token-SEQUENCE choices (position-faithful)
- Phase 3.4/3.5: README GIF evaluated and held (asciinema/agg toolchain +
  a checkpoint whose output is worth showing required; script remains
  ready); r/LocalLLaMA v5.42 draft appended to docs/launch_post_reddit.md
  with explicit HOLD-until-Phase-2-results note
- Also: docs/ROADMAP.md + docs/PHASE0_3_PLAN.md reconstructed from
  LLM_HANDOFF.md (originals never reached origin); 25 large data
  artifacts untracked (.gitignore; on-disk copies preserved)
- Verification: 71/71 test_v5 + 9/9 integration + all suites green

## v5.41 (2026-10-06) - Line Unification Merge + Decision-Engine Phase 1
- Merge of the two diverged lines (local ahead 4 / remote ahead 160+ since
  2026-09-23): the local maintenance line (KDA per-channel decay gate v5.21,
  NVFP4 QAT + NoPE v5.22-L, learned lightning indexer) reunites with the
  calibration/RLCD line (v5.23-v5.40b: agent layer, disagg evolver, chat
  capability, TrustGate, continuous noul, RLCDAlignBench, level-1 ceiling
  0.796, level-2 RLCD-FT experiment). Conflict resolutions: README and
  CHANGELOG keep both histories (local v5.22 keeps a `-L` suffix to
  disambiguate from the remote v5.22 decision-layer audit toolkit);
  `config_v5.py` keeps the remote size-dependent linear-head config AND the
  local `per_channel_decay` flag; `linear_attention.py`, `mla.py`, `qat.py`,
  `test_v5.py` auto-merged with both sides' features intact
- Version string unified at v5.41 (next free number after remote v5.40b)
- Phase 1.1/1.2: `src/decision_head.py` — LM-backed typed decision readout
  (noul P(yes) / choice softmax / score) over the model's own hidden
  states, the LM-facing piece of the local-decision-engine direction;
  oracles in `tests/test_lm_decision_head.py` (5/5). Distinct from the
  toy-scale `agent/decision_head.py` (v5.32), which keeps its own oracles
- Phase 1.4: `src/training/dpo.py` — DPOTrainer (sigmoid-margin loss,
  sequence-level log-probs mirroring the GRPO simplification, frozen
  reference enforced, beta exposed); analytic anchor oracle
  (pi==ref => loss == log 2) plus margin-training and frozen-reference
  oracles in `tests/test_dpo.py` (6/6)
- Verification: 71/71 test_v5 + 9/9 integration + 34 suites + 11 new
  oracles all green at push time

## v5.39c (2026-10-04) - Level 1: Hybrid/Select Ceiling Measured
- benchmarks/level1_hybrid_results.json: TF-IDF 0.774 / hybrid 0.714 /
  Qwen-enc 0.714 / PER-BENCH-SELECT 0.796 (44 benchmarks, oracle leg
  selection on test AUROC). Three findings: (1) CONCAT HURTS -- Qwen
  hidden states add noise to TF-IDF (-0.06), gate beats concat; (2)
  leg selection adds +0.022 over the best single leg; (3) the
  feature-engineering ceiling is ~0.80 -- the remaining 0.06 to Jev's
  0.859 is TRAINING, not engineering. Level 2 (RLCD fine-tune of the
  readout on Qwen3-0.6B) is the recorded and only remaining path



## v5.38j (2026-10-04) - Paper v4b: Direction 1+2 Deep Dives In
- sycophancy decomposition: the inversion is REAL and self-referential
  (P(yes|flattery)=0.890 vs P(yes|control)=0.953, -0.063) -- the
  detector's own sycophancy mirrors the failure it detects. Separately,
  a ceiling-effect audit: 15/16 benchmarks have class-separation effect
  size < 0.03 -- the harmbench 0.879 comes from distribution tails, not
  calibration. Prompt-zero has no calibration ability even on
  surface-cue classes -- STRONGER support for 'calibration requires
  training' than the median alone
- mid-vs-Qwen LOBO gap: specialization hypothesis REJECTED (mid -0.164
  < Qwen -0.246, opposite of prediction); recorded honestly as
  observed-but-unexplained with the capacity confound noted
- benchmarks/direction12_analysis.json + paper v4b PDF



## v5.38i (2026-10-04) - Prompt-Zero Leg: the Strongest Argument FOR Training
- Qwen3-0.6B base, fixed template, P(yes) from full distribution, ZERO
  training: median 0.500 with extreme bimodality -- harmbench 0.879,
  reward_hacking 0.700, instrumental 0.722, verbalized_ece 0.629, but
  sycophancy 0.289 INVERTED (the base model is itself sycophantic about
  flattery -- an alignment failure measured INSIDE the detector) and
  confaide/bias at pure chance. A single fixed prompt has no general
  zero-shot calibration ability
- FINDING: this failure mode is the strongest independent support for
  the commercial decision-layer category's core claim -- calibration
  REQUIRES training. Recorded as the closing row of the ablation
- Paper v4: three new rows (MLP-LOBO 0.589, prompt-zero 0.500),
  ablation rewritten with the training-requirement conclusion
- benchmarks/alignbench_prompt_zero_readout.json



## v5.38h (2026-10-04) - Fine-Tuned Readout Head: Transfer Gain Confirmed
- MLP head (1024->256->1) LOBO median 0.589 vs LR 0.561 (+0.028, seed 0,
  300 epochs): fine-tuning the readout releases a modest but real
  transfer gain on frozen Qwen3-0.6B features. Direction confirmed,
  magnitude small -- prompt-based zero-shot remains the higher-leverage
  recorded direction
- benchmarks/alignbench_mlp_lobo.json records the protocol and result



## v5.38h (2026-10-04) - Fine-Tuned Readout Head: Transfer Gain Confirmed
- MLP head (1024->256->1) LOBO median 0.589 vs LR 0.561 (+0.028, seed 0,
  300 epochs): fine-tuning the readout releases a modest but real
  transfer gain on frozen Qwen3-0.6B features. The frozen-feature
  ceiling is real, not absolute -- but the gain is small enough that
  prompt-based zero-shot remains the higher-leverage direction
- benchmarks/alignbench_mlp_lobo.json records the protocol and result



## v5.38h (2026-10-04) - Fine-Tuned Readout Head: Transfer Gain Confirmed
- MLP head (1024->256->1) LOBO median 0.589 vs LR 0.561 (+0.028, seed 0,
  300 epochs): fine-tuning the readout releases a modest but real
  transfer gain on top of frozen Qwen3-0.6B features. The frozen-feature
  ceiling is real, not absolute -- but the gain is small enough that
  prompt-based zero-shot remains the higher-leverage direction
- benchmarks/alignbench_mlp_lobo.json records the protocol and result



## v5.38g (2026-10-04) - Paper v3: Encoder Legs Complete
- Qwen3-0.6B hidden supervised 0.807 / LOBO 0.561; mid 360M hidden
  supervised 0.692 / LOBO 0.528 (user's GPU runs, 90s each). Encoder
  capacity hypothesis VERIFIED: same readout protocol, encoder ladder
  0.561 -> 0.692 -> 0.807, closing 86% of the gap to Jev zero-shot
  (0.859) and BEATING word-level TF-IDF by +0.08. LOBO transfer rises
  with encoder quality (0.542 -> 0.561) but plateaus near 0.56 --
  frozen features carry some transferable signal (abstentionbench LOBO
  0.933) yet fine-tuned readout heads are the recorded next step
- Paper v3: six-row table + rewritten ablation; PDF rebuilt (tectonic)



## v5.38f (2026-10-03) - Paper v2 + Local Encoder Scripts
- Paper v2 (565a6c16): LOBO row (0.542, near-zero transfer finding),
  encoder ablation paragraph, LM-encoder legs marked in-progress
- examples/alignbench_qwen_encode.py + alignbench_mid_encode.py:
  runnable locally on GPU (~5 min each on a 4060); produce qwen_feats.npz
  / mid_feats.npz for the readout leg. Sandbox legs checkpointed at
  1400/2329 but the FUSE filesystem instability killed them repeatedly;
  local runs are the reliable path



## v5.38e (2026-10-03) - Paper PDF Built and Distributed
- docs/paper_draft_2026-10-03.pdf: tectonic build, 7pp, 3 figures
  embedded, checked into GitHub (eb87ad45) and HF
  (commit f356f081, chienhsinlin/helioslm docs/). HF docs/ is now the
  complete trio: .tex (source) + .md (reading) + .pdf (artifact)
- hf_upload.py second routing bug fixed: inline branch corrupted
  binary files (PDF) via errors=replace -- 400 from the commit API.
  Routing is now content-based (strict utf-8 probe), not size-based
  (12a368fb). Same lesson as the LFS-pointer bug, in reverse



## v5.38d (2026-10-03) - LaTeX Version for arXiv
- docs/paper_draft_2026-10-03.tex: arXiv-ready article (booktabs
  tables, natbib, thebibliography). Self-contained; compiles with
  pdflatex. Submission path: arXiv cs.LG now, TMLR next (reproducibility
  certification target), NeurIPS 2027 main after the zero-shot leg



## v5.38c (2026-10-03) - Paper Draft
- docs/paper_draft_2026-10-03.md: full working paper (abstract,
  intro, related work, stack, evidence chain, trilogy, AlignBench
  comparison, limitations, reproducibility). Every number traces to a
  versioned artifact; honest framing throughout (supervised legs,
  single-point scale, encoder bottleneck)



## v5.38b (2026-10-03) - TF-IDF Leg + Charts: 11 Benchmarks Beat Jev Zero-Shot
- TF-IDF leg (word 1-2 + char_wb 3-5, 20K features each, LR C=1): median
  0.726 (char 0.561 -> BPE 0.569 -> TF-IDF 0.726). WE BEAT Jev zero-shot
  on 11/41 comparable benchmarks (up from 4): sycophancy +0.34, faith_mt
  +0.31, confaide +0.31, injecagent +0.22, abstentionbench 0.986 vs 0.868
- Median gap to Jev (0.859) remains: product model vs research encoder;
  our TF-IDF leg sits at the paper's own baseline feature class
- Charts in benchmarks/charts/ (median summary, scatter, per-benchmark
  bars); scatter marks the 11 wins over the y=x line
- C tuned (0.5/1/2), feature budget 20K vs 30K: 20K+char wins; recorded
  config for reproduction



## v5.38a (2026-10-03) - RLCDAlignBench First Numbers (7,193 instances)
- benchmarks/alignbench_comparison_2026-10-03.txt: three-way table,
  41 comparable benchmarks. Jev zero-shot (its own cached metrics,
  recomputed offline from the gated dataset) median 0.859 (paper
  reports 0.886; delta = variant/battery selection, recorded); our
  SUPERVISED DecisionHead median 0.561 char / 0.569 BPE
- Honest read: the median gap is the honest-minimum encoder (char/
  BPE mean-pool, T35-documented bottleneck), NOT the RLCD readout
  protocol -- the same readout on per-benchmark surface-cue tasks
  matches or beats Jev zero-shot (sycophancy 0.71 vs 0.43, injecagent
  0.80 vs 0.61, open_prompt_injection 0.67 vs 0.51, abstentionbench
  0.88 vs 0.87). We are below the paper's TF-IDF LR baseline too
  (0.75-0.97): word n-grams >> mean-pool at this scale
- Method lesson recorded: the benchmark's state schema is PER-BENCHMARK
  (10+ shapes); the first run rendered empty states for 40/44
  benchmarks (median 0.500) before switching to generic all-field
  rendering (0.561). Any fixed-field assumption on heterogeneous
  benchmarks degrades SILENTLY
- Runner updated: label is it['label'] directly (1=failure); generic
  state render; in-domain BPE leg (4K vocab trained on the data
  itself, encodings cached)



## v5.38 (2026-10-03) - RLCDAlignBench Runner Skeleton
- examples/alignbench_run.py: our stack on the canonical benchmark
  (sumleo/RLCDAlignBench, gated). Legs: inspect (field mapping),
  head (supervised DecisionHead per benchmark -- the honest TF-IDF-
  baseline analog; the paper headline is ZERO-SHOT Jev, a different
  question), jev (offline cache recompute, no key). Protocol mirrors
  the paper: generic prompt+response state, P(yes) as score, AUROC per
  benchmark. BLOCKED on gated access (user to request at the dataset
  page with the chienhsinlin HF account)



## v5.37m (2026-10-03) - The 8.4x Intervention Contrast, Archived
- benchmarks/trust_dual_prefix_console_2026-10-03.md: console record
  of the v5.37k dual-prefix run. ungrounded_p 0.023 / grounded_p 0.190
  (8.4x), tiers emerged (0.71 high / 0.47 medium / <0.3 low); 4
  DIRECT-[grounded] 4/4 wrong without grounding -> TherapyPair
  composition finding; v5 refusal fired then died on oracle.decide
  (fixed in v5.37l). Report evidence chain gains claim 4b



## v5.37o (2026-10-03) - Top-Level Import Actually Added
- v5.37n's import patch replaced the LAZY import inside
  ReviewOracle.decide (first textual occurrence) instead of adding a
  top-level one; main() still had no FixedGate. Same trap family as
  v5.36g: anchor on the wrong occurrence, assert passed on the wrong
  string. Header-scoped verify this time



## v5.37n (2026-10-03) - Refusal Accounting + Import Fix
- v5 rerun (v5.37l oracle fix): 9/12 correct, 3 write_read refusals
  with ZERO hallucination leakage -- but the 360M model recovers by
  QUOTING the abstention text as its answer, not re-walking the chain.
  v5 eval now counts abstained separately (not as wrong): therapy
  outcome is 9 correct + 3 abstained + 0 leaked
- eval_mid_trust: FixedGate import fix (v5.37l replace missed the
  actual import line; NameError before the dual-prefix+TherapyPair run
  could start)



## v5.37m (2026-10-03) - The 8.4x Intervention Contrast, Archived
- benchmarks/trust_dual_prefix_console_2026-10-03.md: console record
  of the v5.37k dual-prefix run. ungrounded_p 0.023 / grounded_p 0.190
  (8.4x), tiers emerged (0.71 high / 0.47 medium / <0.3 low); 4
  DIRECT-[grounded] 4/4 wrong without grounding -> TherapyPair
  composition finding; v5 refusal fired then died on oracle.decide
  (fixed in v5.37l). Report evidence chain gains claim 4b



## v5.37l (2026-10-03) - Therapy-Pair Composition + 8.4x Intervention Contrast
- Dual-prefix trust run: ungrounded_p_mean 0.0228 vs grounded_p_mean
  0.1904 -- intervention conditioning VERIFIED on real weights; tiers
  emerged (high ~0.7 / medium ~0.47 / low <0.3 firing in one run)
- FINDING: 4 DIRECT [grounded] tasks were 4/4 WRONG -- the trust
  harness executed them UNTREATED. Trust calibrated on grounded
  outcomes is only honorcd when the run is ACTUALLY grounded: the
  therapy pair must be COMPOSED, not used alone. TherapyPair wrapper
  (TrustGate routes, GroundingGate fills) added to eval_mid_trust;
  [grounded] prefix now runs composed, [ungrounded] stays bare
- v5 AbstainOracle gained decide() (GroundingGate.decide consults
  inner.decide first; the missing method crashed the rerun at task 10,
  refusal itself had fired correctly)



## v5.37k (2026-10-03) - Acceptance Harness Fixes (Refusal Fired, Harness Died)
- First v5.37j rerun: the premature-finish refusal fired EXACTLY as
  designed (9/9 calc/str correct, then write_read task 10 finish-first
  -> ESCALATE) but FixedGate has no oracle and the harness crashed --
  refusal correct, assembly incomplete. v5 gains AbstainOracle: the
  session receives the abstention note and the model can re-walk the
  evidence chain
- eval_mid_trust runs BOTH intervention prefixes (24 tasks total):
  [ungrounded] alone legitimately reads all-distrust (training says
  ungrounded fails 12/12); the TIERED demonstration is the contrast
  metric ungrounded_p_mean vs grounded_p_mean on fresh identical tasks



## v5.37j (2026-10-03) - Known-Limitations Batch: #1 + #3
- eval_mid_trust v2: MIXED outcome records (v4 12 no + v5 9 yes/3 no)
  with an INTERVENTION prefix ([ungrounded]/[grounded]) -- the trust
  question is P(correct | state, intervention); the 10-02 family-wide
  abstention came from all-negative data missing this dimension
- GroundingGate: PREMATURE FINISH on a groundable multi-step family
  with an empty state machine now ESCALATEs (the 10-02 write_read
  bypass: finals '1 * 1'/'36' passed through ungrounded). Single-step
  families exempt (calc/str finish after one obs is legitimate).
  Sequence-level refusal inside grounding; TrustGate remains the
  state-level counterpart
- T33 7/7 incl. the new premature-finish oracle (finish-first refused,
  calc-then-finish still DIRECT)



## v5.37i (2026-10-02) - Trilogy Artifacts Archived
- benchmarks/mid_{agent_eval_v4,agent_eval_v5,sft_v5.33}_trilogy.json:
  the raw evidence for the self-consistent trilogy (fresh 360M
  checkpoint, paired tokenizer, fingerprint verified). v4 carries full
  per-step transcripts: 0/12 with fresh confabulations (args corruption
  dominant, obs->finish copying intact on calc/str); v5 9/12 with
  verbatim-correct finals on all cured tasks; SFT summary 30/40 + 40/40
- Trust verdict (from console; json local): 12/12 abstained, p_min ~2e-5



## v5.37h (2026-10-02) - The Self-Consistent Trilogy Closes
- Fresh baseline retrain (6728s, loss 0.4129, SFT eval 30/40 + 40/40 --
  reproduces the 9/28 baseline within seeded noise), artifacts
  durable: .pt + .tok.json + .tokfp on HF (chienhsinlin/helioslm,
  commits ab69b699/a784781b/0494ec68)
- Trilogy on ONE checkpoint, fingerprint verified: v4 ungrounded 0/12
  (fresh confabulations, disease reproducible); v5 grounded 9/12 (all
  calc/str cured; 3 write_read failures = finish-first sequences --
  grounding is SEQUENCE-DEPENDENT, it cannot inject into an empty
  state machine; premature finish is TrustGate's territory); TrustGate
  12/12 abstained at p~2e-5 (decisive; all-negative training data ->
  family-wide abstention, tiered behavior needs mixed labels)
- Report sections 3-4 updated to the self-consistent numbers



## v5.37g (2026-10-02) - hf_upload.py Ships in the Repo (Durability Fix)
- The 10-02 post-mortem: the mid baseline weights existed ONLY on the
  user's disk (HF_SYNC was never set locally; the uploader lived only
  in a sandbox /tmp file), and a local overwrite destroyed them. The
  uploader now ships in the repo (stdlib only, token from env) so the
  artifact pipeline has no environmental single point of failure



## v5.37f (2026-10-02) - Tokenizer Fingerprint Guard in the Eval Harness
- _mid_common.load_model_tok verifies the saved .tok.json against the
  training-time .tokfp sha256; mismatch exits loudly with the fix
  recipe. The 10-02 trust-eval run produced confident soup with a
  starved gate (p_min=None, zero decisions) because a rebuilt
  tokenizer drifted under a changed corpus -- exactly the failure mode
  this guard now refuses silently-reproducing. Structural guard for
  the format/training-distribution skew family, part 1



## v5.37e (2026-10-02) - TrustGate Acceptance Harness Fix (v1 Trap, Third Occurrence)
- eval_mid_trust v1 drove AgentLoop (Task:/step i: format) -- the v1
  format-skew trap, third occurrence of that family (v1 -> v2 fix ->
  this). mid SFT renders chat transcripts; under the wrong harness the
  model emits no tool blocks (final=None everywhere) and parse failures
  never reach the gate, silently voiding the abstention measurement
- v2: ChatSession + TrustGate(inner=ReviewOracle); abstained tasks end
  final=None BY DESIGN (do not act). Recorded with the honest count:
  the format-skew family is now THREE documented occurrences -- the
  lesson exists in eval_mid_agent_v2.py's docstring and was still
  re-walked; a structural guard (harness-format assertion against the
  training distribution) is the real fix, queued



## v5.37d (2026-10-02) - Working Report: the RLCD Methodology Paper
- docs/report_rlcd_methodology_2026-10.md: full narrative -- three RL
  paradigms, the seven-link evidence chain (all measurements), the
  therapy pair, honest boundaries, and the task-INTRA positioning vs
  RLCDAlignBench. Every claim cites an artifact and a test



## v5.37c (2026-10-02) - TrustGate x mid Acceptance Script
- examples/eval_mid_trust.py: trains the trust head on the REAL v4
  failure outcomes (12/12 confabulated), then drives fresh mid eval
  tasks behind TrustGate -- expected behavior: ESCALATE on the
  failure-class states (the model is DENIED ungrounded action on its
  own measured failure territory), DIRECT only where trusted. The
  therapy pair, verified end to end on real weights



## v5.37b (2026-10-02) - Engineering Debt: Shared Harness + Grounded Replay
- examples/_mid_common.py: the three mid eval scripts' identical
  preambles (paired-tokenizer + checkpoint load + greedy model_fn with
  per-call confidence capture) now live in one place; scripts pass
  max_new/fewshot and get (model, tok, make_fn)
- grounding.verify_grounded_replay: grounded trajectories are now
  formally re-verifiable -- raw text -> parse -> a FRESH GroundingGate
  reproduces the recorded observations exactly (deterministic tools +
  shared grammar), tamper detection intact. The v5.34 docstring replay
  deviation is CLOSED
- T39 2/2; T33 regression green



## v5.37 (2026-10-02) - TrustGate: Calibrated Abstention, Deployed
- agent/trust_gate.py: the TypeSafe behavior tiers as a Gate wrapper.
  The trust question (continuous Noul, v5.36) is P(the routed action
  will be correct), learned from OUTCOMES (RLCD); the model's own
  confidence is not consulted (measured signal-free at mid). Tiers:
  P >= hi -> DIRECT auto-execute; lo <= P < hi -> DIRECT flagged
  medium; P < lo -> ESCALATE (do not act). Abstain without an oracle is
  a LOUD NotImplementedError, never silent; with inner= it delegates.
  Every decision lands in an audit log (p_trust/tier/route)
- The therapy pair closes: grounding fixes actions the model TAKES;
  TrustGate withholds actions on states the head DISTRUSTS
- T38 4/4; decision suite 7/7 incl. T38



## v5.36h (2026-10-02) - T30 Labels Actually Swapped (block-verified)
- v5.36g's replace targeted a severity line that never existed in the
  v5.36f file -> silent no-op; outputs were bit-identical to the
  previous run, the tell that the edit had not landed. Marker-based
  block-anchored swap this time, verified block-level BEFORE push
  (lesson recorded: identical outputs across a supposed label flip =
  the edit did not take)



## v5.36g (2026-10-02) - T30 Labels Un-swapped
- The continuous readout oracle labeled clean->no / risky->yes; the
  head learned the majority class as the anchor and inverted the
  direction. Semantics: trust P(yes) should be HIGH for clean states.
  Un-swapped. (A flipped-labels run is itself the classic calibration
  bug -- worth the record.)



## v5.36f (2026-10-02) - T30 Data Fix (Real Labels, Real Assert)
- trust targets were neutral (both classes 0.5-ish) -- no learnable
  signal, direction unassertable; the v5.36e direction-only assert
  papered over it. Fix: real labels (clean->no, risky->yes) and the
  strong assert restored. Real signal, real oracle



## v5.36e (2026-10-02) - T30 Margin Assert -> Direction
- CalibratedRouter synthetic registers neutral targets for trust; the
  trained margins sit near 0.5 with tiny separation. Assert DIRECTION
  (p_risky < p_clean), not magnitude -- no fake convergence claims



## v5.36d (2026-10-02) - Noul Targets Accept Continuous Values
- _targets noul path: yes/no strings map via _NOUL_IDX; raw numeric
  targets (probabilities) pass through; None fills 0.5 (neutral).
  Reverse-lookup KeyError'd on 0.1 and None (found in T28/T30)



## v5.36c (2026-10-02) - Continuous-Noul Suite Fully Green
- decision_head._targets: score targets tolerate missing answers
  (fill 0.0) -- records may register more heads than they annotate
- test_decision_head: severity->trust rename completed (set assertion
  included)
- T27/T28/T30/T35 + T37 all green under the continuous API; decision
  suite closes at 6/6



## v5.36b (2026-10-02) - Continuous-Noul Test Fixes
- decision_head._targets: score targets use raw floats, noul targets
  use _NOUL_IDX -- the paths are distinct (mixing them KeyError'd on
  None in test_calibrated_router)
- test_decision_head: forward() set assertion synced to the renamed
  'trust' question
- test_rlcd_head: decide() returns (P, None) post-v5.36; the None is
  not a confidence
- T30/T28/T35 green under the continuous API



## v5.36 (2026-10-02) - Continuous Noul: the Floor Retires
- Noul is now P(yes) in [0,1], NO independent confidence (official
  TypeSafe form; triple-evidenced: our 1/K floor finding + official spec
  + AlignBench readout). Uncertainty IS a probability near 0.5. The
  ternary form and the certainty-Score escape hatch are retired
- decision_head: noul head emits one logit; BCE + outcome-Brier on the
  probability; evaluate reports P-level MAE
- gate.ask noul defaults: DIRECT 0.2 / ESCALATE 0.8; CalibratedRouter
  gains uncertainty() and drops certainty()
- T37 3/3 incl. the strict floor oracle (sub-0.5 targets expressible).
  Decision suite green



## v5.35f (2026-09-29) - Accidental Probe: Confidence Scatters under Tokenizer Mismatch
- A re-run with a mismatched tokenizer (fresh-cleanup artifact) produced
  garbage outputs whose confidences SCATTERED (0.13-0.97) vs the clean
  run's uniform ~0.9999 on both classes
- FINDING: the model's own confidence already flags OOD input; the
  calibration gap is precise -- overconfidence on IN-DISTRIBUTION
  errors only. GroundingGate/TrustGate target exactly that case; the
  OOD case the model flags on its own. benchmarks/
  mid_agent_eval_v4_mismatch_probe.json archived



## v5.35e (2026-09-29) - Copy Curriculum: NO Effect at 360M (controlled A/B)
- User's GPU run: mid + 800 echo copy episodes vs baseline, same eval
  seed, same 12 tasks -> 0/12 BOTH, confabulations BYTE-IDENTICAL
  (`19 * -92` -> `12 * -9`, `mhyIdE9KQ` -> `wIIIIIIK9` on both runs)
- Three eliminations complete: not tokenizer (v5.33), not protocol
  (40/40 mode), not data volume/curriculum (this run). Remaining
  hypothesis: capacity/dynamics wall at 360M -- copy grounding does not
  emerge from SFT at this scale
- Consequence: GroundingGate is the correct answer at this scale, not
  a workaround. Retirement path for grounding -> larger-scale check /
  RL pressure (async GRPO ready) / architecture. benchmarks/
  COPY_CURRICULUM_VERDICT.md records the A/B



## v5.35d (2026-09-29) - TypeSafe/Jev Alignment Report
- docs/typesafe_alignment_2026-09-29.md: calibration of our open-source
  reference against the canonical vendor impl + the RLCDAlignBench
  paper (ICLR 2027 under review). Concept layer fully aligned (three
  primitives, one-forward batching, confidence tiers, risk-scaled
  thresholds); three design differences recorded
- ADOPT (triple-evidenced): Noul -> continuous P(yes) in [0,1], no
  independent confidence (our binary-choice 1/K floor + official spec
  + AlignBench readout all point the same way)
- Positioning: AlignBench is the task-INTER branch (detecting failures
  in OTHER LLMs, 7193 instances, 0.886 median AUROC zero-shot); we are
  the task-INTRA branch (an agent trusting its own tool calls). Same
  RLCD methodology, mutually reinforcing conclusions
- Two implementation adoptions queued: question ID/content split;
  select-vs-score data split for multi-question evals



## v5.35c (2026-09-29) - Copy Curriculum: the Training-Side Therapy
- agent/copy_curriculum.py: EchoEnv ('Repeat back exactly: "{s}"' ->
  plain-text reply, random high-entropy payloads so pattern completion
  cannot substitute for copying; family=echo, budget 1, exact verify),
  gen_echo_episode / build_copy_dataset (echo + chat mixing), and
  scripted_echo_policy shared by tests and SFT data gen
- train_mid_sft.py: MID_COPY_JSON env appends a copy dataset to the SFT
  stream -- the grounding retirement experiment is one GPU run away
- T36 4/4: env verify, copy through the v5.30 text channel via a real
  ChatSession (text-only, 10/10), dataset copy-exactness + ord
  discipline, determinism. Regression 8/8 groups



## v5.35b (2026-09-29) - RLCD on Real mid Outcome Data
- examples/train_rlcd_head.py: trains a DecisionHead on the 24 real
  outcome records from the mid before/after artifacts (v4 0/12 + v5
  12/12, conf ~0.9999 both). Head sees STATE ONLY; outcomes provide the
  RLCD signal (probabilities answer to outcomes, not preferences)
- Quantified on real data: model confidence carries ZERO signal
  (0.9999/0.9999 right vs wrong); the head separates to 0.645 vs 0.521
  with acc 0.625 at n=24 -- mid's confabulation is PARTIALLY
  state-predictable (benchmarks/rlcd_head_summary.json). RLCD-adjusted
  rewards: right 1.0, wrong -2.0 (lam=2)
- Measured (recorded): the honest-minimum char mean-pool encoder
  separates marked states but needs ~1500 epochs (embedding
  organization is the bottleneck); ECE 0.267 at that budget
- T35 2/2; regression 7/7 groups



## v5.35 (2026-09-29) - TASK_GRAMMAR: One Table, Render and Parse
- agent/task_grammar.py: the single source of truth for the task-text
  grammar (calc/str/compose/write_read/write_transform/accumulate/echo).
  Envs RENDER through it; grounding and test policies PARSE through it.
  The v5.34 debug chain fixed the same drift class three times in three
  copies; this makes producer/consumer skew structurally impossible
- Consumers refactored: grounding's regex table deleted (delegates to
  task_grammar.parse); the T33 confabulator lost all its regexes;
  toy_envs and file_env now build task text via grammar.render
- T34 4/4: roundtrip identity per kind, cross-kind contamination
  (write_read must not parse as calc), live-env agreement, echo kind
- Fixed-point oracle: every env-sampled task text satisfies
  render(parse(text)) == text (120/120); full regression 11/11 groups



## v5.34.2 (2026-09-29) - THE VERDICT: 12/12 Grounded (was 0/12)
- mid_agent_eval_v5 on the real mid checkpoint with GroundingGate:
  12/12 correct, finals matching expected verbatim (v4 same seed was
  0/12 with every answer confabulated at 0.9999 confidence). The
  before/after pair (benchmarks/mid_agent_eval_v4.json vs _v5.json)
  closes the loop: policy-in-code is sufficient therapy for the
  copy-shaped failure, at this scale and grammar
- docs/grounding_report_2026-09-29.md: the full narrative -- disease at
  scale, autopsy, therapy, synthetic proof (T33), real verdict, and the
  honest boundaries (therapy not cure; retirement path = copy
  curriculum + RLCD, both queued)



## v5.34.1 (2026-09-29) - Code Review: Semantic Finish Grounding
- HIGH finding fixed: finish grounding was keyed to the sequence's
  LAST EVENT; a mid-sequence deviation (accumulate without the closing
  sum calc) would have produced a grounded-but-wrong answer -- the
  disease the module exists to cure, re-introduced by the therapy.
  Finish now anchors to family SEMANTICS (accumulate sums its two
  observations regardless of sequence shape)
- Docstring precision: replay deviation now states the recorded call
  is itself the grounded (mutated) call; write_transform file_read
  redirection documented; _state growth/thread-safety documented
- T33 6/6 incl. the review oracle: sequence-deviation accumulate runs
  (no closing calc) score 4/4



## v5.34 (2026-09-29) - Deterministic Grounding: the Therapy, Deployed
- agent/grounding.py: GroundingGate wraps any gate; on DIRECT it
  rewrites tool args from the task text and finish answers from the
  deterministically predicted tool chain (tools are deterministic, so
  the prediction IS the obs). ESCALATE/oracle paths never touched;
  unknown task shapes pass through (groundable() reports coverage)
- T33 5/5 incl. the killer oracle: a policy that confabulates EVERY
  argument scores 24/24 across all four env families under grounding --
  the model keeps only the tool-sequence decision. Debug chain of five
  real bugs (parse order, tail-tolerant regexes, last-event tracking,
  write_transform str_op coverage, policy-side greedy regex) all fixed
  and recorded
- examples/eval_mid_agent_v5.py: the same mid checkpoint + GroundingGate
  -- the v4 0/12 confabulation verdict vs the grounded run is the
  before/after pair for the whole grounding thesis
- Replay deviation recorded: grounded transcripts differ from raw-gen
  replay by design; determinism (not old replay) is the T33 oracle



## v5.33 P8 (2026-09-29) - Autopsy CORRECTED: Two-Layer Brittle Copying
- Full v4 transcripts (benchmarks/mid_agent_eval_v4.json) refine P7: the
  dominant failure is task->ARGS confabulation ("19 * -92" -> "12 * -9";
  "mhyIdE9KQ" -> "wIIIIIIK9") -- the obs faithfully answers the wrong
  question. obs->finish copying works in ~7/12 (near-copies with dropped
  chars / digit swaps). The earlier "finish ignores obs" held only for a
  minority
- Menu upgrade: the System-One architecture now fills BOTH ends
  deterministically (policy parses task-derived args; finish = last
  obs) -- the model keeps only the tool-sequence decision. Deployable
  today with the existing gate



## v5.33 P7 (2026-09-29) - Autopsy: Content Confabulation at 0.9999 Confidence
- v4 full-capture eval (mid, healthy ckpt): all 12 finish answers are
  hallucinated content, not copy failures -- calc obs carried the right
  answer and was ignored; reverse tasks produced length-matched
  degenerate repeats (IIIIIIIII, 2v2v2v2v2v): shape statistics learned,
  input-output grounding absent
- Mid's regime named: FORM perfect, GROUNDING missing -- the purest
  demonstration yet that fluency != correctness, and the exact target
  profile for the certified-confidence line
- Menu: copy-curriculum data / policy-in-code grounding for DIRECT
  finish (deployable today) / RLCD on the (conf 0.9999, wrong) pairs



## v5.33 P6 (2026-09-29) - Clean Run: the Three-Point Scale Line Closes
- Third training run (MID_FRESH=1, post-reboot): 3676s, final loss
  0.3675, SFT eval 30/40 exact + 40/40 mode (reproduces the first run)
- Agentic eval on the HEALTHY checkpoint: 0/12 correct BUT protocol
  fluent -- parse works multi-turn, calc tasks show the exact
  calc->finish shape, all 12 finished; finish answers are wrong
- THE LINE (docs/scale_lineage_2026-09-29.md): 8.5M broken-at-tokenizer
  (conf 0.94) -> 360M protocol-fluent-but-wrong (conf 0.9999) -> Qwen3
  0.6B no-finish (conf ~1.0). Confidence-on-wrong GROWS with scale --
  the strongest evidence yet for the certified-confidence thesis
- Every earlier 0/12 was a measurement artifact (tokenizer mismatch /
  corrupted resume), recorded, not counted



## v5.33 P5 (2026-09-29) - Resume Guard: Tokenizer Fingerprint Mismatch
- Found on the user's second run: RESUME loaded step-4600 weights
  (old-tokenizer run) while the process had trained a NEW tokenizer
  (sorted-rglob fix); 88 mismatch steps drove final loss 0.47 -> 15.0
  and wiped exact-match (30/40 -> 0/40). The corrupted checkpoint also
  carried a 'paired' tokenizer that does not match its weights
- train_mid_sft.py: tokenizer sha256 fingerprint (.tokfp) written with
  the checkpoint; resume REFUSES on any difference (loud exit, never
  silent). Companion to the .tok.json pairing: pairing protects eval,
  fingerprinting protects training



## v5.33 P4 (2026-09-28) - Tokenizer Train/Serve Skew: Third of the Skew Family
- v3 prompt-diff oracle: prompts byte-identical (prompt_equal=True), yet
  the model emitted structurally-tool-block-shaped id sequences that
  decoded to source-code soup at confidence 1.0. Mechanism: eval
  processes RETRAINED the BPE tokenizer; build_corpus() rglob order is
  filesystem-dependent, so the retrained vocab assigned different ids to
  the same strings -- a healthy checkpoint decoding through a mismatched
  table produces confident garbage (ids of never-trained embeddings sit
  at init)
- Fixes: build_corpus() sorts file lists (root cause); train_mid_sft
  saves the tokenizer WITH the checkpoint (.tok.json); eval scripts load
  the paired tokenizer first, retrain only as a warned fallback
- Skew family now three: v5.30.1 dataset filter bias -> v5.33 format
  mismatch -> v5.33 tokenizer mismatch. Standing rule: NOTHING between
  train and serve may rely on 'should be the same'
- The prior checkpoint (no saved tokenizer) needs one retrain (~1 h) to
  pair; v2 agency numbers after the retrain are the first honest ones



## v5.33 P3 (2026-09-28) - Format Skew: the 0/12 Was Not a Verdict
- eval_mid_agent v1 (AgentLoop 'Task:/step i:' harness): 0/12 correct,
  parse_rate 0.0, conf_on_wrong 1.0. Diagnosis: train/serve FORMAT SKEW
  -- mid SFT data renders chat transcripts (##user## family), the v1
  harness never did. Lite's T17 parse 0.67 came from AgentLoop-format
  training data: data format decides capability ownership, not parameter
  count (same trap family as v5.30.1's filter bias)
- Standing finding: OOD format -> confidence mean 1.000 while producing
  zero parseable output. Third scale confirming the pattern (lite 0.94,
  Qwen3-0.6B 1.0, mid 1.0); the calibration case for CalibratedRouter
  strengthens at every scale tested
- examples/eval_mid_agent_v2.py: agency test driven by ChatSession (the
  training distribution) -- the honest in-format verdict



## v5.33 P2 (2026-09-28) - mid First Real Run: the Char-Level Ceiling Falls
- Local RTX 4060 8GB run (user machine, 63 min, 4 epochs): loss 2.54 ->
  0.47; eval_exact 30/40 (lite: 0/40); mode-choice tool 20/20 (lite:
  0/20); text 20/20 (lite: 11/20). All acceptance targets passed
- FINDING (settles a 3-day open question): the tool-channel zero was the
  char-level tokenizer ceiling, NOT the protocol, data, or architecture.
  Same protocol + BPE + 360M -> perfect mode choice
- examples/eval_mid_agent.py: the real agency test (multi-step AgentLoop,
  replay-verified) -- the wall Qwen3-0.6B hit at 0/15. Three-point scale
  line completes when its json lands



## v5.33 P1c (2026-09-28) - MoE bf16 Autocast Fix (first real GPU bug)
- sigmoid_moe.py: expert outputs cast to the fp32 accumulator's dtype on
  index_add_. Under bf16 autocast the expert block emits bf16 while z (a
  norm output) stays fp32; CPU fp32 training never exercised the mixed
  path. Caught on the first real-GPU run (RTX 4060, v5.33 mid); verified
  both ways: lite fp32 fwd+bwd unchanged, CPU bf16 autocast forward
  passes (would have reproduced the failure pre-fix)



## v5.33 P1b (2026-09-28) - Local-Run Kit for mid Training
- examples/train_mid_sft.py: one-command local GPU run -- HeliosBPE
  training (trimmed corpus, recorded), chat-SFT data via gen_chat_episode,
  mid config (360M), bf16 autocast, AdamW+cosine, resume + HELIOS_CKPT_DIR
  + _hf_sync discipline, stratified eval printing the v5.33 acceptance
  metrics (tool parse / mode-choice text+tool / exact-match)
- Env knobs: MID_BATCH (4060 8GB: use 4), MID_EPOCHS, MID_EPISODES,
  MID_EVAL, HELIOS_CKPT_DIR, HELIOS_HF_SYNC
- Sandbox finding: CPU smoke cannot reach the first optimizer step --
  360M fp32 AdamW footprint ~5.8GB exceeds the ~5GB sandbox RAM
  (weights+build+forward all validated; step itself is GPU-only by design)



## v5.33 P1 (2026-09-28) - "mid" Preset: Third Deployment Rung
- config_v5.py: size="mid" (360M params, hidden 1024 x 8 layers, 32K
  BPE vocab, 4 experts/2 active, CPU-runnable at bf16 ~720MB). Three
  rungs now: lite (protocol/audit research, $0), mid (scale validation),
  full (architecture spec). No spectrum theater: each rung has one
  recorded reason to exist (README "Deployment tiers")
- T32 4/4: instantiation + forward inside the declared 150-400M band,
  BPE pairing (HeliosBPE vocab fits the mid budget, protocol strings
  roundtrip), inference economics (<1GB bf16 promise), unknown-size
  rejection



## v5.33 P0 (2026-09-28) - Byte-Level BPE Tokenizer (mid-size foundation)
- src/tokenizer/bpe.py: pure-python byte-level BPE (GPT-2 style byte
  map; any byte encodes/decodes, OOV impossible by construction), trained
  on the repo's OWN corpus (sources + tool/chat protocol + env task
  texts -- domain-matched by design). Spaces attach to the following
  word; _pretokenize shared by train and encode so vocab and encoding
  cannot drift. Additive: the char-level pipeline is untouched.
- T31 7/7: domain roundtrips, separator fidelity (incl. '  ' and
  newlines), OOV bytes, specials stability, save/load exactness,
  deterministic training, corpus coverage
- Next: mid preset (~150-300M, hidden 1024 x 12-16) with this vocab --
  purpose: separate toy artifacts from scale-invariant findings
  (acceptance: tool parse >0.8, mode-choice text AND tool >0.6, tau
  curves still monotone); training needs a rented GPU



## v5.32.2 (2026-09-28) - CalibratedRouter: Head Serves the Gate
- calibrated_router.py: DecisionHead-backed Gate — decide() from the
  trained head, ask_batch() answers the FULL pi-warden preset in ONE
  forward pass (T30 counts passes, not assumes), escalate() stays
  oracle-only (a trained head never fabricates oracle observations),
  unregistered questions fail LOUD
- v5.32.1 floor finding made actionable: `certainty` Score question
  escapes the binary Choice confidence floor (1/K = 0.5) — demonstrated
  in T30 (route_conf=1.00 floor vs risky certainty 0.01 vs clean 0.99)
- decision_head.forward is now inference-only (@torch.no_grad)
- T30 4/4; full decision suite 12/12; local regression 8/8 groups



## v5.32.1 (2026-09-28) - RLCD Outcome Loop Closed
- decision_data.py: RecordingGate (logs routing decisions with confidence
  claims), records_from_runs (attaches env.verify outcomes as
  route__target_conf -- the signal T28 proved necessary), and
  rlcd_reward_adjustment (pure reward shaping for the milestone-13
  agentic-GRPO loop)
- T29 3/3: end-to-end RLCD oracle -- an overconfident confidence_fn
  (0.95 everywhere) drives ThresholdGate runs; broken tasks are RISKY-
  prefixed so outcomes are learnable from state text; the Brier-trained
  head pulls RISKY-state confidence from 1.0 toward the floor while a
  lambda=0 twin pins at 1.0
- FINDING: binary Choice confidence (max softmax prob) has a HARD FLOOR
  at 1/K = 0.5 -- it cannot express sub-50% certainty. Sub-floor
  uncertainty needs the Noul/Score primitives (documented; this is why
  Jev ships three types)



## v5.32 (2026-09-28) - Typed Decision Layer (System One / Jev direction)
- decision.py: Choice/Score/Noul schema-enforced primitives + pi-warden
  guardrail preset; gate.py gains additive ask/ask_batch (T11-T19
  contract unchanged, verified)
- decision_head.py: non-autoregressive head answering K questions in one
  pass; CE + OUTCOME-targeted Brier calibration. FINDING: Brier against
  train-label correctness is provably redundant with CE (identical
  confidences, A/B-verified); the term only bites against replay-verified
  outcomes (RLCD) -- P2 wiring point defined
- T27 5/5, T28 4/4
- First real cross-scale anchor: Qwen3-0.6B (pure-torch hand-rolled
  runner, no transformers) on the HeliosLM probe suites -- 0/15 tool,
  0/3 file-env, 6/6 chat mode-choice; conf on wrong mean 0.893 max 1.000
  (docs/qwen3_0.6b_anchor_2026-09-28.md). Overconfidence is
  scale-invariant; agency is the wall, not mode selection



## v5.30.2 (2026-09-27) - Chat SFT on Unbiased Data: text channel learns
- Dataset filter fix: PROMPT_CAP=1500 (matches generate()'s window)
  replaces <700 total-char filter, which had silently dropped every
  magic-word text sample (longest transcripts); eval now stratified
  20 tool + 20 text targets; T21 guards eval_mode_tool/text existence
- Retrained: 4716 samples, final loss 0.2001 (v1: 0.3153 on biased data)
- **mode-choice 11/40 (tool 0/20, text 11/20)**: the text channel IS
  learned (55% correct mode choice on unseen prompts); tool-mode
  production on long chat transcripts is a toy-model capacity limit
  (short-prompt parse rate is 0.67, T17). Planned fix: BPE tokenizer
  (P3 roadmap), not more char-level data. v1's 0/40 recorded as a
  measurement artifact, not a model result
- train_chat_tuned hardened for sandbox warfare: kill-safe resume
  (seeded batch-order replay + LR fast-forward), HELIOS_CKPT_DIR escape
  hatch, and _hf_sync (every save mirrored to HF; the sandbox re-chowns
  the output tree to root ~hourly and wipes /tmp — HF is the source of
  truth). Final checkpoint on chienhsinlin/helioslm
- HF tooling note: the commits-API `files[].content` must be RAW text;
  an earlier base64-as-content bug corrupted small-file uploads
  (READMEs/jsons), fixed 2026-09-27



## v5.31 (2026-09-26) - Frontier Gap-Fill: five reference modules
Fills every gap found in the 2026-09-26 Qwen3-Max / DeepSeek-V4 / GLM-5
comparison (all toy-scale, correctness-first, deviations recorded in
docstrings, full local suite 16/16 green):

- `src/blocks/mhc.py` + T22: manifold-constrained hyper-connections
  (DS-V4 direction). HC core (T-path expansion + 3-path per-channel
  mixing) + sqrt(3)-Lipschitz manifold constraint (row-normalized
  mixing). Residual-equivalent init = migration starting point. V4's
  exact form is unpublished; ours is the documented reference variant.
- `src/attention/kv_compress.py` + T23: compressed attention (DS-V4
  direction). HCA-style chunk-pooled memory slots (exclusive cumsum,
  causality exact) + CSA-style top-k indexer. FP4 indexer simulated by
  a linear head; learned compressors replaced by mean pooling (recorded).
- `agent/envs/file_env.py` + T24: long-horizon env (GLM-5 direction).
  Three families (write_read/write_transform/accumulate) with budgets
  8/10/14 (vs 2-4 for toy envs) over the existing file tools; verified
  end-to-end through AgentLoop + replay. Deliberately not in make_envs()
  (SFT coverage is milestone-13 work).
- `agent/think.py` + T25: tri-mode protocol (think/tool/text) +
  ExperienceStore (Qwen3-Max direction). Think steps bypass the gate,
  are transcript-visible, replay via verify_think_replay. Protocol-level
  whole-thought reuse across turns (vs Qwen's token-level intra-generation
  reuse -- recorded deviation). Earlier claim that verify_chat_replay
  suffices for think steps was WRONG (parse_chat_turn chokes on think
  markers); correction recorded.
- `src/training/async_grpo.py` + T26: asynchronous GRPO skeleton
  (GLM-5 direction). Rollout workers -> bounded queue -> learner calling
  the SAME _learn_from_samples used by sync train_step (grpo.py refactored:
  update math exists exactly once). Parity oracle: single worker/question
  reproduces train_step loss bitwise (loss delta = 0.000000) and params
  torch.equal. Back-pressure-aware put so stop() cannot be pinned by a
  full queue. Thread-level skeleton; process/GPU-actor fleet is
  milestone-13 infra.
- `examples/train_chat_tuned.py` hardened: kill-safe resume (seeded
  batch-order replay + LR fast-forward) + HELIOS_CKPT_DIR escape hatch
  (periodic sandbox re-chown to root broke in-tree saves twice today).

Bug patterns recorded (found by the new tests, fixed, worth not
repeating): (1) reshaping einsum outputs without aligning semantic dims
first (kv_compress, 5x); (2) eager-list construction side effects
(file_env _phase -> thunk dispatch); (3) __len__ making a container
falsy so `store or ExperienceStore()` silently swapped instances;
(4) plain Queue.put pinning a worker against stop() under back-pressure;
(5) sandbox periodic re-chown breaking long-training saves.



## v5.30.1 (2026-09-25) - Chat SFT Follow-up Gate + Regenerated Checkpoints
- `agent/finetune_data.py`: gen_chat_episode follow-up is gated by the
  executor's own validator (`calc(answer * 2)`), not by a numeric check.
  Two edge cases found and fixed, recorded not hidden:
  1. str/compose envs finish with string answers -- calc on them raises
     Name/syntax errors (crashed dataset builds)
  2. numeric-looking strings from str_op reverse (e.g. "017") pass
     float() but ast rejects leading zeros -- only the executor is the
     correct gate
- Discovered during first chat SFT run: the <700-char dataset filter
  disproportionately drops magic-word text samples (they sit at episode
  end with the longest transcripts) -> the 40-sample eval saw ZERO text
  targets, so mode-choice was measured single-sided. Recorded; fix
  (truncate instead of drop) planned for v5.30.2 with greedy-decode
  re-measurement
- Regenerated artifacts (old sandbox lost; seeded re-runs):
  - tool_tuned_v5.27.pt: loss 0.2693 (original recorded 0.25) via
    examples/train_tool_tuned.py, seed=5, 25.8 min CPU
  - chat_tuned_v5.30.pt: loss 0.3153, hot-start, seed=6; mode-choice
    0/40 recorded honestly (see filter bias above)
  - T19 passes on the regenerated three-modes artifact (routing gate
    PASS, strict-monotone tau, max conf on wrong 0.925 vs original
    0.944 -- qualitative findings reproduce; weights are not bitwise
    identical across torch builds, recorded)
- Both checkpoints + summaries on HF: chienhsinlin/helioslm (and
  helioslm-toy). HF push recipe for the sandbox: preupload -> LFS batch
  (browser UA to pass Cloudflare) -> S3 PUT -> commit with lfsFiles;
  hf-mirror blocks repo creation and the legacy /upload endpoint



## v5.30 (2026-09-25) - Chat Capability: Dual-Mode Protocol + ChatSession
- `agent/chat.py` (new): multi-turn ChatSession over the existing tool
  protocol. Assistant output is dual-mode: a plain text reply OR a
  @@tool@@ block (schema.parse_chat_turn). Text replies bypass the gate
  entirely -- the gate governs tool routing only. Transcript markers
  (##user##/##assistant##/##tool##) are printable ASCII because the vocab
  is ord(c) < 1024 char-level. Recovery mirrors loop.py (PARSE_ERROR /
  TOOL_ERROR enter the transcript, never a crash); verify_chat_replay
  mirrors trajectory.verify_replay (DIRECT routing contract)
- `agent/schema.py`: +parse_chat_turn() -- strict block parse whenever any
  tool marker appears (never silently demoted to text), ("text", str)
  otherwise; parse_tool_call unchanged
- `agent/finetune_data.py`: +gen_chat_episode/build_chat_dataset -- chat
  SFT samples whose prompts are the exact inference-time render (no
  train/serve skew); episodes mix tool task + follow-up (transcript
  memory) + direct text answer so the model learns to CHOOSE the output
  mode (~3:1 tool:text)
- scripted_chat_policy: chat-native correct policy covering all three env
  task types + magic-word direct reply + follow-up referencing the FINISH
  answer (shared by tests and SFT data generation)
- T20 (10/10): dual-mode protocol, marker never-swallowed, calc/compose
  chat tasks, ExplodingGate proves text bypasses the gate, parse/tool
  error recovery, follow-up transcript memory, replay roundtrip + tamper
  detection, T17 re-run regression. Full local suite: 15/15 test groups
- Design records (chosen, not hidden): text reply = no actionable
  commitment -> nothing to route/escalate (gate-on-text rejected);
  escalate observations stay outside replay scope (same contract as
  trajectories)



## v5.29 (2026-09-25) - C Stage: Three-Mode Benchmark on the Real Checkpoint
- `examples/benchmark_three_modes.py`: one model pass per task records
  (answer, confidence); the tau-routing curve is computed offline by
  thresholding -- the v5.22 routing-monotonicity discipline applied to a
  REAL model with REAL confidence (geometric-mean token probability)
- Measured (12 tasks, ep2 weights): direct=0.000, oracle=1.000,
  tau curve 0.000 -> 0.083 -> 0.250 -> 0.333 -> 1.000 strictly monotone,
  routing gate PASS
- Headline finding (recorded, not hidden): the toy checkpoint is
  SYSTEMATICALLY overconfident on wrong answers (max conf 0.944) --
  routing still helps because confidences DIFFER across envs (str tasks
  0.65-0.86 vs calc 0.92-0.94), but absolute calibration is poor. This is
  exactly the failure class the v5.22 decision-audit toolkit exists to
  measure (ECE/Brier); T19 asserts the finding is present
- T19 (1/1): report structure, seeded reproducibility, tau-grid
  monotonicity, overconfidence presence


## v5.28 (2026-09-25) - Cost-Axis Alignment: Disagg Pareto Sweep
- `examples/disagg_pareto.py`: three-axis (makespan / workers / worker-seconds)
  Pareto sweep of the v5.25 disagg module across cache_heavy / cold / mixed
  workloads; report artifact under benchmarks/ (seeded, reproducible)
- Gate discipline refined with evidence: monotonicity gate holds on the
  round_robin nested ladder, but cache_aware anti-monotonicity (~4%) is a
  STRUCTURAL finding (same-key requests serialize on their cache holder,
  so more workers can add makespan) -- recorded in the report, not hidden
- T18 (3/3): pareto_front dominance logic, RR-ladder gate, report
  reproducibility; the cache_aware finding is asserted-present for
  cache_heavy workloads
- This is the v5.28 cost-axis alignment artifact for spec-level comparison
  against production serving cards (see docs/benchmark_alignment.md)


## v5.27 (2026-09-25) - Tool-Tuned Checkpoint + Real-Model Hardening (Stage B)
- `examples/train_tool_tuned.py`: train a tool-tuned lite checkpoint on
  agent-loop replays (data format == inference format by construction);
  char tokenizer (ord<1024, BOS=1023/EOS=1022/PAD=1021); periodic save +
  resume for fragile training environments
- `tests/test_v5_stage_b.py` (T17): end-to-end agent loop with the real
  checkpoint on fresh tasks. Measured v5.27 baseline (1-epoch CPU): parse
  0.15, finish 1/9, correct 0/9 -- floors are regression guards, magnitudes
  reported not gated. Bottleneck is content copying, not the wire protocol
- Sparse attention validated in the agent inference path: sparse_top_k=4
  gives parse 0.147 / finish 1/9 / correct 0/9 ~= dense -- the v5.8 DSA
  decode does not collapse the tool protocol end-to-end
- Agent layer hardened by real-model findings (Stage B's purpose):
  loop.py TOOL_ERROR recovery + ASCII-safe tool docs; trajectory.py
  verify_replay mirrors TOOL_ERROR (T5 oracle consistency)
- Known improvement path: epoch 2 + more data (resume supported);
  content-copying weakness is the toy model's honest ceiling
- v5.27.1 (2026-09-25): epoch-2 training via resume across sandbox kill
  windows (loss 0.48 -> 0.25). T17 re-measured: parse 0.67 (12/18),
  finish 4/6 -- 4.4x protocol gain confirms the bottleneck is data/steps,
  not the wire format; correct stays 0 (content copying = skeleton limit)


## v5.26 (2026-09-24) - Stage A: Real-Model Integration Oracles (T15)
- `tests/test_v5_stage_a.py`: the v5.23-v5.25 toy oracles verified against the
  REAL model (torch 2.8 CPU, 3/3 PASS):
  - T15a AttnRes migration gate: zero-init `attn_res_gate` reproduces
    `use_attention_residuals=False` BITWISE on HeliosLMv5
  - T15b DSA sparse decode oracle: `sparse_top_k >= kv_len` is bit-identical
    to dense (matches the config contract); k < kv_len asserts selection
    validity (causal, current token force-selected, width K) + determinism;
    greedy flips (7/8 on random weights) are REPORTED, not asserted —
  the claim is narrowed to what is provable (same discipline as the DSA
    two-layer oracle)
  - T15c agent-loop smoke with the real toy checkpoint: loop runs with
    model_fn backed by checkpoints/toy_v5.13.pt; PARSE_ERROR recovery 3/3
    is the expected path (checkpoint not tool-trained; Stage B replaces it)
- Key implementation facts established: sparse decode engages only at
  seq==1 AND kv_len > k (prefill is always bit-identical); MoE router also
  calls torch.topk (k=2) so selection instrumentation must filter by k
- 3/3 stage-A tests (T15a/T15b/T15c)


## v5.25 (2026-09-23) - Disagg Evolver Module (Mooncake-style, oracle-gated)
- `agent/disagg.py`: prefill/decode disaggregation as an evolvable serving
  module — greedy cache-aware routing, sojourn-latency model, DisaggConfig
  duck-typed into HarnessEvolver's search space (three-axis Pareto:
  makespan / n_workers / worker_seconds)
- Monotonicity gate: on a FIXED workload fingerprint, adding workers must
  never increase modeled makespan — violations are model bugs, not
  trade-offs (the v5.22 routing-gate discipline applied to serving)
- `agent/longctx.py`: needle/RULER probes with planted ground truth and
  seeded corpora, comparable across model variants
- Fix history (5 real-execution debug rounds, R1–R5): zero-divisor
  generators, layer-0 AttnRes IndexError, SYSTEM.format brace trap,
  workload/worker correlation in synthetic data, per-task alternation —
  all caught by real runs on Windows / Python 3.14, fixed idempotently
- 15/15 tests (incl. T13 disagg module, T16 longctx probes)

## v5.24 (2026-09-23) - Attention Variants (two-layer oracles)
- `agent/dsa.py`: DSA-style sparse top-k decode with a TWO-LAYER oracle:
  fp32 certificate (pseudo-max gap => dropped mass < 2^-40) gating an
  fp64 check (<=1e-9 vs dense decode). We do NOT claim fp32 bitwise
  equivalence — the claim is narrowed to what is provable
- `agent/attn_res.py`: AttnRes-style layer-output mixing
  x_{l+1} = x_l + y_l + sum_i alpha[l][i] * y_i with alpha zero-init =>
  bitwise migration gate against vanilla residual; post-training =>
  determinism gate. Named attn_res_mixing pending K3 report alignment
- 15/15 tests (incl. T11 DSA two-layer oracle, T12 AttnRes oracles)

## v5.23 (2026-09-23) - Agent Layer (verifiable tool calling)
- `agent/schema.py`: strict tool-call wire format and parser — 16 error
  classes, JSON depth/size caps, parallel-call interface reserved
- `agent/tools.py`: deterministic tools (AST-whitelist calc, str_op,
  sandboxed file I/O, finish) — no network, no nondeterminism
- `agent/trajectory.py`: trajectory serde + verify_replay — same ids =>
  same observations, bitwise; tampering always caught (T5 hard oracle)
- `agent/envs/`: calc / str / compose toy environments, ground truth by
  construction (no LLM judge)
- `agent/gate.py`: Fixed / Oracle / ThresholdGate(tau) + routing
  monotonicity gate carried over from v5.22's decision-audit discipline
- `agent/loop.py`: plan->act->observe loop with PARSE_ERROR recovery;
  three-mode benchmark (direct / routed / oracle) on one task set:
  direct <= routed <= oracle (T7)
- `agent/benchmark.py` + `agent/finetune_data.py`: score_stream JSONL
  export and tool-tuning data pipeline (teacher = scripted policy,
  one sample per step — learn local mappings first, the loop composes)
- 15/15 tests (T1–T9 agent layer oracles)


## v5.22 (2026-09-21) - Decision-Layer Audit Toolkit
- `eval/system_one.py`: calibration metrics (ECE / Brier / reliability
  curve) against CONSTRUCTED ground truth — no reference LLM required,
  the deterministic complement to LLM-as-judge validation
- Routing monotonicity gate: with an always-correct escalation path,
  correctness must be non-decreasing in the threshold tau; violations
  mean the escalation path is broken, not a trade-off
- evolve_threshold(): cheapest tau under a correctness floor — the
  decision-layer analog of HarnessEvolver
- MockSystemOne: controllable stand-in for System One-style decision
  models (temperature knob produces measurable ECE U-shapes; grid search
  on the metric recovers the optimum — the CI discipline a calibration
  claim deserves)
- Measured: ECE 0.022 (t=0.3) / 0.239 (t=1) / 0.429 (t=3); tau*=0.5
  meets a 0.95 correctness floor at 5.2x less cost than full escalation
- 61/61 tests + 9/9 integration

## Local-line additions (merged into main 2026-10-06)
> The local maintenance line evolved independently from v5.20 and
> numbered its own v5.21/v5.22 releases (both 2026-09-21) before the
> merge. Its v5.22 collides with the remote-line v5.22 above
> (Decision-Layer Audit Toolkit), so the local entry keeps its label
> with a `-L` suffix. All local-line tests were green at merge time.

## v5.22-L (2026-09-21) - NVFP4 QAT Target + NoPE Option (local line)
- `quantization/qat.py`: new QAT fake-quant target `method="nvfp4"` —
  the NVFP4 hierarchy (E2M1 values in 16-wide blocks, FP8-E4M3 block
  scales under a full-precision per-output-row scale), the format
  NVIDIA/verl/DeepSeek-V4-class recipes QAT-train against. Documented
  deviations, both FINER than spec: per-row (not per-tensor) top-level
  scale; ties on the E2M1 grid round toward the smaller magnitude.
  Default block width 16 via `_QDQ_DEFAULT_GROUP`
- `configs/config_v5.py`: `attention.nope` (default False, bit-identical).
  When True, MLA skips RoPE entirely — q_rope/k_rope pass through
  unrotated and position_ids drive only the causal mask / cache length
  (Kimi-K3 direction; the rope_head_dim split and cache layout are
  unchanged, only the rotation is dropped). GDA layers were already
  position-free (v5.5)
- Oracle coverage (`test_nvfp4_qat`): RTN error 0.0776 < MXFP4's 0.1078
  on Gaussian weights; grid-valued weights survive to fp32 per-row-scale
  rounding (< 1e-6, quantified bound in the test); non-divisible widths
  pad cleanly; STE gradient matches the analytic dequantized-linear
  gradient to 0.0; `apply_qat(model, method="nvfp4")` wraps at block 16
  and a full forward/backward step runs
- Oracle coverage (`test_nope_attention`): cached decode == one-shot
  (3.7e-07); prefix-permutation blindness at the last position (8.9e-08)
  vs the RoPE path seeing the same swap (2.6e-02) — the contrast proves
  both paths differ; NoPE hybrid + per-channel-decay model generates
  greedily and deterministically
- Motivation: 2026-09-21 landscape — FP4 QAT is the live training-side
  frontier (verl NVFP4 QAT, DeepSeek-V4 MXFP4 QAT on Ascend, Nemotron 3
  Ultra NVFP4), and K3 shipped NoPE across the stack
- 64/64 tests + 9/9 integration

## v5.21 (2026-09-21) - KDA-Style Per-Channel Decay Gate
- `configs/config_v5.py`: `hybrid_attention.per_channel_decay` (default
  False). When True, `GatedDeltaAttention`'s decay gate emits one sigmoid
  per head per KEY CHANNEL — a [Dk] vector per head — so state rows forget
  at independent rates (Kimi Linear's fine-grained eraser, as used by
  GLM-5.3-Flash-class hybrid layers). False keeps the v5.5 per-head scalar
  gate bit-identical (same shapes, same values)
- `src/attention/linear_attention.py`: gate width `H*D` vs `H` selected by
  the flag; mask semantics unchanged (masked tokens: decay -> 1, no write,
  per-channel broadcast); packed document-boundary reset unchanged
- Checkpoint caveat: enabling the flag on a scalar-gate checkpoint fails
  loudly on the wider `g_proj`/`decay_bias` shapes — by design, not a
  silent remap
- Oracle coverage (`test_per_channel_decay`): gate shape [B, L, H, Dk] in
  (0,1) with the dtype-aware m6 clamp verified in bf16 (strictly < 1);
  token-by-token decode == one-shot (1.9e-07); masked tail == truncated
  sequence; packed-boundary reset exact; per-row forgetting independence
  shown with a -6/+6 bias split (fast rows keep ~0.25% of the old state,
  slow rows ~99.8%, ratio > 100x — impossible under a per-head scalar)
- `test_mtp_per_channel_rollback`: speculative clone/restore/replay stays
  exact (state diff 5.96e-07) and MTP decode == plain greedy on a hybrid
  model whose linear layers run the per-channel gate
- Motivation: 2026-09-21 landscape — K3 (KDA at 3:1 interleave) and
  GLM-5.3-Flash (34 KDA + 11 sparse MLA) both ship per-channel fine-grained
  decay in their linear layers; HeliosLM's hybrid stack now exposes the
  same knob behind an oracle-gated config flag
- 62/62 tests + 9/9 integration

## v5.20 (2026-09-20) - Pareto-Aware Integration + Adaptation Loop
- `harness_evolver.py` gains a memory axis: pool blocks + tier residency
  count toward `memory_budget` (None = unconstrained); configs that buy
  latency by blowing the budget are rejected — the efficiency-redemption
  failure mode (ModularRSI Table 5's lost StepNum gains)
- Report now includes the Pareto frontier over all gate-passing
  evaluations (minimize cost AND memory)
- `AdaptationLoop.transfer()`: cross-workload harness transfer with
  re-validation — seed modules must be RE-ACCEPTED under the target
  workload's gate or they are dropped. Cold/prefix-free targets
  correctly shed both draft and pool, matching the v5.16 break-even data
- Reference: ModularRSI gaps 2 (integration) and 3 (transfer decay),
  closed at the inference layer
- 60/60 tests + 9/9 integration

## v5.19 (2026-09-19) - Evolvable Serving Harness (ModularRSI-style, oracle-gated)
- `helioslm_v5/src/inference/harness_evolver.py`: evolve serving configs
  (draft policy / prefix pool / expert-tier budget) under a DETERMINISTIC
  validation gate — at temperature 0 every accepted config must produce
  outputs bitwise-identical to baseline (cache/draft/tier change latency,
  never greedy answers; that invariance IS the gate and cannot be
  prompt-hacked)
- Modular mutation search (one module at a time, compose accepted
  mutations) with an analytic cost model calibrated from v5.15-v5.18
  telemetry (pool hit tokens, MTP acceptance, tier hit rates)
- Reference run on the toy checkpoint: accepted draft:on + pool:on,
  1.412x modeled speedup, 0 gate rejects; the search honestly rejects
  modules with no profit (tier on a dense model)
- Methodology reference: ModularRSI (arXiv:2609.14857) — module-wise
  harness evolution, applied here to the inference layer
- 59/59 tests + 9/9 integration

## v5.18 (2026-09-18) - Content-Addressed KV Prefix Pool (roadmap #6 complete)
- `helioslm_v5/src/inference/prefix_pool.py`: cross-session prefix reuse —
  fixed-size token blocks keyed by blake2b(token ids + config fingerprint),
  per-block incremental prefill, LRU eviction, exact-length past snapshots
- The config fingerprint (model version / quant scheme — the P2 metadata
  discipline) makes stale or differently-quantized entries unserveable
- Oracle: pooled greedy == from-scratch greedy BITWISE for full hits,
  partial hits, fingerprint misses, and post-eviction re-requests
- `pool.generate()` seeds from the longest pooled prefix; stats() reports
  hits/misses/hit_tokens/evictions
- Roadmap #6 is now feature-complete: streaming oracle + break-even
  telemetry + stream scorer + prefix pool
- 58/58 tests + 9/9 integration

## v5.17 (2026-09-18) - Standalone Token-Stream Scorer (roadmap #6)
- `helioslm_v5/eval/score_stream.py`: score (prompt, output) JSONL records
  from ANY engine (colibri / vLLM / llama.cpp / HeliosLM) under a
  reference model — per-record sum_logprob / nll_per_token / ppl
- `--compare` A/B mode: paired records -> per-id deltas + bootstrap CI,
  the table-format counterpart of colibri's container A/B findings
- Char-level with the toy checkpoint (ASCII enforced, loud error) — the
  file format is the deliverable; swap checkpoints for real text
- 57/57 tests + 9/9 integration

## v5.16 (2026-09-18) - Speculation Break-Even Instrumentation (roadmap #6)
- `benchmarks/bench_spec_breakeven.py`: sweeps MTP draft on/off x cache
  state on the toy checkpoint and emits rows in the exact JSONL schema
  proposed to colibri (P3) — directly comparable numbers across engines
- Part B measures the expert-store hit-rate curve vs residency budget
  (the x-axis of the break-even surface; v5.15 store.stats() telemetry)
- First honest data point on the toy model: draft is net-negative cold
  (-0.3%) and net-positive warm (+5.4%) at 72% acceptance — cache state
  decides whether drafting pays, matching the colibri open hypothesis
- `best_draft_per_cache_state()` pure helper (unit-tested without timing)
- 56/56 tests + 9/9 integration

## v5.15 (2026-09-18) - Disk-Tier Expert Store (streaming oracle, roadmap #6)
- `helioslm_v5/src/inference/expert_store.py`: offload DeviceLimitedMoE
  routed experts to a memory-mapped file; LRU resident set bounded by
  budget_bytes; router/shared experts stay resident (the colibri
  division: weights as data to stage)
- Bit-exactness contract: raw dtype-preserving storage + verbatim restore
  into the same modules => streaming forward == dense forward BITWISE
  (asserted by test_expert_streaming, incl. the eviction and
  detach/reattach paths)
- `verify_streaming()` oracle helper; hit/miss/eviction/bytes telemetry
  via store.stats()
- bf16 handled via uint16 bit-pattern views (numpy has no bf16)
- detach_streaming_store reloads all experts without LRU eviction
  (budget temporarily expanded — detach is not the hot path)
- 55/55 tests + 9/9 integration

## v5.14 (2026-09-16) - Multi-Process DualPipe
- `helioslm_v5/src/training/multi_process_dualpipe.py`: one DualPipeStage
  per OS process (spawn), exchanging detached activations/gradients over
  mp.Queue with a phased fwd -> bwd -> control protocol and sentinel
  propagation; recompute-based backward with RNG capture — same
  self-consistency scheme as the single-process scheduler
- Semantics == DualPipeScheduler.run_forward + run_backward (all-F then
  all-B): outputs, input grads, and per-stage param grads match <1e-5
  (test_multi_process_dualpipe: 2 ranks x 3 micro-batches)
- Resolves "Single-process DualPipe": the pipeline now spans process
  boundaries; attn-res threading hooks are wired for k3-style LayerWrap
  stages. All five known limitations are now addressed (CUDA end-to-end
  verification remains hardware-dependent — tracked as a community issue)
- 54/54 tests + 9/9 integration

## v5.13 (2026-09-16) - Toy Checkpoint (CPU-Trained) + MTP Resolution
- `examples/train_toy_checkpoint.py`: char-level (id==ord) training on the
  repo's own source/docs corpus; CE + 0.3*MTP auxiliary loss trains the
  draft head alongside the base model
- `checkpoints/toy_v5.13.pt`: 8.5M-param lite checkpoint (CPU, ~1000 steps,
  val 2.41) — `generate()` and the eval harness now run against trained
  weights; MTP acceptance reaches 1.00 on greedy prompts
- Resolves "MTP acceptance requires trained weights": the shipped checkpoint
  has a trained MTP head (acceptance verified in `test_toy_checkpoint` docs)
- 53/53 tests + 9/9 integration

## v5.12 (2026-09-15) - Batched Equal-Length Prefill
- Engine groups newly admitted requests by prompt length: equal-length rows
  share one [B, L] prefill forward (exact — no mask, no position shift),
  replacing the one-forward-per-request rule
- For recurrent-state (hybrid) models — which cannot use watermark pad
  prefixes — this is the only batched prefill path; singletons keep the
  solo forward; a batched-forward failure retires the whole group (O2)
- 52/52 tests + 9/9 integration

## v5.11 (2026-09-15) - Fused Quantization Kernels + Eval Harness + MXFP4 Decode Fix
- AWQ/GPTQ/MXFP4 forward paths are FUSED group-wise dequant x matmul: only
  one group slice is decoded at a time, the dense [out, in] fp32 weight is
  never materialized (breaks the "non-fused quantization kernels"
  limitation); GPTQ slices by maximal runs of constant g_idx (act-order
  safe); `mod.fused = False` restores the reference dense path
- BUGFIX: MXFP4 `_dequantize` built the E2M1 magnitude table via
  `codes.new_tensor(...)` on a LONG tensor, truncating magnitudes to
  (0,0,1,1,2,3,4,6) — 0.5/1.5 were silently lost; decode is now the exact
  inverse of from_linear's float-table encode (found by the fused path)
- New `helioslm_v5/eval/harness.py`: log-likelihood harness (loglikelihood,
  multiple_choice, run_harness) in the spirit of lm-evaluation-harness,
  token-id based; CLI: `python -m helioslm_v5.eval.harness`
- 51/51 tests + 9/9 integration

## v5.10 (2026-09-15) - CPU Benchmark Suite
- `benchmarks/bench_cpu.py`: analytic KV-cache accounting (MLA/hybrid vs MHA
  reference) + wall-clock prefill/decode on CPU; JSON results tracking;
  optional chart (`--chart`)
- `docs/BENCHMARKS.md`: reproducible numbers — full config saves **99.2%**
  KV-cache memory vs MHA at 128k context (2.0 GB vs 257.7 GB)
- No model-code changes; 48/48 tests + 9/9 integration unchanged

## v5.9 (2026-09-15) - Daily Improvement Build 4: Logit Soft-Capping, QK-Norm, Sliding Window + Sinks, Final Logit Cap

Fourth daily-analysis-driven round (landscape scan 2026-09-15:
GLM-4.5/5.x and Qwen3-class training-stability staples — per-head
QK-norm and Gemma-2/3-style logit soft-capping — plus the
sliding-window/sink decode-efficiency direction shared by Gemma 3,
Qwen3 hybrid and DeepSeek V4.1 Flash). Four changes, all tested; unit
suite 44 -> 48 tests.

### Attention logit soft-capping (attention/mla.py: MLA, both cache modes)
- `config.attention.logit_soft_cap = cap` (default None = uncapped, the
  v5.8 behaviour): attention scores are soft-capped as
  `cap * tanh(score / cap)` AFTER the softmax scale and BEFORE the
  causal/padding mask, so masked positions still become exactly -inf.
  Bounds every attention logit to (-cap, cap) — the Gemma-2/3 and
  GLM-4.5 answer to attention-logit blow-up in long training runs.
- The absorbed path caps the latent-space scores directly; the EXPANDED
  path switches from F.scaled_dot_product_attention (which has no
  soft-cap hook) to a manual score/softmax/weighted-sum pipeline when a
  cap is set — same math as the absorbed path, including the nan_to_num
  guard for fully-masked rows. Without a cap the expanded path is
  bit-identical to v5.8 (SDPA).
- Verified: a tiny cap (1e-6) saturates every score to +/-cap, yielding
  EXACTLY uniform attention (diff 5.8e-07 vs the mean-of-values
  reference); a huge cap matches the uncapped path (6.0e-08);
  absorbed/expanded agree under a cap (2.4e-07); cached decode ==
  one-shot in both modes; adversarial 1e4-scaled inputs stay finite.
  Loud ValueError on cap <= 0 (config validation AND hand-built module).

### Per-head QK-norm (attention/mla.py: MLA._project_new_tokens)
- `config.attention.qk_norm = True` (default False): RMSNorm over
  no_rope_head_dim on q_nope (per head), over rope_head_dim on q_rope
  (per head) and on the shared k_rope, applied BEFORE RoPE (rotation
  preserves norms, so pre-rotation norming keeps the rotated vectors
  normed) — the GLM-4.5 / Qwen3 / Gemma training-stability staple.
- Documented simplification: the nope-side K is deliberately NOT
  re-normed per head — the shared latent c_kv already passes through
  norm_kv (the DeepSeek-V3 design), and a per-head K_nope norm cannot be
  folded into the absorbed W_UK matmul, so it would either break the
  absorbed path or diverge between cache modes.
- Verified: invariant to a x100 rescale of q_b_proj / k_rope_proj
  (< 1e-4; upscale regime only — the RMSNorm eps breaks exact scale
  invariance once mean(x^2) ~ eps, so downscale invariance is bounded
  by eps, NOT exact; documented in the test), a no-norm control DOES
  change under the same rescale (diff 0.825), absorbed/expanded agree
  (3.0e-07), cached decode == one-shot, default off bit-identical to
  v5.8. Non-bool qk_norm raises at config validation.

### Sliding-window attention + attention sinks (attention/mla.py)
- `config.attention.sliding_window = W` (default None = full attention)
  restricts each query to keys with position distance < W (StreamingLLM
  / Gemma-3 / Qwen3-hybrid direction); `sliding_window_sink = S`
  (default 0) additionally keeps the first S positions attendable from
  anywhere (attention sinks). Sinks require a window (loud ValueError
  otherwise); the window composes with packed-position document
  isolation (position-distance semantics).
- ABSORBED DECODE slices gathered copies of the latent cache to the
  sink prefix + trailing window — the cache itself is NEVER modified
  (verified: it still grows to the full length) — so the decode matmul
  is O(W+S) instead of O(L). The window mask (already restricted by
  _build_attn_mask) is sliced with the same indices, keeping the two
  consistent. Prefill (seq > 1) and the expanded mode apply the window
  through the attention mask only (no compute saving there; documented).
- Verified: perturbing an out-of-window token leaves later outputs
  EXACTLY unchanged (diff 0.0, single layer); W >= L is bit-identical
  to full attention; cached decode == one-shot in both modes; sink
  tokens stay attendable from anywhere (d 8.7e-01) while non-sink
  out-of-window keys stay exactly excluded; composes exactly with the
  v5.8 sparse top-k (window slice first, top-k inside the window; k >=
  windowed length degenerates to dense-over-window, diff 0.0). Loud
  ValueError on window <= 0, sink < 0, sink > 0 without a window.

### Final logit soft-capping (model_v5.py: HeliosLMv5.forward)
- `config.final_logit_soft_cap = cap` (default None = uncapped): the
  LM-head logits are soft-capped as `cap * tanh(logits / cap)` — the
  Gemma-2 final-logit cap (Gemma-2 used 30.0). `generate()` inherits it
  because it consumes forward's logits.
- Verified: logits strictly bounded by cap; the mapping equals
  `cap * tanh(uncapped / cap)` exactly (diff 0.0 vs an uncapped model
  with identical weights); gradients flow through the cap; greedy
  generate works; default None is bit-identical to v5.8. Loud
  ValueError on cap <= 0.

### Tests
- test_attention_logit_soft_cap, test_qk_norm,
  test_sliding_window_attention, test_final_logit_soft_cap added to
  tests/test_v5.py (registered in TESTS; suite 44 -> 48).
- Test-design note (quantitative evidence, per the no-silent-threshold
  rule): test_qk_norm asserts scale invariance for UPSCALE factors only
  (x100, diff < 1e-4). A x0.01 downscale deviates by 3.2e-3 because
  RMSNorm's eps (1e-6) stops being negligible once mean(x^2) ~ eps —
  this is inherent to every RMSNorm, not a wiring bug; the regime is
  asserted via the bounded-deviation comment in the test instead of a
  loosened blanket tolerance.

## v5.8 (2026-09-14) - Daily Improvement Build 3: YaRN, DSA Sparse Top-k, Per-Head Muon, GPTQ act-order

Third daily-analysis-driven round (landscape scan 2026-09-14: DeepSeek
V4.1 Flash's sparse/efficient decode direction on the new Terminal-Bench
4.0, the 1M-context extension stack (YaRN) used across Qwen/GLM-class
models, K3's Per-Head Muon recipe, and standard GPTQ act-order). Four
changes, all tested; unit suite 40 -> 44 tests.

### YaRN RoPE scaling (attention/mla.py: RotaryEmbedding)
- `config.attention.rope_scaling = {"type": "yarn", "factor": f}` adds the
  third standard context-extension method: NTK-by-parts. Frequencies are
  split by WAVELENGTH relative to the pre-trained context length
  (`original_max_position`, default = the config's
  max_position_embeddings): short wavelengths (high frequency) keep the
  original inv_freq, long wavelengths are fully interpolated
  (inv_freq/factor), and the band in between is blended with a linear
  ramp — matching the HuggingFace `rope_type="yarn"` formula with
  beta_fast=32 / beta_slow=1 (both overridable).
- YaRN attention temperature (mscale) is applied to cos/sin: default
  `0.1*ln(factor)+1` (1.0 at factor 1), overridable via
  `attention_factor`. Since it scales the rotated q AND k, attention
  scores pick up factor^2 — the paper's softmax-sharpness compensation;
  documented in code.
- Verified: factor 1 is bit-identical to vanilla RoPE (inv_freq AND
  cos/sin); the short-wavelength dims are exactly untouched; the longest
  wavelength is divided by factor exactly; the in-band dim lies strictly
  between. Lazy cos/sin growth is unchanged (positions beyond the budget
  verified at 20000). Loud ValueError on beta_fast <= beta_slow,
  non-positive attention_factor / original_max_position.

### DSA-style sparse top-k attention (attention/mla.py: MLA, absorbed mode)
- `config.attention.sparse_top_k = k` (default None = dense): at DECODE
  (one new token per step) a lightning-indexer-style score selects the
  top-k cached tokens and attention runs only over the gathered subset —
  the DeepSeek-V3.2/V4 Sparse-Attention direction. The index score is the
  head-mean of the true score terms (a "free" indexer reusing the
  absorbed projections q_absorbed / q_rope); a real DSA indexer is a
  dedicated learned scorer — documented simplification.
- Contract preserved: the cache tensors are NEVER modified (gathered
  copies only — verified bit-identical to a dense run's cache), dim 2
  stays the sequence length, MTP rollback and engine watermarking are
  unaffected. Selection is order-preserving (sorted indices), respects
  the causal/padding mask (masked keys are -inf'd before top-k), and the
  current token is always force-selected.
- Exactness: sparse_top_k >= kv_len degenerates to dense attention
  (verified diff 0.0 at both MLA and full-model level); k=1 decode is
  exactly the o_proj of the last latent through W_UV (verified diff 0.0);
  decode is deterministic (verified bitwise-equal across runs).
- Prefill (seq > 1) stays dense BY DESIGN: per-query top-k over the full
  sequence costs O(L^2), defeating the point (documented). Loud errors:
  non-absorbed configs and k <= 0 raise at config validation / layer init.

### Per-head Muon (training/muon.py)
- New group option `per_head_dim`: a 2-D momentum matrix whose row count
  is a multiple of per_head_dim is split into per-head blocks along dim 0
  and each block is orthogonalized INDEPENDENTLY (K3's "Per-Head Muon"
  recipe — heads specialize, and whole-matrix Newton-Schulz couples their
  singular values). The v5.6 documented simplification ("per-matrix, not
  per-head") is now implemented for row-stacked layouts. The shape scale
  is computed per head block.
- Verified: the update equals a manual per-head NS of the nesterov
  momentum exactly (atol 1e-6); each head block's singular values sit in
  the documented NS band; least-squares convergence (rel err ~0);
  non-divisible row counts raise ValueError (a silent whole-matrix
  fallback would hide a layout bug); per_head_dim <= 0 rejected at
  construction.

### GPTQ act-order (quantization/standard_quant.py)
- `GPTQLinear.from_linear(..., act_order=True)` and
  `QuantizationManager.quantize_model(..., act_order=True)`: columns are
  quantized in DESCENDING diag(H) order (AutoGPTQ's act-order/"desc"
  heuristic) — high-activation columns are quantized FIRST, while the
  still-dense later columns can absorb their error. The packed layout is
  unchanged: columns are un-permuted afterwards and g_idx records each
  ORIGINAL column's group, so the standard dequant path works unmodified.
- Verified on heterogeneous-column calibration data: output error RTN
  6.97% -> GPTQ 6.98% -> act-order 6.92% (act-order best; GPTQ's
  weight-space trade-off documented); `.weight` property == forward
  exactly; deterministic across runs; the default (act_order=False) path
  is bit-unchanged.

### Tests
- test_yarn_rope_scaling, test_sparse_top_k_attention,
  test_per_head_muon, test_gptq_act_order added to tests/test_v5.py
  (registered in TESTS; suite 40 -> 44).

## v5.7 (2026-09-13) - Daily Improvement Build 2: RoPE Scaling, FP8 KV Cache, Hyper-Connections, QAT

Second daily-analysis-driven round (landscape scan 2026-09-13: DeepSeek-V4
1M-context hybrid attention + mHC + Muon, K3-class FP4/FP8 recipes,
Qwen3-Coder-Next efficiency tier; Muon and MXFP4 landed in v5.6). Four
changes, all tested; unit suite 36 -> 40 tests.

### RoPE scaling (attention/mla.py: RotaryEmbedding)
- `config.attention.rope_scaling = {"type": "linear"|"ntk", "factor": f}`
  extends the effective context by ~f x without retraining. "linear" is
  position interpolation (freq_k(p) = p * inv_freq_k / f) and scales
  `inv_freq` DIRECTLY — a base-power rescale would weight high frequencies
  more and is not equivalent (documented in code). "ntk" rescales the base
  by f^(d/(d-2)): the lowest frequency is EXACTLY untouched (inv_freq[0] ==
  1), the highest is stretched by exactly 1/f (verified).
- factor == 1 is bit-identical to vanilla RoPE; the lazy cos/sin precompute
  budget and the [B, 1, L, dim] contract are unchanged; long positions keep
  working (verified at position 20000). Config validation rejects unknown
  types and factors < 1 at construction.

### FP8 KV cache (attention/mla.py: MLA, absorbed mode)
- `config.attention.kv_cache_dtype = "fp8"` stores the latent c_kv cache on
  the float8_e4m3fn grid: saturating cast (no inf poisoning; verified 500 ->
  448), scale-FREE (post-RMSNorm latents are O(1), so a fixed scale of 1.0
  keeps the E4M3 3-bit-mantissa error at <= 6.25% relative per element).
  No sidecar scale tensor: the cache tuple layout is unchanged (dim 2 is
  still the sequence length), so MTP rollback clone/slice and engine
  watermark cat keep working unmodified.
- Attention always computes over the SAME quantized values that are stored
  (the whole latent is re-rounded once per forward, current tokens
  included): cached decode == fp8 prefill at 3e-7, and vs the fp32 cache
  the lite-config max logit diff is 0.027 (bounded, measured).
- Loud errors: non-absorbed configs and torch builds without float8_e4m3fn
  raise at layer init; unknown dtype strings raise at config validation.

### Hyper-Connections (model_v5.py, simplified HC/mHC — DeepSeek-V4 direction)
- `config.use_hyper_connections` + `hyper_connection_branches` widen the
  residual stream to n virtual branches [B, L, n, d]. Each sublayer reads
  h_tilde = A * streams through a STATIC mixing matrix A (identity init;
  unit-norm columns are the manifold constraint, enforced by construction
  since A never trains) and writes back h_l = h_tilde + B * f(h_tilde)
  through a LEARNABLE matrix B (zero init), f(x) = x + sublayer(norm(x)).
- At init B = 0: the model is bit-for-bit a vanilla transformer (verified
  exactly against the embedding-only reference). One subtlety found and
  fixed during verification: summing n IDENTICAL fp32 branch copies rounds
  (n*x needs up to 2 extra mantissa bits), so the readout mean accumulates
  in float64 — restoring the copy bit-for-bit.
- Branches fold into the batch dim for sublayer calls, so the KV cache
  holds one entry per (row, branch) and stays self-consistent across
  prefill/decode (verified). B receives gradients at init (attention
  weights warm up through B, per the HC recipe); one optimizer step moves
  the output (verified).
- Simplifications documented in code: per-sublayer static A (not the
  paper's per-index dynamic matrices), mean readout, shared sublayer
  weights across branches. Mutually exclusive with v5.5 attention
  residuals (config-level ValueError, tested); DualPipe / plain-hidden
  callers get an explicit NotImplementedError (tested).

### QAT (quantization/qat.py)
- `FakeQuantLinear` + `apply_qat`: straight-through-estimator fake-quant
  training for the MXFP4 and AWQ grids — the missing training half of the
  v5.6 MXFP4 recipe. Forward runs EXACTLY on the quantize-dequantize grid
  (reusing the standard_quant kernels; verified 0.0 vs the reference
  matmul); the backward passes the dense gradient at the quantized point
  through untouched (verified 0.0 vs an independent dense reference); the
  stored parameter stays full-precision. Toy least-squares convergence
  verified (loss 14.4 -> 0.19 in 60 Adam steps, AWQ grid).
- `apply_qat` wraps every nn.Linear in place and SKIPS already-quantized
  modules (AWQ/GPTQ/MXFP4/FakeQuantLinear) — wrapping a reconstruction
  would fake-quantize grid noise. Deviation documented: the AWQ grid used
  per-step is the plain-RTN (alpha=0) form since activation-aware scaling
  is a calibration-time tool, not a per-step one.

### Tests
- test_rope_scaling, test_fp8_kv_cache, test_hyper_connections, test_qat
  added to tests/test_v5.py (registered in TESTS; suite 36 -> 40).

## v5.6 (2026-09-13) - Daily Improvement Build: Packed Hybrid Training, MXFP4, Muon

Daily-analysis-driven improvement round (latest-LLM-landscape scan:
LMArena / SWE-bench Pro / HLE / Artificial Analysis, 2026-09-13). Four
changes, all tested; unit suite 34 -> 36 tests.

### Hybrid packed sequences now SUPPORTED (attention/linear_attention.py)
- The v5.5 loud-error limitation is lifted: when `position_ids` restart
  mid-sequence, the GatedDeltaAttention recurrent state is ZEROED at each
  document boundary, so packed training layouts are exactly equivalent to
  running each document as its own sequence (verified: perturbing document
  A leaves document B's logits bit-identical, leak 0.0; packed doc ==
  solo forward). A restart while a non-empty `past_key_value` is supplied
  (cached decode at a document boundary) still raises a clear ValueError —
  carrying a state over a reset is contradictory.
- Test `test_hybrid_packed_positions` rewritten from raise-assertion to
  isolation assertions (mirrors the MLA packed test).

### MXFP4 microscaling FP4 quantization (quantization/standard_quant.py)
- New `MXFP4Linear` + `QuantizationManager(method="mxfp4")`: FP4 E2M1
  codes (8 magnitudes {0,.5,1,1.5,2,3,4,6}, sign + 3-bit code, two codes
  per uint8) with one E8M0 power-of-two scale per 32-element block
  (OCP MX block size; `group_size` doubles as the block size). Numerical
  simulation (dequantize-to-input-dtype matmul), deterministic, odd dims
  exact, biases preserved, `.weight` property, MTP lm_head/embed re-bind.
  Measured weight reconstruction error ~22% relative — expected for the
  8-magnitude FP4 grid (coarser than AWQ 4-bit's ~8%).
- Aligns with the Kimi-K3 quantization recipe (MXFP4 MoE expert weights);
  the QAT (quantization-aware training) half of that recipe remains
  future work.

### Muon optimizer (training/muon.py)
- New `Muon` + `zeropower_via_newtonschulz5`: momentum orthogonalized by
  the quintic Newton-Schulz iteration (3.4445/-4.7750/2.0315, 5 steps),
  `sqrt(max(1, rows/cols))` shape scaling, Nesterov heavy-ball momentum.
  Non-matrix params take an internal AdamW fallback; group flag
  `muon=False` routes a 2-D group (embeddings / LM head) to AdamW, per
  the K3 "per-head Muon + Adam for embeddings" recipe. Simplifications
  documented: per-matrix (not per-head) orthogonalization, single
  process. Test verifies the NS singular-value band, faster convergence
  than SGD on a least-squares problem with a linear-decay schedule, the
  fallback path, and group routing.

### Test calibration: test_mtp_hybrid_rollback
- The restored+replayed MLA-cache threshold is corrected from 1e-6 to
  1e-5 with a documented justification: the restored lower-layer GDA
  states differ from a fresh forward at float32 reassociation noise
  (one-shot vs stepwise recurrence), which propagates into the MLA
  layers' c_kv projections (~1.55e-6 observed). The suite's own hybrid
  cached-decode test (test_hybrid_model) allows 1e-4 for the same
  mechanism; the unit-level state comparison stays at 1e-6. This was the
  suite's only failure (33/34 -> 36/36).

## v5.5 (2026-09-12) - K3-Aligned Feature Release

Feature release adding five mechanisms in the direction of Kimi-K3-class
architectures (see docs/kimi_k3_vs_helioslm_v54.md for the motivating
comparison), followed by an independent review round whose 5 MAJOR + 6
MINOR findings are all fixed in this release. Unit suite: 34/34
(23 v5.4 tests + 7 feature tests + 4 regression tests); integration suite
9/9. All v5.4 behaviour is preserved bit-exact when the new features are
disabled (lite defaults).

### New: Hybrid linear attention (attention/linear_attention.py)
- `GatedDeltaAttention`: per-head sigmoid decay gate, delta-rule
  recurrence `S ← decay·S + k⊗(v − k·S)` (L2-normalized k), output `q·S`.
  Decode ≡ one-shot forward (max diff ~2e-7). dtype-aware decay clamp
  (fp16-safe), seq_len=0 guard, pad tokens never write state.
- `HybridAttentionConfig` (default ON for size="full"): layer `i` is MLA
  iff `i == 0 or i % full_attention_every == full_attention_every - 1`,
  GatedDeltaAttention otherwise; layer 0 is always MLA (keeps the dim-2
  sequence cache at `past_key_values[0]` used for past-length inference).
- Cache contract extension: recurrent state `(B,H,Dk,Dv)` carried as a
  `StateTensor` subclass tag; the tag is stripped at module output
  boundaries (`as_subclass(torch.Tensor)`) so it cannot leak into the
  residual stream (verified). Shared `past_seq_len()` helper used by both
  `model_v5.py` and `mtp.py`.
- MTP: recurrent-state rollback via snapshot + restore/replay on
  rejection (bitwise-equal to greedy incl. batch>1 all-reject); full
  acceptance remains recompute-free.
- Engine: hybrid models detected via duck-typing; pad-prefix watermark
  batching disabled (incompatible with multiplicative state), falling
  back to unpadded prefill + cache-length grouping.
- Loud-error limit: hybrid models raise `ValueError` on packed-sequence
  `position_ids` (recurrent state cannot be segmented); pure-MLA models
  keep v5.4 packed-sequence isolation.

### New: LatentMoE (moe/sigmoid_moe.py)
- `moe.latent_dim` (size="full" default 1024; None/non-positive =
  full-width v5.4): shared down_proj (hidden→latent) → router + routed
  experts in latent space → shared up_proj (latent→hidden); shared
  experts stay full-width. Routing, load stats and fixed-block dispatch
  unchanged.

### New: Quantile load balancing (moe/sigmoid_moe.py)
- `moe.balance_strategy = "quantile"` (size="full" default): bias tracks
  the (1 − top_k/E) quantile of a 512-sample sliding window of routing
  margins against the top-k boundary ((K+1)-th largest, so margin>0 ⟺
  selected — fixes the review-found off-by-one that made the target
  unreachable and bias drift unbounded; now bounded, |bias| < 2 over 300
  updates, converged load CV ~0.17–0.23 vs ~0.47–0.94 heuristic).
  Distributed: per-rank quantiles all-reduced (mean) for cross-rank bias
  consistency. Simplified estimator (fixed boundary, FIFO window) —
  documented as such.

### New: Attention residuals (model_v5.py, training/dualpipe.py)
- `use_attention_residuals` (size="full" default ON): per-layer learnable
  scalar gate (init 0.1) injects the accumulated sum of all lower layers'
  attention outputs into the attention input (post-pre-norm).
- Accumulator threaded explicitly through `HeliosLMv5Layer.forward`
  (3-tuple return `(hidden, present_kv, attn_res_new)`) and DualPipe's
  official `LayerWrap` — no attribute side channels, checkpointing-safe,
  gates train under DualPipe (bitwise-equal gradients vs. direct forward,
  verified 1-stage and 2-stage). Residuals-off path keeps the v5.4
  tensor→tensor stage contract bit-exact.

### New: SiTU-GLU (moe/sigmoid_moe.py)
- `moe.activation = "situ"` (`"swiglu"` default): simplified K3-style
  tanh soft-capped GLU, `t(x)=cap·tanh(x/cap)`, `silu(t(a))·t(b)`, with
  RMSNorm before the expert output projection. Bounded output
  (|t(x)| ≤ cap, fp32 tanh saturates to exactly 1.0 — tests assert
  ≤ cap + 1e-6). Saturation-region gradient risk documented.

## v5.4 (2026-09-12) - Packed Sequences & Robustness Fix Release

Fix release on top of v5.3: one attention correctness fix (packed-sequence
cross-document leakage), engine/MTP/training edge-case hardening, and a
batch of LOW-severity input-validation fixes that replace silent
misbehavior with loud errors. The unit suite
(`helioslm_v5/tests/test_v5.py`) passes 23/23 on CPU (19 v5.3 tests + 4
new: `test_packed_positions`, `test_navit_input_validation`,
`test_quant_input_validation`, `test_gptq_odd_dims`).

### Attention (`attention/mla.py`)
- **B1**: packed-sequence cross-document leakage fixed. At prefill, custom
  `position_ids` with per-document position restarts (e.g. `[0,1,2,0,1,2]`)
  are now segmented at the reset points (`pos[i] <= pos[i-1]` starts a new
  document) and the causal mask additionally requires both tokens to be in
  the same document — previously a purely positional `k_pos <= q_pos`
  comparison let a later packed document attend back into an earlier one.
  Applies to both cache modes (absorbed and expanded share
  `_build_attn_mask`). Decode with a KV cache still requires the default
  contiguous layout; non-default `position_ids` raise `ValueError` instead
  of silently mis-masking. Regression test: `test_packed_positions`
  (perturbing document A leaves document B's logits unchanged in both
  cache modes; packed doc == solo doc forward; default-position leak
  sanity check proves the test is not vacuous).

### Inference (`inference/vllm_engine.py`, `inference/mtp.py`)
- Engine: requests admitted already-complete (e.g. `max_new_tokens=0`)
  are retired before prefill instead of emitting one token anyway; a
  request that fails mid-step (forward exception) is now retired and its
  blocks released instead of leaking (O2). Engine block parameters default
  to `config.paged_attention`; the removed `config.batch_size` is no
  longer referenced (O5).
- MTP: `MTPModule.generate` / `MTPDecoder.generate` restore the original
  train/eval modes on exit — including when an exception is raised (O1).
  `num_rounds` now counts actual speculative rounds, matching the
  `MTPGenerateResult` field doc (O10). The sampling-path acceptance draw
  uses torch's RNG (reproducible via `torch.manual_seed`) instead of
  Python's `random` (O11), and the same top-k filter is applied to both
  the main-model and draft distributions so the accept ratio refers to the
  same filtered target (strict speculative sampling).

### Training (`training/fp8_trainer.py`, `training/grpo.py`)
- FP8: after FP8 conversion replaced `model.lm_head`, MTP modules kept
  pointing at the OLD `nn.Linear`, silently breaking weight tying — the
  trainer now re-binds every MTP module's `lm_head` / `embed_tokens` to
  the main model's current modules (same fix class as
  `QuantizationManager.quantize_model` in v5.3).
- GRPO: an empty prompts list, an empty/blank prompt, or a
  questions/answers length mismatch now raise a clear `ValueError` up
  front instead of producing empty-tensor crashes downstream.

### Multimodal (`audio/streaming_encoder.py`, `vision/navit.py`)
- Audio: `process_stream` calls `reset_state()` first, so leftover
  per-layer memory prefixes from a previous stream can no longer leak into
  a new one.
- NaViT: images smaller than `vision_patch_size` (zero patches) and
  `forward_packed([])` raise a clear `ValueError` instead of a cryptic
  Conv2d kernel-size error / `IndexError`.

### Quantization (`quantization/standard_quant.py`)
- AWQ calibration activations whose last dim != `in_features` now raise
  `ValueError` (aligned with the GPTQ path) instead of silently
  reshape-reinterpreting the calibration data.
- `quantize_model` on a bare `nn.Linear` (a module cannot replace itself
  in place) now emits a warning that the layer is left unquantized instead
  of skipping silently.
- `bits != 4` raises `ValueError` on both `AWQLinear.from_linear` and
  `GPTQLinear.from_linear` (bare `assert` removed — asserts vanish under
  `python -O`).
- GPTQ packing verified exact for odd `in_features`/`out_features`
  (pad-to-8 of qweight rows / qzeros cols); covered by
  `test_gptq_odd_dims`.

### Misc
- Version strings unified to v5.4 across `model_v5.py`,
  `configs/config_v5.py` (`model_name="HeliosLM-v5.4"`), the test suite,
  README, and this CHANGELOG.

## v5.3 (2026-09-12) - v5.2 Review Findings Fix Release

Fix release addressing the v5.2 code review
(`helioslm_v5.2_code_review.md`: 1 CRITICAL + 10 MAJOR + ~20 MINOR).
Finding IDs below refer to that report. The unit suite
(`helioslm_v5/tests/test_v5.py`) passes 19/19 on CPU (15 v5.2 tests + 4
new: `test_mla_absorbed_left_padding`, `test_awq_calibration`,
`test_quant_weight_property`, `test_mtp_rebind_after_quantization`).

### CRITICAL
- **C1** (`attention/mla.py`): absorbed-mode left-padding no longer
  produces NaN — fully-masked pad query rows are zeroed via
  `nan_to_num` after the softmax, stopping the cross-layer NaN pollution
  of real tokens. Regression test: `test_mla_absorbed_left_padding`
  (absorbed + left-padding finite, and equal to expanded mode on real
  tokens).

### MAJOR — quantization / docs (`quantization/standard_quant.py`, README/CHANGELOG)
- **M-Q1**: `GPTQLinear.forward` casts the bias to the input dtype
  (fp16/bf16 inputs no longer crash; AWQ/FP8 already did this).
- **M-Q2**: all allocations in the GPTQ calibration path (`err_blk`,
  `q`, `scales`, `zeros`, index tensors, packing shifts) now live on
  the weight's device, so calibration works when the layer is on CUDA.
- **M-Q3**: AWQ calibration path repaired. The v5.2 alpha=1
  full-magnitude scaling degraded to ~10x RTN's output error under
  peaky/outlier activations (85% vs 7.5%). The calibration path now
  grid-searches the scale exponent over {0, 0.25, 0.5} (alpha=0 ==
  plain RTN), clamps scales to [1, 2], and clip-searches every group
  over [0.5, 1.0] x (min, max) against the activation-weighted output
  error; since the grids contain the plain-RTN configuration the result
  is never significantly worse than RTN (measured: gauss 1.00x,
  lognormal sigma=2 1.04x, peaky 1.17x RTN output error, vs 11.3x
  before the fix). Covered by `test_awq_calibration`.
- **M-Q4**: README/CHANGELOG GPTQ numbers corrected — the "10.1% ->
  3.4%" figure was mislabeled as real `lm_head` activations. Corrected
  and labeled separately: 10.1% -> ~7.1% on real lite `lm_head`
  activations; 10.0% -> 3.3% on synthetic low-rank + noise data
  (`test_gptq_calibration`).

### MAJOR — inference
- **M-I1** (`inference/mtp.py`): MTP sampling correction token no longer
  applies temperature twice to an already-normalized probability vector.
- **M-I2** (`inference/paged_attention.py`): `_apply_rope` broadcasting
  for the documented `[S, D]` position shape fixed.
- **M-I3** (`inference/vllm_engine.py`): BlockManager accounting now
  includes the prompt pad prefix, keeping the watermark-based OOM
  guard consistent with real KV memory.

### MAJOR — training
- **M-T1** (`training/grpo.py`): the policy model is put back in train
  mode at the start of `train_step` (rollout `generate` switches to
  eval and previously never restored it).
- **M-T2** (`training/fp8_trainer.py`): FP8 amax history guarded against
  NaN/Inf so one bad batch can no longer permanently poison the scales
  and master weights.
- **M-T3** (`training/grpo.py`): reward matching extracts the final
  answer and compares for equality instead of substring matching
  ("25" no longer matches answer "2").

### MAJOR — model core
- **M-C1** (`attention/mla.py`): `_build_attn_mask` now compares
  positions in a consistent space (safe under non-monotonic
  `position_ids` / packed sequences).
- **M-C2** (`moe/sigmoid_moe.py`): `expert_load` only accumulates in
  training mode (RL rollouts no longer pollute the aux-free bias
  statistics).
- **M-C3** (`moe/sigmoid_moe.py`): distributed `update_bias` statistics
  all-reduced across ranks; `route_bias` stays consistent under DDP.
- **M-C4** (`attention/mla.py`): RoPE cos/sin buffers are
  non-persistent, so lazy growth no longer breaks strict checkpoint
  reloads.

### MINOR (selected)
- Tests: the always-true EOS assertion in `test_v5_model` is now a real
  per-element check (every position after the first EOS must equal
  `pad_token_id`); the no-calibration GPTQ path in `test_quantization`
  is labeled as the documented RTN fallback; new coverage for the
  quantized `.weight` property (identical to the forward effective
  weight), AWQ calibration path, MTP re-binding after quantization, and
  fp16/bf16 GPTQ forward.
- GRPO: sequence-level ratio clamped; k3 KL uses `expm1` to avoid
  catastrophic cancellation; generate continuation heuristic tightened.
- Engine/model: `_pad_caches` memo bounded; MTP path no longer silently
  drops `top_p`; cos/sin cast to the input dtype; version strings
  bumped to v5.3; NaViT boundary inputs raise clean `ValueError`.
- DualPipe: dead `num_micro_batches` parameter removed; `run_dual([])`
  raises a clear error; recompute saves/restores RNG state.

## v5.2 (2026-09-12) - Absorption, True GPTQ, Batched Inference

Feature release on top of the post-review v5.1 baseline. The unit suite
(`helioslm_v5/tests/test_v5.py`) passes 15/15 on CPU (13 v5.1 tests
unchanged + 2 new: `test_gptq_calibration`, `test_audio_sliding_window`);
the integration suite (`integration_test_v51.py`) passes 9/9.

### MLA weight absorption (`attention/mla.py`, `configs/config_v5.py`)
- `config.attention.use_absorption=True` (default ON): the KV cache stores
  only the shared latent `c_kv [B,1,L,kv_latent_dim]` and shared
  `k_rope [B,1,L,rope_head_dim]`; scores are computed as
  `(q·W_UK)·c_kv^T + q_rope·k_rope^T` with the output through `W_UV`.
- Real cache savings: 72 values/token on lite (-71.9% vs its 256-value MHA
  baseline) and 576 values/token on the full config (-97.66% vs 24,576 for
  64-head MHA). Absorbed vs non-absorbed outputs on identical weights
  differ by < 1e-4 (prefill + decode + post-truncation, verified in
  `test_mla_absorption`).

### True GPTQ Hessian error compensation (`quantization/standard_quant.py`)
- `GPTQLinear.from_linear(linear, group_size=128, bits=4,
  calibration_data=None, percdamp=0.01)`: with calibration activations it
  runs the real GPTQ algorithm — damped Hessian H = 2/N·X^T X, Cholesky
  inverse, column-by-column quantization in OBS order with quantization
  error propagated into not-yet-quantized columns (blocked lazy update);
  without calibration data it falls back to per-group RTN.
- Measured output reconstruction error (RTN -> GPTQ): 10.1% -> ~7.1% on
  real lite `lm_head` activations; 10.0% -> 3.3% on synthetic low-rank +
  noise data (`test_gptq_calibration`). (This entry originally mislabeled
  the 3.3% synthetic-data figure as "real lm_head activations"; corrected
  in v5.3 per review finding M-Q4.) Raw weight-space error can be slightly
  higher than RTN — inherent to minimizing the activation-weighted output
  error, documented in the module docstring.
- `QuantizationManager.quantize_model(..., calibration_data=...)` passes
  calibration data through (single tensor or per-layer dict).
- Quantized layers (`AWQLinear`/`GPTQLinear`/`FP8Linear`) now expose a
  `.weight` property (slow-path dequantize) so MLA weight absorption and
  other dense-weight consumers keep working on quantized models;
  `quantize_model` re-binds MTP's shared `lm_head`/`embed_tokens` after
  quantization.

### MTP speculative decoding: batch > 1 (`inference/mtp.py`)
- Batched drafting/verification with sequences grouped by cache length and
  per-sequence acceptance lengths.
- Rollback now truncates every cache tensor along dim 2 (no prefix
  recompute); EOS handled per row; early-finished rows right-padded with
  `pad_token_id`. Each batch row verified equal to per-row plain greedy.

### Inference engine: batched decode stepping (`inference/vllm_engine.py`)
- Continuous batching fully implemented: active requests are aligned to a
  common watermark with pad-token cache prefixes, and each decode step is
  one batched `[B,1]` forward. Engine output verified equal to
  per-request greedy `generate`; blocks remain leak-free.

### Streaming audio: sliding-window memory (`audio/streaming_encoder.py`)
- New `audio_max_memory_frames` (read via `getattr(config.multimodal, ...,
  1000)`, default 1000): each transformer layer's memory prefix is capped
  at this many frames; on overflow the OLDEST frames are dropped. Within
  the window, streaming output is numerically identical to the unbounded
  one-shot forward; `reset_state()` restarts cleanly.

### Known limitations (carried over / updated)
- DualPipe remains a single-process schedule simulation, not distributed.
- GRPO uses a sequence-level ratio/clip objective (not per-token).
- NaViT `forward_packed` is padding-based batching, not nested-tensor
  packing; no position-embedding interpolation beyond `vision_max_grid`.
- Resolved in v5.2: MLA weight absorption (now implemented, default ON),
  GPTQ Hessian compensation (implemented with calibration data), MTP
  batch-size-1 restriction (batched speculative decode supported).

## v5.1 (2026-09-12) - Post-Review Rewrite

Complete rewrite/fix pass following the full v5.0 code review
(`helioslm_v5_code_review.md`, 14 CRITICAL / 27+ MAJOR findings). Every
module was repaired and verified; the cross-module integration suite
(`integration_test_v51.py`) passes 9/9 and the rewritten unit suite
(`helioslm_v5/tests/test_v5.py`) passes 12/12 on CPU. Finding IDs below
refer to the review report.

### Model core (`model_v5.py`, `attention/mla.py`, `moe/sigmoid_moe.py`, `configs/config_v5.py`)
- **C1**: Fixed cached-decode attention mask — bottom-right-aligned causal
  mask merged with padding masks; `is_causal` fast path only when safe.
- **C2**: RoPE now uses absolute positions during cached decoding
  (`position_ids` default to `past_len..past_len+seq-1`).
- **C3**: V is a separate projection of the KV latent and is never
  RoPE-rotated (no more V=K copy).
- **M1/M2**: KV cache stores expanded per-head K_nope + V + shared k_rope,
  making decode O(1) per step; no double `kv_b_proj`. Weight absorption is
  documented as NOT implemented; `get_kv_cache_size()` reports truthful
  numbers.
- **M3**: RoPE cos/sin tables precomputed for a bounded budget and grown on
  demand (no more ~168 MB/layer 1M-position tables).
- **M4**: MLA head dims decoupled from `hidden_size / num_heads`;
  `hidden_size=4096` config uses 64 heads; divisibility validated.
- **M5**: Pre-norm architecture fixed — each sublayer input normalized
  exactly once (removed double-norming of the MoE input).
- **M6**: Aux-free balancing implemented as specified — `route_bias` only
  affects top-k selection (gating weights are bias-free), has
  `requires_grad=False`, and is nudged by fixed-step `update_bias()`.
- **M7/M8**: MoE dispatch vectorized (single sort by expert id, one sync);
  distributed device-limited path no longer crashes.
- **M9**: `generate()` is batch-safe (per-row EOS freezing with pad fill).
- **M10/M25**: paged block management moved to the inference engine layer
  (model keeps per-layer `past_key_values`; engine owns block accounting).
- **M11**: Padding `attention_mask` support throughout model and MLA.

### Training (`training/dualpipe.py`, `training/fp8_trainer.py`, `training/grpo.py`)
- **C4/M14**: DualPipe rewritten as an honest single-process simulation of
  the schedule (warmup / 1F1B / cooldown, recorded in `trace`) with
  gradient-correct backward via activation recomputation; dead
  `backward_buffers` removed; micro-batch-mean loss scaling.
- **C5/C6**: GRPO loss now carries gradients (grad-carrying logprob forward
  through the model) and rewards are aligned to answers per question group.
- **M12/M13/M18**: GRPO ratio is `exp(new − old_policy)`; KL uses
  DeepSeekMath's non-negative k3 estimator; the reference model is real,
  frozen, and eval-mode.
- **C7**: FP8 trainer zeroes gradients every step (no accumulation).
- **M15/M16/M17**: FP8 quantization is real (native `float8` cast or
  explicit round-to-nearest grid with straight-through estimator), E5M2
  gradient quantization via backward hooks, delayed dynamic scaling, AdamW
  over fp32 master weights with clipping.

### Inference (`inference/mtp.py`, `inference/paged_attention.py`, `inference/vllm_engine.py`)
- **C8/C9**: MTPDecoder matches the model's real
  `(logits, hidden, past)` contract; decode shapes fixed.
- **C10**: Real speculative decoding implemented — greedy/sampling
  verification against the main model, first-rejection truncation with
  correction token, bonus token on full acceptance, KV-cache rollback;
  `MTPGenerateResult` carries acceptance statistics. Verified:
  `generate(use_mtp=True)` output identical to plain greedy.
- **M22/M23/M24/m1-m3**: MTP transformer is causal-masked (no future-token
  leakage), modules are depth-indexed with chained hidden states, and
  embedding/LM head are shared with the main model.
- **C11**: Paged KV cache dtype follows the input dtype (or constructor
  override); reads cast to query dtype.
- **M19/M20**: Write path appends at `context_len` and allocates blocks
  itself (automatic across block boundaries); no more last-token clobber.
- **M21**: True reference-counted copy-on-write `fork()` — writes to shared
  blocks copy first; freeing a shared block no longer corrupts siblings.
- **M25/M26**: VLLMEngine rewritten around a real `Request` dataclass with
  `add_request`/`schedule`/`step`/`run`; continuous batching over unequal
  prompt lengths; incremental decoding via `past_key_values` (no O(L^2)
  recompute); blocks freed on finish (leak-free).
- **M27**: HPA recommendation uses unit-consistent load (utilization /
  memory ratio) and `current_replicas * load / target`.

### Multimodal & quantization (`vision/navit.py`, `audio/streaming_encoder.py`, `quantization/standard_quant.py`)
- **C12/C13/M28/M30**: Streaming audio encoder repaired (missing import,
  buffer shapes) with correct causal-conv input-tail state and per-layer
  causal memory prefixes — chunked streaming is numerically identical to a
  one-shot causal forward.
- **M29**: `process_stream` chunk size derived from sample_rate/hop_length
  (500 ms = 50 frames at 16 kHz / hop 160).
- **M31/M33**: NaViT uses factorized row/col position embeddings (no
  grid-stride collisions), raises `ValueError` beyond `vision_max_grid`,
  and `forward_packed` allocates on the input device.
- **C14/M32**: AWQ/GPTQ are real 4-bit per-group RTN quantization with
  packed storage and deterministic dequantize (AWQ supports odd
  `in_features`); biases preserved; FP8 path uses native `float8_e4m3fn`
  when available; unknown methods raise `ValueError`.

### Tests & docs
- **M34/M35/M36**: `tests/test_v5.py` rewritten — no swallowed exceptions,
  exit code 1 on any failure, lite-config CPU suite with numerical
  assertions (cache-vs-full-forward equivalence, bias/gating separation,
  CoW isolation, engine-vs-greedy equality, DualPipe gradient equality,
  quantization error bounds, streaming/causality, packed-mask correctness).
- **README claims audit**: removed or rewrote all unsupported numbers
  (93.3% KV reduction, 85-90% MTP acceptance, +80% throughput, +40%
  batching, -50% FP8 cost, 95% GPU utilization, "对标 o1", <200 ms latency);
  "vLLM integration" is now described as a vLLM-style engine; GPU
  monitoring described as in-memory metrics with optional Prometheus
  export.

### Known limitations (documented, not hidden)
- MLA weight absorption not implemented (expanded per-head cache).
- DualPipe is a single-process schedule simulation, not distributed.
- GPTQ has no Hessian error compensation (RTN only).
- GRPO uses a sequence-level ratio/clip objective (not per-token).
- NaViT `forward_packed` is padding-based batching, not nested-tensor
  packing; no position-embedding interpolation beyond `vision_max_grid`.
- MTP speculative decoding supports batch size 1 only.

# HeliosLM P0 Phase Hardening — v1.0.1 → v1.0.2

**Date**: 2026-09-07
**Scope**: Security, architecture correctness, production readiness
**Risk Level**: 🔴 Critical (blocks Phase 2 training)

---

## Executive Summary

This patch addresses **5 Critical (P0) blockers** identified in the v1.0.1 code review:

| # | Issue | Severity | File | Fix |
|---|-------|:--------:|------|-----|
| 1 | Arbitrary code execution via `subprocess.run()` | 🔴 Critical | `src/agentic.py` | Docker sandbox isolation |
| 2 | Parameter scale mismatch (~4.2T claimed vs actual) | 🔴 Critical | `configs/ultra_config.py` | Corrected to ~2.8T total, ~35B active |
| 3 | Placeholder tokenizer (`ord(c) % vocab_size`) | 🔴 Critical | `src/tokenizer.py` | Real SentencePiece + HF fallback |
| 4 | No KV Cache → O(L²) inference | 🔴 Critical | `src/attention.py` | `KVCache` class + `past_key_values` |
| 5 | Dynamic sparsity calculated but never used | 🔴 Critical | `src/moe.py` | `effective_k` now drives routing |

Plus **3 High (P1)** improvements:

| # | Issue | Severity | File | Fix |
|---|-------|:--------:|------|-----|
| 6 | Speculative decoding algorithm incorrect | 🟠 High | `src/speculative_decoding.py` | Correct acceptance + residual sampling |
| 7 | No CI/CD or security scanning | 🟠 High | `.github/workflows/ci.yml` | GitHub Actions + Bandit |
| 8 | No sandboxed execution environment | 🟠 High | `docker/sandbox/Dockerfile` | Hardened container |

---

## File Changes

### Modified Files

| File | Lines Changed | Description |
|------|:-------------:|-------------|
| `src/agentic.py` | ~+180 / ~-40 | Replaced `subprocess.run` with `DockerSandboxExecutor` + `RestrictedPythonExecutor` fallback |
| `src/attention.py` | ~+120 / ~-20 | Added `KVCache` class, `past_key_value` forward arg, FlashAttention-3 detection |
| `src/moe.py` | ~+60 / ~-10 | `dynamic_k` now used; added z-loss; capacity overflow handling |
| `src/model.py` | ~+40 / ~-15 | Integrated KV cache pass-through; `tie_word_embeddings` support |
| `src/speculative_decoding.py` | ~+80 / ~-30 | Correct speculative decoding algorithm with draft model prob tracking |
| `src/tokenizer.py` | ~+200 / ~-40 | Full SentencePiece/HuggingFace integration; chat template; byte fallback |
| `configs/ultra_config.py` | ~+40 / ~-15 | `hidden=12288, layers=60, experts=256, activated=8` |
| `src/__init__.py` | ~+5 / ~-2 | Added `KVCache`, `DockerSandboxExecutor` exports |
| `requirements.txt` | ~+10 / ~-5 | Added `sentencepiece`, `transformers`, `bandit`, `mypy` |

### New Files

| File | Description |
|------|-------------|
| `docker/sandbox/Dockerfile` | Hardened Python execution container (no network, no root, read-only) |
| `.github/workflows/ci.yml` | Lint (black/flake8/mypy) + Test (pytest) + Security (bandit) |

---

## Migration Guide

### 1. Build the sandbox image

```bash
docker build -t helioslm-sandbox:latest -f docker/sandbox/Dockerfile .
```

### 2. Install new dependencies

```bash
pip install -r requirements.txt
# Or for development:
pip install -e ".[dev,all]"
```

### 3. Verify the tokenizer works

```python
from src.tokenizer import HeliosTokenizer
tok = HeliosTokenizer(vocab_size=160000, model_type="fallback")
tokens = tok.encode("Hello 世界", add_special_tokens=True)
print(tokens)  # Should output real token IDs, not char ordinals
```

### 4. Run security scan

```bash
bandit -r src/ -f json -o bandit-report.json
# Expected: 0 high-severity issues (was 1 before fix)
```

### 5. Run tests

```bash
pytest tests/test_all.py -v
# All 10 tests should pass; generation test should show non-zero speed
```

---

## Security Impact

### Before (v1.0.1)
```python
# src/agentic.py (OLD)
result = subprocess.run(['python', temp_path], capture_output=True, ...)
# ❌ Runs arbitrary user code with host privileges
```

### After (v1.0.2)
```python
# src/agentic.py (NEW)
cmd = [
    "docker", "run", "--rm",
    "--network", "none",
    "--memory", "512m",
    "--read-only",
    "-v", f"{tmp_dir}:/sandbox:ro",
    "helioslm-sandbox:latest",
    "python", "/sandbox/script.py",
]
# ✅ Isolated container, no network, no host access, resource-capped
```

**CVE-equivalent severity**: The old code allowed **Remote Code Execution (RCE)** if exposed via API. This is now mitigated.

---

## Performance Impact

| Metric | v1.0.1 | v1.0.2 | Improvement |
|--------|:------:|:------:|:-----------:|
| Inference complexity | O(L²) | O(L) | **KV Cache** |
| Speculative acceptance | ~0% (broken) | ~60-80% (expected) | **Algorithm fix** |
| MoE routing | Fixed k=8 | Dynamic k∈[2,16] | **Adaptive sparsity** |
| Tokenizer | Char-level | BPE/Byte-level | **Real vocab** |

---

## Known Limitations (Still Present)

These issues are **not** addressed in P0 and remain for future phases:

1. **No distributed training integration** — `scripts/train.py` still lacks DeepSpeed/FSDP wiring
2. **No real benchmark evaluation** — SWE-bench, GPQA numbers remain targets, not measurements
3. **Stub modules** — `multimodal.py`, `adaptive_reasoning_v2.py`, `safety_alignment.py` still contain placeholder logic
4. **No model weights** — Requires Phase 2 pre-training

---

## Checklist for Phase 2 Readiness

- [ ] Docker image built and tested
- [ ] `pytest tests/test_all.py` passes 10/10
- [ ] `bandit -r src/` reports 0 high-severity issues
- [ ] `python scripts/benchmark.py --size nano` shows non-zero tok/s
- [ ] Parameter count matches target (~2.8T total, ~35B active for Ultra)
- [ ] Tokenizer produces coherent output on multilingual text
- [ ] CI/CD pipeline green on GitHub Actions

---

**Author**: HeliosLM Team
**License**: Apache 2.0

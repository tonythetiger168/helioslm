# Toy BPE tokenizer and checkpoint (Issue #5)

The original `examples/train_toy_checkpoint.py` remains the character-token baseline. The new `examples/train_toy_bpe.py` trains a **ByteLevel BPE** tokenizer using the repository source corpus, then trains a fresh Lite model on its token IDs.

## Setup and commands

```bash
pip install torch tokenizers
pytest -q helioslm_v5/tests/test_toy_bpe.py
python examples/train_toy_bpe.py --steps 10 --batch 2 --seq 32  # smoke
python examples/train_toy_bpe.py --steps 400 --batch 8 --seq 64  # candidate
```

Outputs (default): `checkpoints/toy_bpe.pt`, `checkpoints/toy_bpe.tokenizer.json`, `checkpoints/toy_bpe.metrics.json`. The tokenizer JSON **must be shipped alongside the checkpoint**; the original char checkpoint cannot be decoded with the BPE tokenizer.

## Evaluation integrity

- Source files are sorted and seeded-shuffled before a file-level train/validation split; the BPE vocabulary sees **only training files**.
- The held-out file list and SHA-256 of held-out UTF-8 text are saved for audit.
- Report `val_loss_per_token` for debugging but **do not compare it directly with character-level loss**. Different tokenizations change the unit of loss.
- Compare **validation negative log-likelihood per UTF-8 byte** using the **same held-out text, context policy, training budget and evaluation script** for both char and BPE candidates. The original char baseline script samples validation windows from the training corpus, so its published validation loss is **not an equivalent held-out baseline**. Re-run a corrected char baseline before claiming improvement.
- This training script does not yet reproduce the original script's MTP auxiliary training/acceptance metric. MTP parity is a separate gate.

## Release checklist

- [ ] `pytest -q helioslm_v5/tests/test_toy_bpe.py` passes
- [ ] Smoke run succeeds and tokenizer decodes readable words
- [ ] Full BPE training run completes and artifacts are retained
- [ ] Matching character-level baseline is retrained on the identical split
- [ ] Per-byte held-out NLL is no worse than the char baseline
- [ ] Generation samples are reviewed for actual readable words
- [ ] Optional MTP training and acceptance parity validated if replacing the original toy checkpoint

**Status:** implementation committed; no training results or improvement claims until the runs above finish.

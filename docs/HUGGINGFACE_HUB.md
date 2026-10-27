# Hugging Face Hub interchange (Issue #4)

This module adds **optional** Hub interoperability to the existing `helioslm_v5` model. It does not imply that a pretrained public checkpoint exists.

## Offline round-trip

```python
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.hub import save_pretrained, load_pretrained

model = HeliosLMv5(HeliosLMv5Config(size="lite"))
save_pretrained(model, "checkpoints/hub_demo")
restored = load_pretrained("checkpoints/hub_demo")
```

The directory contains `config.json` (nested dataclass config) and `pytorch_model.bin` (tensor-only state dictionary). Loading uses `torch.load(weights_only=True)` and strict key/shape checking. **Only load trusted local files.**

## Optional online integration

```bash
pip install huggingface_hub
```

```python
from helioslm_v5.hub import push_to_hub, from_hub

# Auth via huggingface-cli login or an explicitly provided token.
push_to_hub(model, "YOUR_USERNAME/helioslm-toy", private=True)

# Trust the repository owner and pin an immutable commit revision.
model2 = from_hub("YOUR_USERNAME/helioslm-toy",
                  revision="FULL_COMMIT_SHA",
                  trust_checkpoint=True)
```

**Security and provenance:** PyTorch `.bin` checkpoints are not a universal safe serialization format. Only download from trusted authors, pin revisions, and avoid running downloaded model code. Do not put access tokens in scripts or Git. The API creates/uploads a Hub repository only when `push_to_hub` is explicitly called. No network calls occur during import or offline tests.

## Acceptance checklist

- [ ] Offline `pytest -q helioslm_v5/tests/test_hub.py` passes
- [ ] Nested config and model state round-trip exactly
- [ ] Upload to an authorized test Hub repository succeeds
- [ ] Download at a pinned revision reproduces state and logits
- [ ] Record dependency versions, Hub commit, model card and licensing
- [ ] Verify no secrets or private training data were published

Until online upload/download has been verified, Issue #4 remains **open**.

"""v5.217: LoRA SFT (src/training/lora.py, sft_hf.py train --lora).

Pins:
1. apply_lora wraps only the targeted projections (vision / audio
   excluded), freezes everything else, and step 0 equals the base model;
2. two masked-SFT steps on CPU through sft_hf.train: finite loss, only
   adapter tensors change, base weights are bit-identical;
3. merge oracle: the merged plain-Linear model matches base + adapter
   within tolerance, and no LoRALinear is left;
4. save_adapter writes peft key names and adapter_config.json; the
   weights load back into a fresh adapter-wrapped model;
5. a target list that matches nothing is refused;
6. with transformers installed: a tiny Llama and a tiny Qwen2 model
   train two LoRA steps and merge (skipped otherwise).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from helioslm_v5.src.training.lora import (  # noqa: E402
    LoRALinear, apply_lora, load_adapter_weights, lora_modules, merge_lora,
    save_adapter)


class Out:
    def __init__(self, logits):
        self.logits = logits


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj, self.k_proj = nn.Linear(d, d), nn.Linear(d, d)
        self.v_proj, self.o_proj = nn.Linear(d, d), nn.Linear(d, d, bias=False)
        self.up_proj, self.down_proj = nn.Linear(d, 2 * d), nn.Linear(2 * d, d)

    def forward(self, h):
        a = torch.softmax(self.q_proj(h) @ self.k_proj(h).transpose(1, 2), -1)
        h = h + self.o_proj(a @ self.v_proj(h))
        return h + self.down_proj(torch.relu(self.up_proj(h)))


class Tiny(nn.Module):
    """HF-shaped toy causal LM: model(input_ids=x).logits."""

    def __init__(self, vocab=32, d=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([Block(d), Block(d)])
        self.vision_tower = nn.Sequential()
        self.vision_tower.q_proj = nn.Linear(d, d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids=None, **kw):
        h = self.embed(input_ids)
        for b in self.layers:
            h = b(h)
        return Out(self.lm_head(h))


def _data(n=8, seq=12, vocab=32):
    rng = np.random.default_rng(0)
    ids = rng.integers(0, vocab, size=(n, seq + 1)).astype(np.uint32)
    mask = np.ones_like(ids, dtype=np.uint8)
    mask[:, : seq // 2] = 0
    return ids, mask


def test_apply_targets_and_identity():
    torch.manual_seed(0)
    m = Tiny()
    x = torch.randint(0, 32, (2, 6))
    ref = m(input_ids=x).logits
    info = apply_lora(m, r=4, alpha=8, dropout=0.0)
    assert len(info["modules"]) == 12
    assert not any("vision" in n for n in info["modules"])
    assert all(n.startswith("layers.") for n in info["modules"])
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and all(".lora_" in n for n in trainable)
    assert info["trainable"] == sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert torch.allclose(m(input_ids=x).logits, ref)   # B = 0 at step 0
    with pytest.raises(ValueError, match="no nn.Linear"):
        apply_lora(Tiny(), targets=["nope_proj"])


def test_two_steps_only_adapter_changes_and_merge_oracle(tmp_path):
    import sft_hf
    torch.manual_seed(0)
    m = Tiny()
    apply_lora(m, r=4, alpha=8, dropout=0.0)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    ids, mask = _data()
    res = sft_hf.train(m, ids, mask, epochs=1, micro_batch=2, grad_accum=2, lr=1e-2,
                       warmup=0, autocast=False, eval_every=0)
    assert res["steps"] == 2 and np.isfinite(res["loss"]) and res["stopped"] is None
    for n, p in m.named_parameters():
        changed = not torch.equal(p.detach(), before[n])
        if ".lora_B" in n:
            assert changed, n
        elif ".lora_A" not in n:
            assert not changed, n           # base weights bit-identical
    x = torch.randint(0, 32, (3, 10))
    m.eval()
    with torch.no_grad():
        adapted = m(input_ids=x).logits
    adapter_path = save_adapter(m, tmp_path / "adapter", "toy")
    assert merge_lora(m) == 12 and not lora_modules(m)
    assert all(type(mod) is nn.Linear for n, mod in m.named_modules() if n.endswith("_proj"))
    with torch.no_grad():
        merged = m(input_ids=x).logits
    assert torch.allclose(adapted, merged, atol=1e-5, rtol=1e-4)
    # the adapter reloads into a fresh wrapped copy of the same base
    cfg = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert cfg["peft_type"] == "LORA" and cfg["r"] == 4 and cfg["lora_alpha"] == 8
    assert "q_proj" in cfg["target_modules"]
    torch.manual_seed(0)
    fresh = Tiny()
    fresh.load_state_dict({k.replace(".base.", "."): v for k, v in before.items()
                           if ".lora_" not in k}, strict=True)
    apply_lora(fresh, r=4, alpha=8, dropout=0.0)
    assert load_adapter_weights(fresh, adapter_path) == 24
    fresh.eval()
    with torch.no_grad():
        assert torch.allclose(fresh(input_ids=x).logits, adapted, atol=1e-5, rtol=1e-4)


def test_peft_key_names():
    from helioslm_v5.src.training.lora import lora_state_dict
    m = Tiny()
    apply_lora(m, r=2, alpha=2, targets=["q_proj"])
    keys = sorted(lora_state_dict(m))
    assert keys == ["base_model.model.layers.0.q_proj.lora_A.weight",
                    "base_model.model.layers.0.q_proj.lora_B.weight",
                    "base_model.model.layers.1.q_proj.lora_A.weight",
                    "base_model.model.layers.1.q_proj.lora_B.weight"]
    assert isinstance(m.layers[0].q_proj, LoRALinear)
    assert m.layers[0].q_proj.weight is m.layers[0].q_proj.base.weight


def test_rank_must_be_positive():
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 4), r=0, alpha=1)


@pytest.mark.parametrize("arch", ["llama", "qwen2"])
def test_tiny_hf_models(arch):
    transformers = pytest.importorskip("transformers")
    import sft_hf
    common = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=64)
    if arch == "llama":
        cfg = transformers.LlamaConfig(**common)
        model = transformers.LlamaForCausalLM(cfg)
    else:
        cfg = transformers.Qwen2Config(**common)
        model = transformers.Qwen2ForCausalLM(cfg)
    torch.manual_seed(0)
    info = apply_lora(model, r=4, alpha=8, dropout=0.0)
    assert len(info["modules"]) == 2 * 7
    ids, mask = _data(vocab=64)
    res = sft_hf.train(model, ids, mask, epochs=1, micro_batch=2, grad_accum=2, lr=1e-2,
                       warmup=0, autocast=False, eval_every=0)
    assert res["steps"] == 2 and np.isfinite(res["loss"])
    x = torch.randint(0, 64, (2, 8))
    model.eval()
    with torch.no_grad():
        a = model(input_ids=x).logits
        merge_lora(model)
        b = model(input_ids=x).logits
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)

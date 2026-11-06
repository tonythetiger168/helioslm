"""v1.0 L-Edge pilot: prune a Gemma 4 text model and distil it
(examples/prune_kd_gemma4.py).

Pins, on a tiny random Gemma 4 with the E2B layout (sliding/full pattern,
KV-shared tail, per-layer embeddings, double-wide shared MLPs):

1. choose_layers keeps the last non-shared layer of every attention type
   and the last layer, and the requested counts;
2. build_student fills every student tensor from a source slice; keeping
   every layer and channel reproduces the source logits;
3. the pruned student loads with AutoModelForCausalLM and generates;
4. kd_loss equals the full-logits KL and its gradient;
5. a few kd steps lower the KL to the teacher.

Run from repo root: python3 -m pytest helioslm_v5/tests/test_prune_kd_gemma4.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.gemma4")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import prune_kd_gemma4 as pk  # noqa: E402


def tiny_cfg(hidden=64, layers=10, shared=5, ffn=96):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    types = ["sliding_attention" if (i + 1) % 5 else "full_attention"
             for i in range(layers)]
    return Gemma4TextConfig(
        vocab_size=128, hidden_size=hidden, intermediate_size=ffn,
        num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, global_head_dim=32, num_kv_shared_layers=shared,
        layer_types=types, hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=128, use_double_wide_mlp=True,
        sliding_window=16, max_position_embeddings=256,
        final_logit_softcapping=30.0, pad_token_id=0, bos_token_id=2,
        eos_token_id=1, tie_word_embeddings=True)


def tiny_model(seed=0, **kw):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    torch.manual_seed(seed)
    m = Gemma4ForCausalLM(tiny_cfg(**kw)).eval()
    with torch.no_grad():                    # break the all-ones norms
        for n, p in m.named_parameters():
            if p.ndim == 1:
                p.add_(0.1 * torch.randn_like(p))
    return m


def tiny_data(tmp, n=20000, vocab=128):
    rng = np.random.default_rng(0)
    # a learnable sequence: mostly +1 steps, so KD has signal
    x = (np.cumsum(rng.integers(1, 3, size=n)) % (vocab - 3) + 3).astype(np.uint32)
    (tmp / "main").mkdir(parents=True)
    x.tofile(tmp / "main" / "a.bin")
    man = {"schema": "cpt-data/1", "dtype": "uint32", "sources": {
        "a": {"phase": "main", "path": "main/a.bin", "tokens": n, "weight": 1.0}}}
    (tmp / "manifest.json").write_text(json.dumps(man))
    return tmp


def test_choose_layers():
    cfg = tiny_cfg()
    bi = [0.1 * i for i in range(10)][::-1]          # early layers matter most
    kv, sh = pk.choose_layers(cfg, bi, 3, 2)
    assert len(kv) == 3 and len(sh) == 2
    assert 3 in kv and 4 in kv        # last non-shared sliding / full
    types = cfg.layer_types
    fs = pk.first_shared(cfg)
    for t in set(types[:fs]):
        assert max(i for i in range(fs) if types[i] == t) in kv
    assert sh[-1] == 9 and types[9] == "full_attention"
    with pytest.raises(ValueError):
        pk.choose_layers(cfg, bi, 1, 2)


def test_identity_prune_matches_source(tmp_path):
    src = tiny_model()
    cfg = pk.text_config(src)
    x = torch.randint(3, 128, (1, 24))
    stats = pk.calibrate(src, [x], "cpu")
    fs = pk.first_shared(cfg)
    kv, sh = list(range(fs)), list(range(fs, cfg.num_hidden_layers))
    with pytest.raises(ValueError):    # non-shared layers have 96, not 192
        pk.build_student(src, stats, kv, sh, 64, 192)
    # single-wide source: identity prune is exact
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    c = tiny_cfg()
    c.use_double_wide_mlp = False
    torch.manual_seed(1)
    s2 = Gemma4ForCausalLM(c).eval()
    stats2 = pk.calibrate(s2, [x], "cpu")
    st = pk.build_student(s2, stats2, kv, sh, 64, 96).eval()
    with torch.no_grad():
        a = s2(input_ids=x).logits
        b = st(input_ids=x).logits
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_prune_loads_and_generates(tmp_path):
    from transformers import AutoModelForCausalLM
    src = tiny_model()
    x = torch.randint(3, 128, (2, 32))
    stats = pk.calibrate(src, [x[:1], x[1:]], "cpu")
    kv, sh = pk.choose_layers(pk.text_config(src), stats["bi"], 3, 2)
    st = pk.build_student(src, stats, kv, sh, 48, 64)
    c = pk.param_counts(st)
    assert c["effective"] < pk.param_counts(src)["effective"]
    st.save_pretrained(tmp_path / "s")
    re = AutoModelForCausalLM.from_pretrained(tmp_path / "s").eval()
    assert re.config.num_hidden_layers == 5 and re.config.num_kv_shared_layers == 2
    assert re.config.hidden_size == 48 and re.config.layer_types[-1] == "full_attention"
    out = re.generate(x[:1, :8], max_new_tokens=4, do_sample=False)
    assert out.shape[1] == 12
    with torch.no_grad():
        assert torch.isfinite(re(input_ids=x).logits).all()


def test_kd_loss_matches_full():
    torch.manual_seed(0)
    hs = torch.randn(10, 8, requires_grad=True)
    ht = torch.randn(10, 6)
    ws = torch.randn(32, 8, requires_grad=True)
    wt = torch.randn(32, 6)
    got = pk.kd_loss(hs, ht, ws, wt, 30.0, 30.0, chunk=3)
    ls = pk.softcap(hs @ ws.t(), 30.0)
    lt = pk.softcap(ht @ wt.t(), 30.0)
    ref = torch.nn.functional.kl_div(ls.log_softmax(-1), lt.log_softmax(-1),
                                     log_target=True, reduction="batchmean")
    assert torch.allclose(got, ref, atol=1e-5)
    g1 = torch.autograd.grad(got, [hs, ws])
    g2 = torch.autograd.grad(ref, [hs, ws])
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-5)


def test_prune_and_kd_cli(tmp_path):
    data = tiny_data(tmp_path / "d")
    teacher = tiny_model(seed=3)
    teacher.save_pretrained(tmp_path / "t")
    src = tiny_model(seed=4)
    src.save_pretrained(tmp_path / "src")
    tok = _tok(tmp_path)
    for p in ("t", "src"):
        tok.save_pretrained(tmp_path / p)
    r = pk.main(["prune", "--model", str(tmp_path / "src"), "--data", str(data),
                 "--out", str(tmp_path / "p"), "--layers-kv", "3",
                 "--layers-shared", "2", "--hidden", "48", "--ffn", "64",
                 "--calib", "4", "--seq", "32"])
    assert r["params"]["effective"] < r["source_params"]["effective"]
    res = pk.main(["kd", "--student", str(tmp_path / "p" / "final"),
                   "--teacher", str(tmp_path / "t"), "--data", str(data),
                   "--out", str(tmp_path / "k"), "--tokens", str(40 * 4 * 32),
                   "--seq", "32", "--micro-batch", "4", "--batch-tokens", "128",
                   "--lr", "3e-3", "--warmup", "2", "--kd-chunk", "40",
                   "--freeze-ple"])
    rows = [json.loads(l) for l in (tmp_path / "k" / "log.jsonl").read_text().splitlines()]
    assert res["steps"] == 40 and rows[-1]["phase"] == "decay"
    first = np.mean([r["kl"] for r in rows[:5]])
    last = np.mean([r["kl"] for r in rows[-5:]])
    assert last < first, (first, last)
    assert (tmp_path / "k" / "final" / "config.json").exists()


def _tok(tmp):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {f"t{i}": i for i in range(128)}
    t = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=t, unk_token="t0",
                                   pad_token="t0", eos_token="t1", bos_token="t2")

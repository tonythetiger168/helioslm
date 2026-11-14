"""Own-base JAX pretraining path (v5.268): jax_pretrain.py,
examples/jax_pretrain_tpu.py, examples/own_base_shards.py.

Numpy-only parts (mixture loader, checkpoints, LR schedule, sharding,
the committed mixture) always run; the model / training tests need jax
and skip without it (CI installs no jax; run them with
``pip install jax`` locally or on the TPU VM).

1. MixtureLoader: windows come from the right shards, mixture weights
   hold, a shard advances after its size in tokens, a restored state
   draws exactly what an uninterrupted loader would.
2. Checkpoint .npz round trip keeps nested trees and the loader state.
3. lr_at: warmup ramps to the peak, cosine ends at the floor.
4. own_base_shards: val tail cut before sharding, uint16 when ids fit,
   weights follow tokens, shard SHA-256 recorded.
5. The committed own_base_0p3b_v1 mixture validates (allow-listed
   licences only, phases sum to 1).
6. [jax] 0.3b preset is ~304M parameters; chunked loss == loss from full
   logits; causality (future tokens do not change earlier logits).
7. [jax] export_hf state dict run through an independent numpy Llama
   forward (HF layout, rotate_half RoPE, GQA) reproduces our logits.
8. [jax] training on the synthetic pattern lowers the loss, and a run
   cut by --max-hours and resumed ends bit-identical to a straight run.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.training import jax_pretrain as jp  # noqa: E402


def _write_shards(tmp, spec):
    """spec {name: [values per shard]} -> mixture dict with local files."""
    srcs = []
    for name, (weight, shards) in spec.items():
        files = []
        for i, val in enumerate(shards):
            f = tmp / f"{name}_{i}.bin"
            np.full(400, val, np.uint16).tofile(f)
            files.append(str(f))
        srcs.append({"name": name, "weight": weight, "files": files})
    return {"dtype": "uint16", "sources": srcs}


def test_mixture_loader_weights_advance_and_resume(tmp_path):
    mix = _write_shards(tmp_path, {"a": (0.75, [1, 2]), "b": (0.25, [7])})
    ld = jp.MixtureLoader(mix, seq=16, seed=3)
    seen = []
    for _ in range(40):
        x, y = ld.next_batch(8)
        assert x.shape == (8, 16) and y.shape == (8, 16)
        assert x.dtype == np.int32
        seen.append(x[:, 0])
    vals = np.concatenate(seen)
    share_b = np.mean(vals == 7)
    assert 0.15 < share_b < 0.35
    # source a has 2 shards of 400 tokens -> both are visited
    assert {1, 2} <= set(vals.tolist())

    ref = jp.MixtureLoader(mix, seq=16, seed=3)
    for _ in range(25):
        ref.next_batch(8)
    st = json.loads(json.dumps(ref.state()))
    tail_ref = [ref.next_batch(8)[0] for _ in range(10)]
    res = jp.MixtureLoader(mix, seq=16, seed=3)
    res.load_state(st)
    tail_res = [res.next_batch(8)[0] for _ in range(10)]
    for p, q in zip(tail_ref, tail_res):
        np.testing.assert_array_equal(p, q)


def test_mixture_loader_rejects_tiny_shard_and_empty_mix(tmp_path):
    f = tmp_path / "s.bin"
    np.zeros(10, np.uint16).tofile(f)
    with pytest.raises(ValueError):
        jp.MixtureLoader({"sources": [{"name": "x", "weight": 1,
                                       "files": [str(f)]}]},
                         seq=16).next_batch(2)
    with pytest.raises(ValueError):
        jp.MixtureLoader({"sources": [{"name": "x", "weight": 0,
                                       "files": [str(f)]}]}, seq=4)


def test_checkpoint_roundtrip(tmp_path):
    params = {"embed": np.arange(6, dtype=np.float32).reshape(2, 3),
              "layers": {"wq": np.ones((2, 3, 3), np.float32)}}
    opt = {"m": params, "v": params, "step": np.asarray(5, np.int32)}
    cfg = jp.PRESETS["tiny"]
    jp.save_checkpoint(tmp_path, params, opt, 42, cfg,
                       {"main": {"shard": [1], "drawn": [3], "batches": 9}})
    p, o, meta = jp.load_checkpoint(tmp_path)
    np.testing.assert_array_equal(p["layers"]["wq"], params["layers"]["wq"])
    assert int(o["step"]) == 5 and meta["step"] == 42
    assert meta["loader"]["main"]["batches"] == 9
    assert meta["config"]["hidden"] == cfg.hidden
    assert jp.load_checkpoint(tmp_path / "missing") is None


def test_lr_schedule():
    assert jp.lr_at(0, 1.0, 10, 100) == pytest.approx(0.1)
    assert jp.lr_at(9, 1.0, 10, 100) == pytest.approx(1.0)
    assert jp.lr_at(100, 1.0, 10, 100, 0.1) == pytest.approx(0.1)
    mid = jp.lr_at(55, 1.0, 10, 100, 0.1)
    assert 0.1 < mid < 1.0


def test_own_base_shards(tmp_path):
    sys.path.insert(0, str(_REPO / "examples"))
    import own_base_shards as obs
    data = tmp_path / "data"
    (data / "main").mkdir(parents=True)
    (data / "decay").mkdir()
    np.arange(1000, dtype=np.uint32).tofile(data / "main" / "zh.bin")
    np.arange(500, dtype=np.uint32).tofile(data / "main" / "en.bin")
    np.arange(300, dtype=np.uint32).tofile(data / "decay" / "top.bin")
    (data / "manifest.json").write_text(json.dumps(
        {"mixture": "t", "tokenizer": "tok.json", "eos": 1, "bos": None}))
    out = tmp_path / "out"
    obs.build(data, out, shard_tokens=300, val_tokens=100,
              hf_repo="u/tokens")
    main = json.loads((out / "mix_main.json").read_text())
    assert main["dtype"] == "uint16" and main["hf_repo"] == "u/tokens"
    by = {s["name"]: s for s in main["sources"]}
    assert by["zh"]["tokens"] == 900 and len(by["zh"]["files"]) == 3
    assert by["en"]["tokens"] == 400
    assert by["zh"]["weight"] == pytest.approx(900 / 1300, abs=1e-5)
    val = np.fromfile(out / "val.bin", np.uint16)
    assert len(val) == 200
    np.testing.assert_array_equal(val[:100], np.arange(400, 500))  # en
    np.testing.assert_array_equal(val[100:], np.arange(900, 1000))  # zh
    first = np.fromfile(out / "main" / "zh" / "000.bin", np.uint16)
    np.testing.assert_array_equal(first, np.arange(300))
    decay = json.loads((out / "mix_decay.json").read_text())
    assert decay["sources"][0]["tokens"] == 300  # no val cut from decay
    idx = json.loads((out / "shards.json").read_text())
    assert len(idx["shards"]["main/zh"][0]["sha256"]) == 64
    # the shards feed MixtureLoader directly
    ld = jp.MixtureLoader({**main, "hf_repo": None,
                           "sources": [{**s, "files": [str(out / f)
                                                        for f in s["files"]]}
                                       for s in main["sources"]]}, seq=32)
    x, _ = ld.next_batch(4)
    assert x.shape == (4, 32)


def test_committed_mixture_validates():
    from helioslm_v5.src.training.data_mixture import load_mixture
    mix = load_mixture(_REPO / "data" / "mixtures" / "own_base_0p3b_v1.json")
    assert all(s["license"] != "review" for s in mix["sources"])
    zh_tw = sum(s["weight"] for s in mix["sources"]
                if s["phase"] == "main" and "tw" in (s.get("zh_variants")
                                                     or []))
    assert zh_tw >= 0.35


# ------------------------------------------------------------- jax parts

def _jax():
    return pytest.importorskip("jax")


def test_preset_size():
    cfg = jp.PRESETS["0.3b"]
    assert 290e6 < cfg.n_params() < 320e6
    assert cfg.hidden % cfg.heads == 0 and cfg.heads % cfg.kv_heads == 0


def test_chunked_loss_matches_full_logits_and_causal():
    jax = _jax()
    import jax.numpy as jnp
    cfg = jp.PRESETS["tiny"]
    params = jp.init_params(cfg, 0)
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.integers(0, cfg.vocab_size, (2, 64)), jnp.int32)
    y = jnp.asarray(rng.integers(0, cfg.vocab_size, (2, 64)), jnp.int32)
    logits = jp.forward(params, x, cfg)
    full = -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits),
                                         y[..., None], -1))
    chunked = jp.loss_fn(params, x, y, cfg, jnp.float32, ce_chunk=16)
    assert float(chunked) == pytest.approx(float(full), rel=1e-5)
    x2 = x.at[:, 40:].set(0)
    l2 = jp.forward(params, x2, cfg)
    np.testing.assert_allclose(np.asarray(logits[:, :40]),
                               np.asarray(l2[:, :40]), rtol=1e-5, atol=1e-5)


def _np_llama(sd, hf, ids):
    """Independent numpy Llama forward over an HF-layout state dict."""
    H, KV = hf["num_attention_heads"], \
        hf["num_key_value_heads"]
    hd, eps = hf["head_dim"], hf["rms_norm_eps"]
    T = len(ids)

    def norm(x, w):
        return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w

    inv = 1.0 / hf["rope_theta"] ** (np.arange(0, hd, 2) / hd)
    ang = np.outer(np.arange(T), inv)
    cos, sin = np.cos(np.concatenate([ang, ang], -1)), \
        np.sin(np.concatenate([ang, ang], -1))

    def rope(x):  # [T, h, hd]
        rot = np.concatenate([-x[..., hd // 2:], x[..., :hd // 2]], -1)
        return x * cos[:, None] + rot * sin[:, None]

    x = sd["model.embed_tokens.weight"][ids].astype(np.float64)
    for i in range(hf["num_hidden_layers"]):
        g = lambda n: sd[f"model.layers.{i}.{n}"].astype(np.float64)  # noqa
        h = norm(x, g("input_layernorm.weight"))
        q = rope((h @ g("self_attn.q_proj.weight").T).reshape(T, H, hd))
        k = rope((h @ g("self_attn.k_proj.weight").T).reshape(T, KV, hd))
        v = (h @ g("self_attn.v_proj.weight").T).reshape(T, KV, hd)
        k, v = np.repeat(k, H // KV, 1), np.repeat(v, H // KV, 1)
        s = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(hd)
        s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        a = np.einsum("hqk,khd->qhd", p, v).reshape(T, H * hd)
        x = x + a @ g("self_attn.o_proj.weight").T
        h = norm(x, g("post_attention_layernorm.weight"))
        gate = h @ g("mlp.gate_proj.weight").T
        x = x + ((gate / (1 + np.exp(-gate))) * (h @ g("mlp.up_proj.weight").T)
                 ) @ g("mlp.down_proj.weight").T
    x = norm(x, sd["model.norm.weight"])
    return x @ sd["model.embed_tokens.weight"].T


def test_export_hf_matches_reference_llama(tmp_path):
    _jax()
    import jax.numpy as jnp
    cfg = jp.PRESETS["tiny"]
    params = jp.init_params(cfg, 1)
    ids = np.random.default_rng(1).integers(0, cfg.vocab_size, 24)
    ours = np.asarray(jp.forward(params, jnp.asarray(ids[None], jnp.int32),
                                 cfg))[0]
    sd, hf = jp.export_hf(params, cfg, tmp_path / "hf", dtype="float32")
    assert hf["architectures"] == ["LlamaForCausalLM"]
    assert hf["tie_word_embeddings"] is True
    assert (tmp_path / "hf" / "config.json").exists()
    ref = _np_llama({k: np.asarray(v) for k, v in sd.items()}, hf, ids)
    np.testing.assert_allclose(ours, ref, rtol=2e-4, atol=2e-4)


def test_train_learns_and_resume_is_exact(tmp_path):
    _jax()
    sys.path.insert(0, str(_REPO / "examples"))
    import jax_pretrain_tpu as cli
    common = ["--preset", "tiny", "--synthetic", "--batch", "4",
              "--seq", "64", "--ce-chunk", "32", "--steps", "40",
              "--log-every", "10", "--save-every", "1000", "--lr", "3e-3"]
    cli.main(common + ["--out", str(tmp_path / "a")])
    log = [json.loads(s) for s in
           (tmp_path / "a" / "log.jsonl").read_text().splitlines()]
    assert log[-1]["loss"] < log[0]["loss"] - 0.3
    # cut after one step, three times, then finish
    for _ in range(3):
        cli.main(common + ["--out", str(tmp_path / "b"),
                           "--max-hours", "1e-9"])
    cli.main(common + ["--out", str(tmp_path / "b")])
    pa = jp.load_checkpoint(tmp_path / "a")[0]
    pb = jp.load_checkpoint(tmp_path / "b")[0]
    np.testing.assert_array_equal(pa["embed"], pb["embed"])
    np.testing.assert_array_equal(pa["layers"]["w_down"],
                                  pb["layers"]["w_down"])


def test_train_preserves_checkpoint_on_nonfinite_metrics(tmp_path):
    _jax()
    import jax.numpy as jnp

    sys.path.insert(0, str(_REPO / "examples"))
    import jax_pretrain_tpu as cli

    common = ["--preset", "tiny", "--synthetic", "--batch", "1", "--seq", "8",
              "--precision", "fp32", "--attn", "xla", "--steps", "3",
              "--log-every", "10", "--save-every", "1"]
    adamw_update = cli.adamw_update

    def nan_loss(params, *args, **kwargs):
        return jnp.sum(params["embed"]) * 0 + jnp.float32(jnp.nan)

    def inf_grad_norm(*args, **kwargs):
        params, opt, _ = adamw_update(*args, **kwargs)
        return params, opt, jnp.float32(jnp.inf)

    for symbol, faulty, expected in (("loss_fn", nan_loss, "loss=nan"),
                                     ("adamw_update", inf_grad_norm, "grad_norm=inf")):
        out = tmp_path / symbol
        args = common + ["--out", str(out)]
        assert cli.main(args + ["--max-hours", "1e-9"]) == 1
        before = {name: (out / name).read_bytes() for name in ("ckpt.npz", "state.json")}
        with patch.object(cli, symbol, faulty):
            with pytest.raises(SystemExit, match=f"step 2: .*{expected}"):
                cli.main(args)
        for name, contents in before.items():
            assert (out / name).read_bytes() == contents
        assert jp.load_checkpoint(out)[2]["step"] == 1
        assert cli.main(args) == 3


def test_shards_upload_uses_available_client_method(tmp_path):
    sys.path.insert(0, str(_REPO / "examples"))
    import own_base_shards as obs
    calls = []

    class New:  # huggingface_hub 2.x: no upload_large_folder
        def upload_folder(self, **kw):
            calls.append(("folder", kw["repo_id"], kw["folder_path"]))

    class Old(New):
        def upload_large_folder(self, **kw):
            calls.append(("large", kw["repo_id"], kw["folder_path"]))

    obs.upload(New(), "u/t", str(tmp_path))
    obs.upload(Old(), "u/t", str(tmp_path))
    assert [c[0] for c in calls] == ["folder", "large"]

"""jax_pretrain_tpu.py — pretrain the own small base on TPU (v5.266).

Data-parallel over every local device (Kaggle TPU v5e-8: 8 chips);
parameters and AdamW states are replicated (the 0.3B preset needs about
5 GB per chip of the 16 GB HBM), the batch is split across chips.

  # CPU smoke (seconds):
  python examples/jax_pretrain_tpu.py --preset tiny --synthetic \
      --steps 20 --batch 4 --seq 64 --out runs/jax_smoke

  # Kaggle TPU v5e-8 (see examples/notebooks/kaggle_tpu_own_base.ipynb):
  python examples/jax_pretrain_tpu.py --preset 0.3b --mix mix.json \
      --tokens 6e9 --batch 128 --seq 2048 --grad-accum 2 \
      --max-hours 8.3 --out /kaggle/working/run \
      --hf-ckpt-repo <user>/helioslm-own-0p3b-ckpt

Every run resumes from ``--out`` (or, when that is empty and
``--hf-ckpt-repo`` is set, from the checkpoint in that private model
repo), so a 9-hour Kaggle session limit is handled by re-running the
same command next session. ``--max-hours`` stops cleanly, saves and
uploads before the session is killed.

Log: ``--out``/log.jsonl (step, loss, lr, grad_norm, tokens, tok/s,
MFU against --peak-tflops per chip; v5e bf16 = 197).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.src.training import jax_pretrain as _jp  # noqa: E402
from helioslm_v5.src.training.jax_pretrain import (  # noqa: E402
    PRESETS, ModelConfig, MixtureLoader, adamw_init, adamw_update,
    export_hf, init_params, load_checkpoint, loss_fn, lr_at,
    save_checkpoint)


def _args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", default="0.3b", choices=sorted(PRESETS))
    ap.add_argument("--vocab", type=int, default=0,
                    help="override the preset vocab size (must match the "
                         "tokenizer that wrote the shards)")
    ap.add_argument("--mix", help="mixture JSON (MixtureLoader)")
    ap.add_argument("--mix-decay",
                    help="mixture for the last --decay-frac of the steps "
                         "(the high-quality anneal data)")
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--synthetic", action="store_true",
                    help="random tokens with a learnable pattern (smoke)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--tokens", type=float, default=0,
                    help="total training tokens (sets --steps)")
    ap.add_argument("--batch", type=int, default=128,
                    help="global micro-batch (sequences) per step")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=0,
                    help="warmup steps (default 1%% of steps, >= 100)")
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--ce-chunk", type=int, default=512)
    ap.add_argument("--attn", default="auto", choices=["auto", "flash", "xla"],
                    help="auto: the Pallas flash kernel on TPU when a probe "
                         "call works, else plain XLA attention")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-windows", type=int, default=64)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--max-hours", type=float, default=0,
                    help="stop, save and upload after this wall time")
    ap.add_argument("--peak-tflops", type=float, default=197.0)
    ap.add_argument("--hf-ckpt-repo",
                    help="private HF model repo for checkpoints (resume + "
                         "upload); needs HF_TOKEN in the environment")
    ap.add_argument("--keep-shards", action="store_true",
                    help="keep downloaded token shards on disk")
    ap.add_argument("--export", help="after training, write a Llama-format "
                                     "HF checkpoint into this directory")
    ap.add_argument("--tokenizer", help="tokenizer.json copied into --export")
    return ap.parse_args(argv)


class _Synthetic:
    """Tokens where t[i+1] = (t[i] * 3 + 1) % V half of the time: a
    pattern a working model learns within a few dozen steps."""

    def __init__(self, vocab, seq, seed):
        self.v, self.seq, self.seed, self.batches = vocab, seq, seed, 0

    def next_batch(self, batch):
        rng = np.random.default_rng([self.seed, self.batches])
        self.batches += 1
        x = np.empty((batch, self.seq + 1), np.int64)
        x[:, 0] = rng.integers(0, self.v, batch)
        noise = rng.integers(0, self.v, (batch, self.seq))
        keep = rng.random((batch, self.seq)) < 0.5
        for i in range(self.seq):
            det = (x[:, i] * 3 + 1) % self.v
            x[:, i + 1] = np.where(keep[:, i], det, noise[:, i])
        x = x.astype(np.int32)
        return x[:, :-1], x[:, 1:]

    def val_windows(self, n):
        return None

    def state(self):
        return {"batches": self.batches}

    def load_state(self, st):
        self.batches = int(st["batches"])


def _pick_attn(choice, platform, cfg):
    if choice != "auto":
        return choice
    if platform != "tpu":
        return "xla"
    import jax
    import jax.numpy as jnp
    try:
        q = jnp.zeros((1, 512, cfg.heads, cfg.head_dim), jnp.bfloat16)
        _jp.ATTN_IMPL = "flash"
        jax.block_until_ready(jax.jit(_jp._attention)(q, q, q))
        return "flash"
    except Exception as e:  # kernel missing in this jax build
        print(f"[attn] flash probe failed ({type(e).__name__}: {e}); xla")
        return "xla"


def _hf_download(repo, out):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_dir=str(out),
                          allow_patterns=["ckpt.npz", "state.json"])
        return (Path(out) / "ckpt.npz").exists()
    except Exception as e:  # first session: the repo may not exist yet
        print(f"[resume] no checkpoint pulled from {repo}: {e}")
        return False


def _hf_upload(repo, out):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    for name in ("state.json", "log.jsonl", "ckpt.npz"):
        f = Path(out) / name
        if f.exists():
            api.upload_file(path_or_fileobj=str(f), path_in_repo=name,
                            repo_id=repo)


def main(argv=None):
    a = _args(argv)
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = PRESETS[a.preset]
    if a.vocab:
        cfg = ModelConfig(**{**cfg.__dict__, "vocab_size": a.vocab})
    if a.seq > cfg.max_seq:
        cfg = ModelConfig(**{**cfg.__dict__, "max_seq": a.seq})
    tokens_per_step = a.batch * a.grad_accum * a.seq
    steps = a.steps or int(a.tokens // tokens_per_step)
    if steps <= 0:
        raise SystemExit("set --steps or --tokens")
    warmup = a.warmup or max(100, steps // 100)
    warmup = min(warmup, max(1, steps // 2))
    dtype = jnp.bfloat16 if a.precision == "bf16" else jnp.float32

    devs = jax.devices()
    _jp.ATTN_IMPL = _pick_attn(a.attn, devs[0].platform, cfg)
    print(f"[attn] {_jp.ATTN_IMPL}")
    if a.batch % len(devs):
        raise SystemExit(f"--batch {a.batch} must divide by {len(devs)} "
                         "devices")
    mesh = Mesh(np.array(devs), ("data",))
    rep = NamedSharding(mesh, P())
    data_sh = NamedSharding(mesh, P(None, "data", None))  # [accum, B, T]

    if a.synthetic:
        loader = _Synthetic(cfg.vocab_size, a.seq, a.seed)
    elif a.mix:
        import os

        def mk(path, seed):
            return MixtureLoader(path, a.seq, seed=seed,
                                 cache_dir=str(out / "token_cache"),
                                 keep_files=a.keep_shards,
                                 token=os.environ.get("HF_TOKEN"))
        loader = mk(a.mix, a.seed)
    else:
        raise SystemExit("set --mix or --synthetic")
    decay = mk(a.mix_decay, a.seed + 1) if a.mix and a.mix_decay else None
    decay_from = int(steps * (1 - a.decay_frac)) if decay else steps

    def loader_state():
        st = {"main": loader.state()}
        if decay:
            st["decay"] = decay.state()
        return st

    start = 0
    got = load_checkpoint(out)
    if got is None and a.hf_ckpt_repo and _hf_download(a.hf_ckpt_repo, out):
        got = load_checkpoint(out)
    if got is not None:
        params, opt, meta = got
        if meta["config"] != cfg.__dict__:
            raise SystemExit(f"checkpoint config {meta['config']} != {cfg}")
        start = meta["step"]
        if meta.get("loader"):
            loader.load_state(meta["loader"]["main"])
            if decay and "decay" in meta["loader"]:
                decay.load_state(meta["loader"]["decay"])
        opt["step"] = np.asarray(opt["step"], np.int32)
        print(f"[resume] step {start}")
    else:
        params = init_params(cfg, a.seed)
        opt = adamw_init(params)
    params = jax.device_put(params, rep)
    opt = jax.device_put(opt, rep)

    n_params = cfg.n_params()
    flops_tok = cfg.flops_per_token(a.seq)
    print(f"[model] {a.preset}: {n_params / 1e6:.1f}M params, "
          f"{len(devs)} x {devs[0].platform}, {tokens_per_step} tok/step, "
          f"{steps} steps = {steps * tokens_per_step / 1e9:.2f}B tokens")

    def grads_of(p, xs, ys):
        def micro(carry, xy):
            loss, g = jax.value_and_grad(loss_fn)(p, xy[0], xy[1], cfg,
                                                  dtype, a.ce_chunk)
            return (carry[0] + loss,
                    jax.tree_util.tree_map(jnp.add, carry[1], g)), None
        zero = (jnp.zeros((), jnp.float32),
                jax.tree_util.tree_map(jnp.zeros_like, p))
        (loss, g), _ = jax.lax.scan(micro, zero, (xs, ys))
        k = xs.shape[0]
        return loss / k, jax.tree_util.tree_map(lambda t: t / k, g)

    @jax.jit
    def train_step(p, o, xs, ys, lr):
        loss, g = grads_of(p, xs, ys)
        p, o, gn = adamw_update(p, g, o, lr, weight_decay=a.wd)
        return p, o, loss, gn

    @jax.jit
    def eval_loss(p, x, y):
        return loss_fn(p, x, y, cfg, dtype, a.ce_chunk)

    val = loader.val_windows(a.eval_windows)
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    t_start = time.time()
    t_last, step_last = time.time(), start

    def save(step):
        save_checkpoint(out, jax.device_get(params), jax.device_get(opt),
                        step, cfg, loader_state(),
                        {"tokens": step * tokens_per_step,
                         "preset": a.preset})
        if a.hf_ckpt_repo:
            _hf_upload(a.hf_ckpt_repo, out)
        print(f"[save] step {step}")

    step = start
    while step < steps:
        src = decay if step >= decay_from else loader
        xs, ys = zip(*[src.next_batch(a.batch)
                       for _ in range(a.grad_accum)])
        xs = jax.device_put(np.stack(xs), data_sh)
        ys = jax.device_put(np.stack(ys), data_sh)
        lr = lr_at(step, a.lr, warmup, steps, a.min_lr_ratio)
        params, opt, loss, gn = train_step(params, opt, xs, ys,
                                           jnp.float32(lr))
        step += 1
        if step % a.log_every == 0 or step == steps:
            loss_f = float(loss)
            dt = time.time() - t_last
            tps = (step - step_last) * tokens_per_step / max(dt, 1e-9)
            rec = {"step": step, "loss": round(loss_f, 4),
                   "lr": lr, "grad_norm": round(float(gn), 4),
                   "tokens": step * tokens_per_step,
                   "tok_s": round(tps),
                   "mfu": round(tps * flops_tok / (a.peak_tflops * 1e12
                                                   * len(devs)), 4)}
            if not np.isfinite(loss_f):
                raise SystemExit(f"non-finite loss at step {step}: {rec}")
            if val is not None and (step % a.eval_every == 0
                                    or step == steps):
                vl = [float(eval_loss(params, val[0][i:i + a.batch],
                                      val[1][i:i + a.batch]))
                      for i in range(0, len(val[0]) - a.batch + 1,
                                     a.batch)] or [float("nan")]
                rec["val_loss"] = round(float(np.mean(vl)), 4)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            t_last, step_last = time.time(), step
        if step % a.save_every == 0:
            save(step)
        if a.max_hours and time.time() - t_start > a.max_hours * 3600:
            print(f"[stop] --max-hours {a.max_hours} reached at step {step}")
            break
    if step % a.save_every:
        save(step)
    log.close()
    if a.export and step >= steps:
        export_hf(jax.device_get(params), cfg, a.export, a.tokenizer)
        print(f"[export] Llama-format checkpoint in {a.export}")
    return step


if __name__ == "__main__":
    main()

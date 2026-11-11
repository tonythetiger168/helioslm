"""own_base_shards.py — cut fetch_mixture.py output into TPU-sized token
shards and write the mixture files for jax_pretrain_tpu.py (v5.266).

  python examples/own_base_shards.py --data /workspace/own_data \\
      --out /workspace/own_shards --shard-tokens 2.5e8 \\
      [--upload <user>/helioslm-own-0p3b-tokens]

Input: ``--data`` as written by ``examples/fetch_mixture.py`` (its
``manifest.json`` and ``<phase>/<name>.bin``, uint32). Output in
``--out``:

  - ``<phase>/<name>/000.bin`` ... (uint16 when every id fits, which is
    the case for the own 32,768-token BPE; otherwise uint32);
  - ``val.bin``: the last ``--val-tokens`` of every main-phase source,
    cut off before sharding, so validation text is never trained on;
  - ``mix_main.json`` / ``mix_decay.json``: ``MixtureLoader`` mixtures,
    each source weighted by the tokens it actually has (its share of the
    phase budget after gated / skipped sources were dropped);
  - ``shards.json``: per-shard token counts and SHA-256;
  - ``tokenizer.json``: copied from the manifest's tokenizer path.

``--upload`` pushes the folder to a private Hugging Face dataset repo
(needs HF_TOKEN); the mixture files then name that repo as ``hf_repo``
so the Kaggle notebook downloads one shard per source at a time.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def shard_source(src, dst_dir, shard_tokens, val_tokens, dtype):
    """Split one uint32 .bin; returns (shard records, val array)."""
    data = np.memmap(src, dtype=np.uint32, mode="r")
    n = len(data)
    val = None
    if val_tokens and n > 4 * val_tokens:
        val = np.asarray(data[n - val_tokens:])
        n -= val_tokens
    dst_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, s in enumerate(range(0, n, shard_tokens)):
        chunk = np.asarray(data[s:min(n, s + shard_tokens)])
        if dtype == "uint16":
            chunk = chunk.astype(np.uint16)
        path = dst_dir / f"{i:03d}.bin"
        chunk.tofile(path)
        recs.append({"file": path.as_posix(), "tokens": int(len(chunk)),
                     "sha256": hashlib.sha256(chunk.tobytes()).hexdigest()})
    return recs, val


def max_id(path, block=1 << 26):
    data = np.memmap(path, dtype=np.uint32, mode="r")
    return max((int(data[i:i + block].max())
                for i in range(0, len(data), block)), default=0)


def build(data_dir, out_dir, shard_tokens, val_tokens, hf_repo=None):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    man = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    bins = sorted(data_dir.glob("*/*.bin"))
    if not bins:
        raise SystemExit(f"no <phase>/<name>.bin under {data_dir}")
    top = max(max_id(b) for b in bins)
    dtype = "uint16" if top < 65536 else "uint32"
    out_dir.mkdir(parents=True, exist_ok=True)
    mixes, vals, index = {}, [], {}
    for b in bins:
        phase, name = b.parent.name, b.stem
        recs, val = shard_source(b, out_dir / phase / name, shard_tokens,
                                 val_tokens if phase == "main" else 0, dtype)
        if not recs:
            continue
        if val is not None:
            vals.append(val.astype(np.uint16 if dtype == "uint16"
                                   else np.uint32))
        rel = [Path(r["file"]).relative_to(out_dir).as_posix() for r in recs]
        for r, f in zip(recs, rel):
            r["file"] = f
        index[f"{phase}/{name}"] = recs
        tokens = sum(r["tokens"] for r in recs)
        mixes.setdefault(phase, []).append(
            {"name": name, "weight": tokens, "files": rel})
    if vals:
        np.concatenate(vals).tofile(out_dir / "val.bin")
    for phase, srcs in mixes.items():
        total = sum(s["weight"] for s in srcs)
        for s in srcs:
            s["tokens"] = s["weight"]
            s["weight"] = round(s["weight"] / total, 6)
        mix = {"dtype": dtype, "sources": srcs,
               "mixture": man.get("mixture"),
               "tokenizer": man.get("tokenizer"),
               "eos": man.get("eos"), "bos": man.get("bos")}
        if vals:
            mix["val"] = "val.bin"
        if hf_repo:
            mix["hf_repo"] = hf_repo
        (out_dir / f"mix_{phase}.json").write_text(
            json.dumps(mix, indent=1), encoding="utf-8")
    tok = man.get("tokenizer")
    if tok and Path(tok).is_file():
        import shutil
        shutil.copy(tok, out_dir / "tokenizer.json")
    (out_dir / "shards.json").write_text(json.dumps(
        {"dtype": dtype, "max_id": top, "shard_tokens": shard_tokens,
         "val_tokens_per_source": val_tokens, "shards": index}, indent=1),
        encoding="utf-8")
    return mixes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-tokens", type=float, default=2.5e8)
    ap.add_argument("--val-tokens", type=float, default=2e6)
    ap.add_argument("--upload", help="private HF dataset repo")
    a = ap.parse_args(argv)
    mixes = build(a.data, a.out, int(a.shard_tokens), int(a.val_tokens),
                  a.upload)
    for ph, srcs in mixes.items():
        for s in srcs:
            print(f"{ph:5} {s['name']:24} {s['tokens'] / 1e9:7.3f}B  "
                  f"w={s['weight']:.3f}  {len(s['files'])} shards")
    if a.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.upload, repo_type="dataset", private=True,
                        exist_ok=True)
        api.upload_large_folder(repo_id=a.upload, repo_type="dataset",
                                folder_path=a.out)
        print(f"[upload] {a.out} -> datasets/{a.upload}")


if __name__ == "__main__":
    main()

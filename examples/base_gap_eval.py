"""base_gap_eval.py — route B: how far is the untouched base from every
MediaTek model? (v5.128)

One pod run, before any training is paid for: score the base
(google/gemma-4-E4B) and the four MediaTek Research text models on
TMMLU+ (5-shot), TW-JusticeBench (0-shot), C-Eval val (5-shot) and CMMLU
(5-shot) with the same scorer (helioslm_v5/eval/tw_bench.py, log-likelihood
of the answer letter), then write the base-minus-rival gap per task with a
95% paired-bootstrap interval (eval/prereg.paired_bootstrap).

  # on the pod (docs/BASE_GAP_EVAL.md has the full steps)
  python examples/base_gap_eval.py --fetch --cleanup --stop-pod

  # smoke: first 20 questions per task, smallest models only
  python examples/base_gap_eval.py --fetch --limit 20 \\
      --models google/gemma-4-E4B MediaTek-Research/Llama-Breeze2-3B-Instruct

Needs transformers, accelerate, datasets and huggingface_hub on the pod,
and HF_TOKEN for gated model cards. Models are scored one at a time and
freed; --cleanup also deletes each model's Hub cache afterwards so the
whole run fits in ~120 GB of disk. A model or task that fails is recorded
and the run goes on. The output holds scores and task ids only, never
question text (TW1 licence rule). Results are rewritten after every
model, so a pod that dies keeps what finished.
"""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval import tw_bench  # noqa: E402
from helioslm_v5.eval.prereg import paired_bootstrap  # noqa: E402

BASE = "google/gemma-4-E4B"
RIVALS = (
    "MediaTek-Research/Breexe-8x7B-Instruct-v0_1",
    "MediaTek-Research/Llama-Breeze2-8B-Instruct",
    "MediaTek-Research/Llama-Breeze2-3B-Instruct",
    "MediaTek-Research/Breeze-7B-Instruct-v1_0",
)
# bench: (dataset id, eval split, shot split, shots). The splits are the
# file-name suffixes fetch() writes.
TASKS = {
    "tmmluplus": ("ikala/tmmluplus", "test", "dev", 5),     # CSV snapshot
    "tw_justice": ("JerryWu/TW-JusticeBench", "test", "dev", 0),
    "ceval": ("ceval/ceval-exam", "val", "dev", 5),
    "cmmlu": ("haonan-li/cmmlu", "test", "dev", 5),
}
# Hub split name -> file suffix
_SPLITS = {"test": "test", "validation": "val", "val": "val", "dev": "dev",
           "train": "train"}


def _fetch_tmmluplus(out):
    # the CSV layout gpu_batch_run.py already uses: data/<subject>_<split>.csv
    from huggingface_hub import snapshot_download
    tmp = out.parent / "_tmmluplus_raw"
    snapshot_download("ikala/tmmluplus", repo_type="dataset",
                      local_dir=str(tmp), allow_patterns=["data/*.csv"])
    n = 0
    for f in sorted((tmp / "data").glob("*.csv")):
        shutil.copy(f, out / f.name)
        n += 1
    return n


def _fetch_datasets(repo, out):
    from datasets import get_dataset_config_names, load_dataset
    configs = [c for c in get_dataset_config_names(repo)
               if c not in ("all", "default")] or [None]
    for cfg in configs:
        ds = load_dataset(repo, cfg)
        subject = cfg or "all"
        for split, rows in ds.items():
            suffix = _SPLITS.get(split, split)
            with open(out / f"{subject}_{suffix}.jsonl", "w",
                      encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(configs)


def _fetch_zip(repo, out):
    """Fallback for repos that ship a zip of per-split CSV folders
    (C-Eval: val/<subject>_val.csv; CMMLU: test/<subject>.csv)."""
    import zipfile
    from huggingface_hub import snapshot_download
    tmp = out.parent / ("_raw_" + out.name)
    snapshot_download(repo, repo_type="dataset", local_dir=str(tmp),
                      allow_patterns=["*.zip"])
    for z in tmp.rglob("*.zip"):
        zipfile.ZipFile(z).extractall(tmp / "x")
    n = 0
    for f in (tmp / "x").rglob("*.csv"):
        split = _SPLITS.get(f.parent.name)
        if split is None:
            continue
        shutil.copy(f, out / f"{tw_bench.subject_of(f)}_{split}.csv")
        n += 1
    if not n:
        raise RuntimeError(f"no per-split CSV folders in {repo}'s zip")
    return n


def fetch(data_dir, benches):
    """Download each bench's questions to data_dir/<bench>/ as
    <subject>_<split>.{csv,jsonl}. Returns {bench: status}."""
    status = {}
    for bench in benches:
        repo = TASKS[bench][0]
        out = Path(data_dir) / bench
        if out.exists() and any(out.glob("*_*.*")):
            status[bench] = "already there"
            continue
        out.mkdir(parents=True, exist_ok=True)
        tries = ([_fetch_tmmluplus] if bench == "tmmluplus"
                 else [lambda o, r=repo: _fetch_datasets(r, o),
                       lambda o, r=repo: _fetch_zip(r, o)])
        errors = []
        for fn in tries:
            try:
                status[bench] = f"ok ({fn(out)} files/subjects)"
                break
            except Exception as e:                   # recorded, run goes on
                errors.append(f"{type(e).__name__}: {e}"[:300])
        else:
            status[bench] = "FAILED: " + " | ".join(errors)
    return status


def load_tasks(data_dir, bench, limit=None):
    _, split, shot_split, k = TASKS[bench]
    path = Path(data_dir) / bench
    tasks = tw_bench.load_mcq(path, bench, split)
    if limit:
        tasks = tasks[:limit]
    tw_bench.self_check(tasks)
    shots = None
    if k:
        shots = tw_bench.shots_by_subject(
            tw_bench.load_mcq(path, bench, shot_split), k)
    return tasks, shots


def gap_table(scores, base=BASE, rivals=RIVALS, seed=0):
    """scores: {model: {bench: {task_id: correct}}} -> {bench: {rival:
    paired_bootstrap(base, rival) over the task ids both scored}}."""
    out = {}
    for bench in sorted({b for m in scores.values() for b in m}):
        ours = scores.get(base, {}).get(bench)
        if not ours:
            continue
        row = {}
        for r in rivals:
            theirs = scores.get(r, {}).get(bench)
            if not theirs:
                continue
            ids = sorted(set(ours) & set(theirs))
            if not ids:
                continue
            row[r] = paired_bootstrap([[float(ours[i]) for i in ids]],
                                      [float(theirs[i]) for i in ids],
                                      seed=seed)
        out[bench] = row
    return out


def _free(model_id, cleanup):
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    if cleanup:
        from huggingface_hub.constants import HF_HUB_CACHE
        d = Path(HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--"))
        shutil.rmtree(d, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[BASE, *RIVALS])
    ap.add_argument("--benches", nargs="+", default=list(TASKS),
                    choices=list(TASKS))
    ap.add_argument("--data", default="data/tw_bench")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N questions per task (smoke; not comparable)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete each model's Hub cache after scoring it")
    ap.add_argument("--stop-pod", action="store_true")
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"base_gap_{stamp}.json"
    report = {"schema": "base-gap/1", "started": stamp, "base": BASE,
              "limit": a.limit, "fetch": None, "models": {}, "gaps": {}}
    scores = {}

    def save():
        report["gaps"] = gap_table(scores)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")

    if a.fetch:
        report["fetch"] = fetch(a.data, a.benches)
        print(json.dumps(report["fetch"], ensure_ascii=False, indent=2))
    sets = {}
    for bench in a.benches:
        try:
            sets[bench] = load_tasks(a.data, bench, a.limit)
        except Exception as e:
            report.setdefault("task_errors", {})[bench] = \
                f"{type(e).__name__}: {e}"[:500]
    save()

    from examples.tw_bench_run import load_hf
    for mid in a.models:
        row = report["models"][mid] = {"status": "running", "benches": {}}
        t0 = time.time()
        try:
            model, tok, sha = load_hf(mid, device=a.device)
            row["revision"] = sha
            score = tw_bench.hf_score_fn(model, tok, a.max_length)
            for bench, (tasks, shots) in sets.items():
                t1 = time.time()
                try:
                    res = tw_bench.run_loglik(tasks, score, shots)
                    s = tw_bench.summarize(res)
                    row["benches"][bench] = {
                        "n": len(res), "accuracy": s["accuracy"],
                        "macro": s.get("macro"),
                        "seconds": round(time.time() - t1)}
                    scores.setdefault(mid, {})[bench] = {
                        r["task"]: r["correct"] for r in res}
                except Exception as e:
                    row["benches"][bench] = {
                        "error": f"{type(e).__name__}: {e}"[:500]}
                save()
            row["status"] = "done"
            del model, tok, score
        except Exception as e:
            row["status"] = "FAILED"
            row["error"] = traceback.format_exc()[-1500:]
        row["seconds"] = round(time.time() - t0)
        _free(mid, a.cleanup)
        save()
        print(json.dumps({mid: {k: v for k, v in row.items()
                                if k != "error"}}, ensure_ascii=False))
    report["finished"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save()
    print(f"wrote {out}")
    if a.stop_pod and os.environ.get("RUNPOD_POD_ID"):
        subprocess.run(["runpodctl", "stop", "pod", os.environ["RUNPOD_POD_ID"]])
    return report


if __name__ == "__main__":
    main()

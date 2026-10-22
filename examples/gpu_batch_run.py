"""gpu_batch_run.py — every GPU-pending job in one go, one results file.

Run from the repo root on the GPU machine (RunPod pod or a local GPU):

    nohup python examples/gpu_batch_run.py --stop-pod > /workspace/batch.log 2>&1 &
    python examples/gpu_batch_run.py --plan        # what would run / skip, then exit

Jobs, in order (each has its own time limit; a failed or timed-out job is
recorded and the next one still runs):

  cuda_smoke   examples/cuda_smoke.py (#1, all CUDA + training checks)
  os5_mtp      examples/mtp_rollout_bench.py (OS5 MTP speculative rollouts)
  os7_longctx  examples/longctx_arch_probe.py (OS7 RULER-lite, 4 layouts)
  pilot_data   examples/pilot_prepare_data.py, only when OS6 or tw_bench
               needs the 100M pilot data and it is not there yet
  os6_sweep    examples/moe_sweep_100m.py (OS6), skipped until #72
               (moe.dispatch) is on this checkout
  tw_fetch     TMMLU+ CSVs from the Hub into data/tw_bench/ (local only,
               git-ignored; TW1 licence rule), skipped until #86 is merged
  tw_bench     examples/tw_bench_run.py (E3/TW3 real scores) on a pilot
               checkpoint (--tw-run, default the earlier dense pilot)

Results: <out-dir>/gpu_batch_<UTC stamp>.json holds the environment (git
commit, GPU, torch), one row per job (status PASS / FAIL / TIMEOUT / SKIP,
seconds, command, log path, last log lines) and each job's own JSON output
inlined under "result". It is rewritten after every job, so a pod that
dies mid-run still leaves what finished. Logs: <out-dir>/logs/<job>.log.

--smoke runs every job at toy size (CPU is fine): it checks the wiring and
the result file, not CUDA. Uses only the standard library, so --plan works
without torch.
"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCHEMA_ID = "gpu-batch/1"
SKIP_EXIT = 3          # moe_sweep_100m.py: "not applicable on this checkout"
JOBS = ("cuda_smoke", "os5_mtp", "os7_longctx", "pilot_data", "os6_sweep",
        "tw_fetch", "tw_bench")


@dataclass
class Job:
    name: str
    cmd: list
    timeout_min: float
    output: str = None           # JSON the job writes; inlined in the results
    skip: str = None             # reason, when the job does not apply
    ok_codes: tuple = (0,)
    env: dict = field(default_factory=dict)


def has_moe_dispatch(repo):
    """#72 merged: MoEConfig has a ``dispatch`` field (text check, no torch)."""
    src = (repo / "helioslm_v5" / "configs" / "config_v5.py").read_text()
    return re.search(r"^\s+dispatch:\s*str\b", src, re.M) is not None


def has_tw_bench(repo):
    """#86 merged: the TW3 runner and eval module exist."""
    return ((repo / "examples" / "tw_bench_run.py").exists()
            and (repo / "helioslm_v5" / "eval" / "tw_bench.py").exists())


def pilot_run_ok(run):
    run = Path(run)
    return (run / "run.json").exists() and (run / "ckpt.pt").exists()


def build_jobs(args, repo=REPO):
    py = args.python
    out = Path(args.out_dir)
    dev = args.device
    smoke = args.smoke
    data = Path(args.pilot_data)
    jobs = []

    smoke_only = ["--only", "forward_fp32", "greedy_generate"] if smoke else []
    jobs.append(Job("cuda_smoke", [
        py, "examples/cuda_smoke.py", "--device", dev,
        "--out", str(out / "gpu_smoke.json"), *smoke_only],
        45, str(out / "gpu_smoke.json")))

    os5 = ["--tiny-vocab", "64", "--groups", "1", "--group-size", "2",
           "--max-new", "4", "--tv-samples", "50", "--tv-max", "1.0"] if smoke else []
    jobs.append(Job("os5_mtp", [
        py, "examples/mtp_rollout_bench.py", "--device", dev,
        "--output", str(out / "mtp_rollout.json"), *os5],
        30, str(out / "mtp_rollout.json")))

    os7 = (["--steps", "5", "--train-len", "64", "--arms", "dense", "--hidden", "64",
            "--eval-batches", "1", "--batch", "4"] if smoke else [])
    jobs.append(Job("os7_longctx", [
        py, "examples/longctx_arch_probe.py", "--device", dev,
        "--output", str(out / "longctx_arch.json"), *os7],
        120, str(out / "longctx_arch.json")))

    want_os6 = "os6_sweep" in args.jobs
    want_tw = "tw_bench" in args.jobs
    tw_run = Path(args.tw_run) if args.tw_run else None
    os6_skip = None if has_moe_dispatch(repo) else \
        "#72 (moe.dispatch) is not on this checkout"
    tw_skip = None if has_tw_bench(repo) else \
        "#86 (eval/tw_bench.py) is not on this checkout"
    tw_data = Path(args.tw_data)
    tw_files = (tw_data / "tmmluplus" / "data").is_dir()
    if tw_skip is None and smoke and not tw_files:
        tw_skip = f"smoke: no TMMLU+ files under {tw_data} (no Hub download)"
    if tw_skip is None and not smoke and (tw_run is None or not pilot_run_ok(tw_run)):
        tw_skip = f"no pilot checkpoint at {tw_run} (run.json + ckpt.pt); pass --tw-run"

    # the pilot data feeds OS6 only (tw_bench reads the data its run names)
    data_skip = None
    if smoke:
        data_skip = "smoke: OS6 runs as --dry-run"
    elif not (want_os6 and os6_skip is None):
        data_skip = "no job that needs it will run"
    elif (data / "meta.json").exists():
        data_skip = f"already there ({data})"
    jobs.append(Job("pilot_data", [
        py, "examples/pilot_prepare_data.py",
        "--hf", "HuggingFaceFW/fineweb-edu:sample-10BT",
        "--max-docs", "100000", "--vocab", "32768", "--out", str(data)],
        60, skip=data_skip))

    if smoke:
        sweep = ["--dry-run", "--arms", "base", "gran_x2", "shared0"]
    else:
        sweep = ["--data", str(data), "--steps", str(args.os6_steps)]
        if args.compile:
            sweep.append("--compile")
        if args.peak_tflops:
            sweep += ["--peak-tflops", str(args.peak_tflops)]
    jobs.append(Job("os6_sweep", [
        py, "examples/moe_sweep_100m.py", "--out", str(out / "os6_runs"),
        "--output", str(out / "moe_sweep.json"), "--device", dev, *sweep],
        args.os6_timeout, str(out / "moe_sweep.json"), skip=os6_skip))

    fetch_skip = tw_skip
    if fetch_skip is None and not want_tw:
        fetch_skip = "no job that needs it will run"
    if fetch_skip is None and tw_files:
        fetch_skip = f"already there ({tw_data / 'tmmluplus'})"
    jobs.append(Job("tw_fetch", [
        py, "-c",
        "import sys; from huggingface_hub import snapshot_download; "
        "snapshot_download('ikala/tmmluplus', repo_type='dataset', "
        "local_dir=sys.argv[1], allow_patterns=['data/*.csv'])",
        str(tw_data / "tmmluplus")], 30, skip=fetch_skip))

    tw_args = ["--limit", "8"] if smoke else (
        ["--limit", str(args.tw_limit)] if args.tw_limit else [])
    tw_out = out / "tw_bench_tmmluplus.json"
    if smoke and tw_skip is None:
        tw_cmd = [py, "examples/tw_bench_run.py", "--bench", "tmmluplus",
                  "--data", str(tw_data / "tmmluplus" / "data"),
                  "--check-only", *tw_args]
        tw_out = None
    else:
        tw_cmd = [py, "examples/tw_bench_run.py", "--bench", "tmmluplus",
                  "--data", str(tw_data / "tmmluplus" / "data"),
                  "--run", str(tw_run), "--shots", str(args.tw_shots),
                  "--device", dev, "--out", str(tw_out), *tw_args]
    jobs.append(Job("tw_bench", tw_cmd, 180,
                    str(tw_out) if tw_out else None, skip=tw_skip))

    for j in jobs:
        if j.name not in args.jobs and j.skip is None:
            j.skip = "not selected (--jobs)"
    return jobs


def git_info(repo):
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=repo, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"commit": git("rev-parse", "HEAD"),
            "subject": git("log", "-1", "--format=%s"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def gpu_info():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()] or None
    except (OSError, subprocess.SubprocessError):
        return None


def torch_info(py):
    code = ("import json, torch; print(json.dumps({'torch': torch.__version__, "
            "'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available()}))")
    try:
        r = subprocess.run([py, "-c", code], capture_output=True, text=True,
                           timeout=120)
        return json.loads(r.stdout) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _tail(path, n=15):
    try:
        return Path(path).read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def run_job(job, log_dir, repo=REPO):
    row = {"job": job.name, "cmd": job.cmd, "timeout_min": job.timeout_min}
    if job.skip:
        row.update(status="SKIP", reason=job.skip)
        return row
    log = Path(log_dir) / f"{job.name}.log"
    row["log"] = str(log)
    if job.output:
        Path(job.output).unlink(missing_ok=True)      # never inline a stale result
    t0 = time.time()
    with open(log, "w") as f:
        try:
            p = subprocess.run(job.cmd, cwd=repo, stdout=f,
                               stderr=subprocess.STDOUT,
                               env={**os.environ, **job.env},
                               timeout=job.timeout_min * 60)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = None
    row["seconds"] = round(time.time() - t0, 1)
    row["returncode"] = rc
    if rc is None:
        row["status"] = "TIMEOUT"
    elif rc == SKIP_EXIT and job.name == "os6_sweep":
        row["status"] = "SKIP"
        row["reason"] = "the job reported it does not apply (see log)"
    else:
        row["status"] = "PASS" if rc in job.ok_codes else "FAIL"
    if job.output:
        row["output"] = job.output
        try:
            row["result"] = json.loads(Path(job.output).read_text())
        except (OSError, ValueError):
            row["result"] = None
    row["tail"] = _tail(log)
    return row


def write_results(path, doc):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def stop_pod():
    pod = os.environ.get("RUNPOD_POD_ID")
    if shutil.which("runpodctl") and pod:
        print("stopping pod", pod, flush=True)
        subprocess.run(["runpodctl", "stop", "pod", pod])
    else:
        print("runpodctl or RUNPOD_POD_ID missing: stop the pod by hand",
              flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default="/workspace/results")
    ap.add_argument("--repo", default=str(REPO), help=argparse.SUPPRESS)
    ap.add_argument("--jobs", nargs="+", default=list(JOBS), choices=JOBS,
                    help="jobs to run (default all; pilot_data and tw_fetch "
                         "run only when a selected job needs them)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--pilot-data", default="/workspace/data/pilot100k")
    ap.add_argument("--os6-steps", type=int, default=600,
                    help="steps per OS6 arm (8 arms; 600 x 32k tokens ~ 20M)")
    ap.add_argument("--os6-timeout", type=float, default=360,
                    help="minutes for the whole OS6 sweep")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the OS6 arms")
    ap.add_argument("--peak-tflops", type=float,
                    help="MFU denominator for OS6 (RTX 6000 Ada bf16: 182)")
    ap.add_argument("--tw-run", default="/workspace/results/dense_compile",
                    help="pilot_100m.py run directory to score on TMMLU+")
    ap.add_argument("--tw-data", default="/workspace/data/tw_bench")
    ap.add_argument("--tw-shots", type=int, default=5)
    ap.add_argument("--tw-limit", type=int, default=None,
                    help="score only the first N questions (not comparable)")
    ap.add_argument("--smoke", action="store_true",
                    help="toy-size run of every job (wiring check, CPU ok)")
    ap.add_argument("--plan", action="store_true",
                    help="print what would run or be skipped, then exit")
    ap.add_argument("--stop-pod", action="store_true",
                    help="runpodctl stop pod $RUNPOD_POD_ID at the end")
    args = ap.parse_args(argv)
    # pilot_data / tw_fetch follow the jobs that need them
    args.jobs = list(args.jobs) + [j for j in ("pilot_data", "tw_fetch")
                                   if j not in args.jobs]

    out = Path(args.out_dir)
    repo = Path(args.repo)
    jobs = build_jobs(args, repo)
    if args.plan:
        for j in jobs:
            state = f"SKIP ({j.skip})" if j.skip else "RUN"
            print(f"{j.name:12s} {state}\n    {' '.join(j.cmd)}")
        return 0

    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = out / f"gpu_batch_{stamp}.json"
    doc = {"schema": SCHEMA_ID, "started": stamp, "finished": None,
           "smoke": args.smoke, "device": args.device,
           "environment": {"git": git_info(repo), "gpu": gpu_info(),
                           "torch": torch_info(args.python),
                           "python": platform.python_version(),
                           "platform": platform.platform()},
           "jobs": []}
    write_results(path, doc)
    print("results:", path, flush=True)
    for j in jobs:
        if not j.skip:
            print(f"=== {time.strftime('%H:%M:%S', time.gmtime())} {j.name}",
                  flush=True)
        row = run_job(j, log_dir, repo)
        doc["jobs"].append(row)
        write_results(path, doc)
        print(f"[{row['status']}] {j.name} {row.get('seconds', '')}"
              f"{' ' + row['reason'] if 'reason' in row else ''}", flush=True)
    doc["finished"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    counts = {}
    for r in doc["jobs"]:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    doc["summary"] = counts
    write_results(path, doc)
    print("summary", json.dumps(counts), "->", path, flush=True)
    if args.stop_pod:
        stop_pod()
    return 1 if counts.get("FAIL") or counts.get("TIMEOUT") else 0


if __name__ == "__main__":
    sys.exit(main())

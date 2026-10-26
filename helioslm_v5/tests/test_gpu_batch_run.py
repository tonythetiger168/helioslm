"""examples/gpu_batch_run.py: the one-shot GPU batch runner.

The orchestration tests use a fake repo whose example scripts are stubs,
so they need neither torch nor a GPU. The last two tests run the real
scripts at toy size on the CPU (torch only).
"""
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "gpu_batch_run", ROOT / "examples" / "gpu_batch_run.py")
gbr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gbr)

STUB = textwrap.dedent("""
    import json, sys, time
    a = sys.argv[1:]
    out = next((a[i + 1] for i, x in enumerate(a) if x in ("--out", "--output")), None)
    if {behaviour!r} == "sleep":
        time.sleep(30)
    if out:
        open(out, "w").write(json.dumps({{"stub": {name!r}, "argv": a}}))
    print("stub", {name!r}, "done")
    sys.exit({code})
""")


def fake_repo(tmp_path, dispatch=False, tw=False, codes=None, sleep=()):
    codes = codes or {}
    repo = tmp_path / "repo"
    (repo / "examples").mkdir(parents=True)
    (repo / "helioslm_v5" / "configs").mkdir(parents=True)
    (repo / "helioslm_v5" / "eval").mkdir(parents=True)
    cfg = "class MoEConfig:\n    first_k_dense: int = 0\n"
    if dispatch:
        cfg += '    dispatch: str = "blocked"\n'
    (repo / "helioslm_v5" / "configs" / "config_v5.py").write_text(cfg)
    scripts = ["cuda_smoke", "mtp_rollout_bench", "longctx_arch_probe",
               "pilot_prepare_data", "moe_sweep_100m"]
    if tw:
        scripts.append("tw_bench_run")
        (repo / "helioslm_v5" / "eval" / "tw_bench.py").write_text("")
    for s in scripts:
        (repo / "examples" / f"{s}.py").write_text(STUB.format(
            name=s, code=codes.get(s, 0),
            behaviour="sleep" if s in sleep else ""))
    return repo


def args_for(tmp_path, *extra):
    out = tmp_path / "results"
    return ["--out-dir", str(out), "--pilot-data", str(tmp_path / "data"),
            "--tw-data", str(tmp_path / "tw"), "--tw-run", str(tmp_path / "run"),
            *extra]


def parse(argv):
    """The args main() builds, without running anything."""
    captured = {}
    orig = gbr.build_jobs

    def spy(args, repo=gbr.REPO):
        captured["args"] = args
        return orig(args, repo)
    gbr.build_jobs = spy
    try:
        gbr.main([*argv, "--plan"])
    finally:
        gbr.build_jobs = orig
    return captured["args"]


def jobs_by_name(tmp_path, repo, *extra):
    args = parse(args_for(tmp_path, *extra))
    return {j.name: j for j in gbr.build_jobs(args, repo)}


def test_skips_until_prs_are_merged(tmp_path):
    repo = fake_repo(tmp_path)
    jobs = jobs_by_name(tmp_path, repo)
    assert list(jobs) == list(gbr.JOBS)
    for name in ("cuda_smoke", "os5_mtp", "os7_longctx"):
        assert jobs[name].skip is None
    assert "#72" in jobs["os6_sweep"].skip
    assert "#86" in jobs["tw_bench"].skip and "#86" in jobs["tw_fetch"].skip
    assert jobs["pilot_data"].skip == "no job that needs it will run"


def test_runs_os6_and_tw_once_merged(tmp_path):
    repo = fake_repo(tmp_path, dispatch=True, tw=True)
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text("{}")
    (run / "ckpt.pt").write_text("")
    jobs = jobs_by_name(tmp_path, repo)
    assert all(j.skip is None for j in jobs.values()), \
        {n: j.skip for n, j in jobs.items()}
    assert "--data" in jobs["os6_sweep"].cmd
    assert str(run) in jobs["tw_bench"].cmd
    # data already there: no re-download
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "meta.json").write_text("{}")
    (tmp_path / "tw" / "tmmluplus" / "data").mkdir(parents=True)
    jobs = jobs_by_name(tmp_path, repo)
    assert jobs["pilot_data"].skip.startswith("already there")
    assert jobs["tw_fetch"].skip.startswith("already there")


def test_tw_bench_needs_a_checkpoint(tmp_path):
    repo = fake_repo(tmp_path, tw=True)
    jobs = jobs_by_name(tmp_path, repo)
    assert "no pilot checkpoint" in jobs["tw_bench"].skip
    assert jobs["tw_fetch"].skip == jobs["tw_bench"].skip


def test_job_selection(tmp_path):
    repo = fake_repo(tmp_path, dispatch=True)
    jobs = jobs_by_name(tmp_path, repo, "--jobs", "os7_longctx")
    assert jobs["os7_longctx"].skip is None
    assert jobs["cuda_smoke"].skip == "not selected (--jobs)"
    assert jobs["pilot_data"].skip == "no job that needs it will run"


def test_end_to_end_results_file(tmp_path):
    repo = fake_repo(tmp_path, dispatch=True,
                     codes={"mtp_rollout_bench": 1, "moe_sweep_100m": 3},
                     sleep=("longctx_arch_probe",))
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "meta.json").write_text("{}")
    out = tmp_path / "results"
    out.mkdir()
    (out / "gpu_smoke.json").write_text('{"stale": true}')
    orig = gbr.build_jobs

    def short(args, repo=gbr.REPO):
        jobs = orig(args, repo)
        for j in jobs:
            if j.name == "os7_longctx":
                j.timeout_min = 0.02            # ~1 s: the stub sleeps 30 s
        return jobs
    gbr.build_jobs = short
    try:
        rc = gbr.main([*args_for(tmp_path), "--repo", str(repo),
                       "--python", sys.executable])
    finally:
        gbr.build_jobs = orig
    assert rc == 1                              # a FAIL and a TIMEOUT
    files = list(out.glob("gpu_batch_*.json"))
    assert len(files) == 1
    doc = json.loads(files[0].read_text())
    assert doc["schema"] == gbr.SCHEMA_ID and doc["finished"]
    rows = {r["job"]: r for r in doc["jobs"]}
    assert [r["job"] for r in doc["jobs"]] == list(gbr.JOBS)
    assert rows["cuda_smoke"]["status"] == "PASS"
    assert rows["cuda_smoke"]["result"]["stub"] == "cuda_smoke"   # not the stale file
    assert rows["os5_mtp"]["status"] == "FAIL"
    assert rows["os5_mtp"]["returncode"] == 1
    assert rows["os7_longctx"]["status"] == "TIMEOUT"
    assert rows["pilot_data"]["status"] == "SKIP"
    assert rows["os6_sweep"]["status"] == "SKIP"                  # exit code 3
    assert rows["tw_bench"]["status"] == "SKIP"
    assert any("done" in ln for ln in rows["cuda_smoke"]["tail"])
    assert Path(rows["cuda_smoke"]["log"]).exists()
    assert doc["summary"] == {"PASS": 1, "FAIL": 1, "TIMEOUT": 1, "SKIP": 4}


def test_plan_prints_without_running(tmp_path, capsys):
    repo = fake_repo(tmp_path)
    assert gbr.main([*args_for(tmp_path), "--repo", str(repo), "--plan"]) == 0
    text = capsys.readouterr().out
    assert "os6_sweep    SKIP" in text and "cuda_smoke   RUN" in text
    assert not (tmp_path / "results").exists()


# ------------------------------------------------------------ real scripts

def test_moe_sweep_arms_keep_params_equal():
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location(
        "moe_sweep_100m", ROOT / "examples" / "moe_sweep_100m.py")
    ms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ms)
    import torch
    from helioslm_v5.src.training.pretrain import HeliosLMv5, count_params

    def params(arm):
        cfg = ms.arm_config(arm, vocab_size=512, hidden=64, layers=4,
                            intermediate=256, seq=64)
        with torch.device("meta"):
            return count_params(HeliosLMv5(cfg)), cfg

    base, _ = params("base")
    for arm in ("gran_x2", "gran_x4", "shared0"):
        p, cfg = params(arm)
        # equal FFN parameters; only the router width differs
        for k in ("total", "active"):
            assert abs(p[k] - base[k]) / base[k] < 0.01, (arm, k, p, base)
    _, cfg = params("gran_x4")
    assert (cfg.moe.num_experts, cfg.moe.num_activated_experts,
            cfg.moe.num_shared_experts, cfg.moe.expert_hidden_size) == (16, 8, 4, 32)
    assert params("fkd3")[1].moe.first_k_dense == 3
    assert params("bal_qbal")[1].moe.balance_strategy == "quantile_balancing"


def test_smoke_run_on_cpu(tmp_path):
    pytest.importorskip("torch")
    out = tmp_path / "results"
    r = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "gpu_batch_run.py"),
         "--smoke", "--device", "cpu", "--out-dir", str(out),
         "--jobs", "cuda_smoke", "os5_mtp", "os7_longctx", "os6_sweep"],
        capture_output=True, text=True, timeout=600, cwd=ROOT)
    files = list(out.glob("gpu_batch_*.json"))
    assert files, r.stdout[-2000:] + r.stderr[-2000:]
    doc = json.loads(files[0].read_text())
    rows = {x["job"]: x for x in doc["jobs"]}
    for name in ("cuda_smoke", "os5_mtp", "os7_longctx"):
        assert rows[name]["status"] == "PASS", rows[name]
        assert rows[name]["result"], name
    assert rows["os6_sweep"]["status"] in ("PASS", "SKIP")   # SKIP until #72
    if rows["os6_sweep"]["status"] == "PASS":
        arms = [x["arm"] for x in rows["os6_sweep"]["result"]["results"]]
        assert arms == ["base", "gran_x2", "shared0"]
    assert r.returncode == 0, r.stdout[-2000:]

"""agentic_candidate_run.py — roadmap Phase 3 (P1): the first measured
five-component candidate run for the HeliosLM Agentic Index.

Runs ONE checkpoint through all five agentic components and writes the
native artifacts the adapters accept:

    terminal       run_term_bench() rows            artifacts/terminal.json
    swe            run_swe_bench() rows             artifacts/swe.json
    long_horizon   evaluate_policy() dict           artifacts/long_horizon.json
    tool_recovery  tool-recovery-run/1 object       artifacts/tool_recovery.json
    hdl            hdl-bench/1 dict                 artifacts/hdl.json

then generates the five ``agentic-component/1`` descriptors
(``evidence_type=model_run``, ``verification_tier=measured``) and the
aggregate ``agentic_index.json`` in the results directory.

Usage (from the repo root):

    python3 examples/agentic_candidate_run.py \
        --checkpoint checkpoints/toy_v5.13.pt \
        --results results/agentic/2026-11-23_toy-v5.13

Determinism: greedy decoding (temperature=0), fixed task order, fixed
seeds for the tool-recovery corpus — a rerun on the same checkpoint and
task sets reproduces every artifact byte-for-byte (HDL wall-clock fields
are not part of the schema).

Candidate notes: the committed toy checkpoint is a char-ascii model with
vocab 1024 and 2048 positions; prompts are truncated head+tail to fit,
and characters outside the vocab are mapped to '?'. Both are candidate
limitations, recorded here — they are not harness modifications.

HDL data: fetched from the pinned upstream commit. ``--hdl-cache`` may
point at an existing ``hdl_bench.fetch`` cache; otherwise the runner
fetches via one codeload tarball (raw.githubusercontent is intermittent
in some sandboxes) and verifies every sha256 through the normal
``hdl_bench.load`` path. Untrusted generated samples are simulated by
hdl_bench in per-sample temporary directories under timeouts; run this
inside a container, as the hdl_bench safety note requires.
"""
import argparse
import hashlib
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from helioslm_v5.eval import hdl_bench as hb  # noqa: E402
from helioslm_v5.eval import swe_bench as sb  # noqa: E402
from helioslm_v5.eval import term_bench as tb  # noqa: E402
from helioslm_v5.eval.agentic_adapters import generate_component  # noqa: E402
from helioslm_v5.eval.agentic_index import generate as generate_index  # noqa: E402
from helioslm_v5.eval.calibration_board import canonical_sha256  # noqa: E402
from helioslm_v5.eval.long_horizon_bench import (  # noqa: E402
    SCENARIOS, evaluate_policy)

# char-ascii vocab is 1024; prompts are truncated to fit the 2048-position
# window with the decode budget reserved.
PROMPT_HEAD = 1100
PROMPT_TAIL = 700
MAX_NEW = 128


def _clip(text):
    if len(text) <= PROMPT_HEAD + PROMPT_TAIL + 8:
        return text
    return text[:PROMPT_HEAD] + "\n[...]\n" + text[-PROMPT_TAIL:]


class CharCandidate:
    """One checkpoint -> every component's generate/policy function."""

    def __init__(self, checkpoint):
        import torch
        from helioslm_v5.configs.config_v5 import HeliosLMv5Config
        from helioslm_v5.src.model_v5 import HeliosLMv5
        self._torch = torch
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ckpt.get("tokenizer") != "char-ascii":
            raise ValueError("this runner expects the char-ascii toy checkpoint")
        self.model = HeliosLMv5(HeliosLMv5Config(size="lite"))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.checkpoint_sha256 = hashlib.sha256(
            Path(checkpoint).read_bytes()).hexdigest()
        self.model_name = f"helioslm-toy-{ckpt.get('meta', {}).get('val_loss', 0):.2f}"

    def generate(self, prompt, max_new=MAX_NEW):
        torch = self._torch
        ids = torch.tensor([[min(ord(c), 1023) if ord(c) < 0x110000
                             else ord("?") for c in _clip(prompt)]])
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new,
                                      temperature=0)
        return "".join(chr(int(t)) for t in out[0, ids.shape[1]:])

    # term_bench / swe_bench episode interface: (prompt, sample) -> text
    def episode(self, prompt, sample):
        return self.generate(prompt)

    # long_horizon policy interface: (scenario, observations) -> text
    def policy(self, scenario, observations):
        prompt = ("You are an agent that replies with exactly one tool call.\n"
                  f"Task: {scenario.task}\n"
                  + "".join(f"observation: {o}\n" for o in observations)
                  + "Your reply:")
        return self.generate(prompt)

    # hdl_bench interface: (system, user, i) -> [text] * n
    def hdl(self, system, user, i):
        return [self.generate(system + "\n" + user, max_new=384)]


# ---------------------------------------------------------------- components

def run_terminal(cand, out):
    tasks = tb.load_tasks()
    rows = tb.run_term_bench(tasks, cand.episode)
    (out / "terminal.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return ("term-bench-50-v1",
            canonical_sha256([{"id": t["id"], "instruction": t["instruction"]}
                              for t in tasks]))


def run_swe(cand, out):
    tasks = sb.load_tasks()
    rows = sb.run_swe_bench(tasks, cand.episode)
    (out / "swe.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return ("swe-bench-mini-v1",
            canonical_sha256([t["id"] for t in tasks]))


def run_long_horizon(cand, out):
    result = evaluate_policy(cand.policy, name="toy-candidate")
    (out / "long_horizon.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return ("long-horizon-5-v1",
            canonical_sha256([sc.id for sc in SCENARIOS]))


def run_tool_recovery(cand, out):
    """Model repairs the 7 corruption classes x 10 seeded samples.

    Correct = strict parse of the model's reply equals the original calls.
    The corpus generators come from the T99 sidecar so the task set is the
    same distribution the recovery unit tests fuzz."""
    import random
    from helioslm_v5.tests.test_tool_recovery import (
        REG, REPAIRABLE, _sample_calls)
    from helioslm_v5.agent.schema import ToolCallError, parse_tool_call
    rows = []
    rng = random.Random(20261123)
    for cls, (corrupt, _repair) in sorted(REPAIRABLE.items()):
        for i in range(10):
            calls = _sample_calls(rng)
            text = corrupt(calls)
            prompt = ("Repair the following malformed tool-call block into a "
                      "valid one. Reply with only the repaired block, nothing "
                      f"else.\n\n{text}\n")
            reply = cand.generate(prompt, max_new=256)
            try:
                got = parse_tool_call(reply, REG)
                ok = got == calls
            except ToolCallError:
                ok = False
            rows.append({"task": f"{cls}:{i}", "correct": bool(ok)})
    artifact = {"schema": "tool-recovery-run/1", "rows": rows}
    (out / "tool_recovery.json").write_text(
        json.dumps(artifact, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return ("tool-recovery-repairable-7x10-v1",
            canonical_sha256([r["task"] for r in rows]))


def _tarball_get(cache_dir):
    """Serve hdl_bench.fetch's raw.githubusercontent URLs from one codeload
    tarball of the pinned commit (raw endpoints are intermittent here)."""
    url = (f"https://codeload.github.com/{hb.UPSTREAM}/tar.gz/{hb.COMMIT}")
    data = urllib.request.urlopen(url, timeout=120).read()
    tmp = Path(cache_dir) / "_upstream.tar.gz"
    tmp.write_bytes(data)
    tf = tarfile.open(tmp)
    prefix = f"verilog-eval-{hb.COMMIT}/"

    def get(u):
        rel = u.split(f"/{hb.COMMIT}/", 1)[1]
        return tf.extractfile(prefix + rel).read()
    return get


def run_hdl(cand, out, cache):
    cache = Path(cache)
    if not (cache / "MANIFEST.json").is_file():
        cache.mkdir(parents=True, exist_ok=True)
        hb.fetch(cache, get=_tarball_get(cache))
    probs = hb.load(cache)
    result = hb.score(probs, cand.hdl, n=1, ks=(1,))
    (out / "hdl.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_sha = hashlib.sha256(
        (cache / "MANIFEST.json").read_bytes()).hexdigest()
    return (f"verilog-eval-v2/spec-to-rtl@{hb.COMMIT[:8]}", manifest_sha)


COMPONENTS = {
    "terminal": ("terminal.json", run_terminal),
    "swe": ("swe.json", run_swe),
    "long_horizon": ("long_horizon.json", run_long_horizon),
    "tool_recovery": ("tool_recovery.json", run_tool_recovery),
    "hdl": ("hdl.json", run_hdl),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--components", default=",".join(COMPONENTS),
                        help="comma-separated subset (default: all five)")
    parser.add_argument("--hdl-cache", default="data/hdl_bench",
                        help="hdl_bench fetch cache (default: data/hdl_bench)")
    parser.add_argument("--skip-adapters", action="store_true",
                        help="only (re)run components; do not regenerate "
                             "descriptors or the index")
    args = parser.parse_args(argv)

    results = args.results
    artifacts = results / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    selected = [c.strip() for c in args.components.split(",") if c.strip()]
    unknown = set(selected) - set(COMPONENTS)
    if unknown:
        parser.exit(2, f"unknown components: {sorted(unknown)}\n")

    cand = CharCandidate(args.checkpoint)
    print(f"candidate: {args.checkpoint} sha256={cand.checkpoint_sha256[:12]}…",
          flush=True)

    task_sets = {}
    existing = results / "task_sets.json"
    if existing.is_file():
        task_sets = json.loads(existing.read_text(encoding="utf-8"))
    for name in selected:
        artifact, runner = COMPONENTS[name]
        t0 = time.perf_counter()
        if name == "hdl":
            task_sets[name] = runner(cand, artifacts, args.hdl_cache)
        else:
            task_sets[name] = runner(cand, artifacts)
        print(f"[{name}] artifact written "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
    (results / "task_sets.json").write_text(
        json.dumps(task_sets, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")

    if args.skip_adapters:
        return 0
    if set(task_sets) != set(COMPONENTS):
        parser.exit(2, "all five components must be present before the "
                       "descriptors and index can be generated "
                       "(rerun without --components or --skip-adapters)\n")
    for name in COMPONENTS:
        task_set, task_set_sha = task_sets[name]
        generate_component(
            results, name, Path("artifacts") / COMPONENTS[name][0],
            model=cand.model_name,
            checkpoint_sha256=cand.checkpoint_sha256,
            task_set=task_set, task_set_sha256=task_set_sha,
            evidence_type="model_run", output=Path(f"{name}.json"))
        print(f"[{name}] descriptor written", flush=True)
    index_path = results / "agentic_index.json"
    index = generate_index(results, index_path)
    print(f"Agentic Index = {index['agentic_index']:.6f} "
          f"(is_model_score={index['is_model_score']}) -> {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

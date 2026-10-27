"""HDL benchmark harness: NVIDIA CVDP v1.1.0, non-agentic code generation
(v5.176, HDL track H0).

CVDP (nvidia/cvdp-benchmark-dataset; text CC-BY-4.0, code Apache-2.0)
datapoints carry a prompt, optional input files, the files the model must
write (``output.context``) and a cocotb harness (``harness.files``: a
docker-compose service, ``src/.env``, cocotb tests). Upstream builds a
docker image per datapoint. A RunPod pod is itself a container with no
docker, so this harness runs the same harness files directly: the repo
files go to ``<tmp>/code``, ``src/*`` to ``<tmp>/src``, the paths ``/code/``,
``/src/`` and ``/rundir/`` in the harness text are rewritten to the temp
dirs, and each compose service's ``command`` runs with its ``.env``. A
datapoint passes when every service exits 0 (upstream's rule).

Only the open-source flow is supported: the ``no_commercial`` file, and
datapoints whose Dockerfile is ``FROM __OSS_SIM_IMAGE__`` and whose
services use no other image. Code-comprehension categories (scored by an
LLM judge upstream) are skipped. Everything skipped is listed by id.

Upstream's simulator image has cocotb 2.0.1, pytest 8.3.2, Icarus
v13_0, Yosys 0.40 and Verilator 5.038; install the same on the pod.

Prompt and response parsing follow upstream ``dataset_processor`` and
``model_helpers``: one expected file -> plain text (first fenced block if
any); several -> JSON ``{"code": [{"<name>": "<code>"}]}``.

Generation and scoring are separate (``score`` takes texts), so the
simulations can run on CPU after the GPU time is over. Samples are
untrusted code: run them only inside a container.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from helioslm_v5.eval import hdl_bench as hb

SCHEMA_ID = "cvdp-bench/1"
DATASET = "nvidia/cvdp-benchmark-dataset"
FILE = "cvdp_v1.1.0_nonagentic_code_generation_no_commercial.jsonl"
CODE_GEN_CATEGORIES = (2, 3, 4, 5, 7, 12, 13, 14, 16)
RUN_TIMEOUT_S = 300
OSS_IMAGE = "__OSS_SIM_IMAGE__"

FOLDERS = """You are a helpful assistance.
Consider that you have a folder structure like the following:\n
    - rtl/*   : Contains files which are RTL code.
    - verif/* : Contains files which are used to verify the correctness of the RTL code.
    - docs/*  : Contains files used to document the project, like Block Guides, RTL Plans and Verification Plans.

When generating files, return the file name in the correct place at the folder structure.
"""
SCHEMA = ['{ "code": [{ "<name>" : "<code>"}] }', '{ "response": "<response>" }']
GUIDANCE = {
    2: "You are solving an 'RTL Code Completion' problem. To solve this problem correctly, you should only respond with the RTL code generated according to the requirements.",
    3: "You are solving a 'Specification to RTL Translation' problem. To solve this problem correctly, you should only respond with the RTL code translated from the specification.",
    4: "You are solving an 'RTL Code Modification' problem. To solve this problem correctly, you should only respond with the modified RTL code according to the requirements.",
    5: "You are solving a 'Specification to RTL Translation: Module Instantiation and Component Reuse' problem. To solve this problem correctly, you should only respond with the RTL code translated from the specification and with proper module instantiation and component reuse.",
    7: "You are solving an 'RTL Lint Improvement or Power-Performance Optimization' problem. To solve this problem correctly, you should only respond with improved RTL code to address lint issues or optimize for power/performance.",
    12: "You are solving a 'Test Plan to Testbench Stimulus Generation' problem. To solve this problem correctly, you should only respond with the testbench stimulus code generated based on the test plan specification.",
    13: "You are solving a 'Test Plan to Testbench Checker Generation' problem. To solve this problem correctly, you should only respond with the testbench checker code generated based on the test plan specification.",
    14: "You are solving a 'Test Plan to Assertions Generation' problem. To solve this problem correctly, you should only respond with the assertions for the testbench based on the test plan specification.",
    16: "You are solving an 'RTL Debugging and Bug Fixing' problem. To solve this problem correctly, you should only respond with the RTL code that is debugged and fixed to address the bug.",
}
_FENCE = re.compile(r"```(?:(\w+)\n)?(.*?)```", re.S)


class CvdpError(ValueError):
    """Raised for a malformed datapoint or a bad dataset file."""


# ------------------------------------------------------------------ data

def category(row):
    cats = row.get("categories") or []
    if not cats or not str(cats[0]).startswith("cid"):
        raise CvdpError(f"{row.get('id')}: no cid category")
    return int(str(cats[0])[3:])


def parse_compose(text):
    """Services of a harness docker-compose.yml: ``{name: {key: value}}``.

    Only the flat keys CVDP uses are read (image, build, env_file, command,
    environment as a list); anything nested is ignored.
    """
    services, cur, in_services, list_key = {}, None, False, None
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        body = line.strip()
        if indent == 0:
            in_services = body == "services:"
            cur = None
            continue
        if not in_services:
            continue
        if indent == 2 and body.endswith(":"):
            cur = services.setdefault(body[:-1].strip(), {})
            list_key = None
            continue
        if cur is None:
            continue
        if body.startswith("- ") and list_key:
            cur.setdefault(list_key, []).append(body[2:].strip())
            continue
        if indent == 4 and ":" in body:
            k, v = body.split(":", 1)
            k, v = k.strip(), v.strip()
            list_key = k if not v else None
            if v:
                cur[k] = v
    return services


def parse_env(text):
    env = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def unsupported(row):
    """Why a datapoint cannot run in the open-source flow, else None."""
    cat = category(row)
    if cat not in CODE_GEN_CATEGORIES:
        return f"category {cat} is not code generation"
    files = row["harness"]["files"]
    docker = files.get("Dockerfile")
    if docker is not None and docker.strip() != f"FROM {OSS_IMAGE}":
        return "custom Dockerfile"
    compose = files.get("docker-compose.yml")
    if not compose:
        return "no docker-compose.yml"
    services = parse_compose(compose)
    if not services:
        return "no services"
    for name, svc in services.items():
        if "image" in svc and svc["image"] != OSS_IMAGE:
            return f"service {name} uses image {svc['image']}"
        if "command" not in svc:
            return f"service {name} has no command"
    if not row["output"]["context"]:
        return "no output files"
    return None


def load(path):
    """Read a CVDP jsonl; returns ``(rows, skipped)`` with skip reasons."""
    path = Path(path)
    if not path.is_file():
        raise CvdpError(f"dataset not found: {path}")
    rows, skipped = [], {}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise CvdpError(f"{path.name}:{i + 1}: not JSON: {e}") from e
        for key in ("id", "input", "output", "harness"):
            if key not in row:
                raise CvdpError(f"{path.name}:{i + 1}: missing {key}")
        why = unsupported(row)
        if why:
            skipped[row["id"]] = why
        else:
            rows.append(row)
    return rows, skipped


def fetch(dest, revision=None):
    """Download the no_commercial file from the Hub (needs HF access)."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(DATASET, FILE, repo_type="dataset",
                           revision=revision, local_dir=dest)


# ------------------------------------------------------------------ prompt

def build_prompt(row):
    """``(system, user)`` as upstream builds them for a non-golden run."""
    files = list(row["output"]["context"])
    user = ""
    for name, text in row["input"]["context"].items():
        user += (f"\nConsider the following content for the file {name}:"
                 f"\n```\n{text}\n```")
    user += f"\nProvide me one answer for this request: {row['input']['prompt']}\n"
    system = FOLDERS + f"\n{GUIDANCE[category(row)]}\n"
    if len(files) == 1:
        user += ("Please provide your response as plain text without any JSON "
                 f"formatting. Your response will be saved directly to: {files[0]}.\n")
    else:
        user += f"Name the files as: {files}.\n"
        system += "\nProvide the response in one of the following JSON schemas: \n"
        system += "\nor\n".join(SCHEMA)
        system += ("\nThe response should be in JSON format, including "
                   "double-quotes around keys and values, and proper escaping "
                   "of quotes within values, and escaping of newlines.")
    return system, user


def _first_block(text):
    m = _FENCE.search(text)
    return m.group(2) if m else text


def parse_response(text, files):
    """Map a response to ``{file: content}`` (upstream rules); {} if bad."""
    if len(files) == 1:
        m = _FENCE.search(text)
        return {files[0]: m.group(2) if m else text.strip()}
    body = re.sub(r"[\x00-\x1F\x7F]", "", text.strip())
    start, end = body.find("{"), body[::-1].find("}")
    if start < 0:
        return {}
    body = body[start:-end if end else None]
    try:
        out = json.loads(body)
    except json.JSONDecodeError:
        return {}
    result = {}
    for item in out.get("code", []) if isinstance(out, dict) else []:
        if isinstance(item, dict):
            for name, code in item.items():
                if isinstance(code, str):
                    result[name] = _first_block(code)
    return result


# ------------------------------------------------------------------ running

_ABS = re.compile(r"(?<![\w./-])/(code|src|rundir)(?=/|\b)")


def _rewrite(text, root):
    """Point the harness's absolute /code, /src, /rundir at ``root``."""
    return _ABS.sub(lambda m: f"{root}/{m.group(1)}", text)


def safe_name(name):
    """A relative path that stays inside the repo dir (model-chosen names)."""
    parts = Path(name).parts
    return bool(parts) and not Path(name).is_absolute() and ".." not in parts


def run(row, outputs, timeout=RUN_TIMEOUT_S):
    """Run a datapoint's harness on ``outputs``; ``{"passed", "log"}``."""
    hb.need_tool("iverilog")
    harness = row["harness"]["files"]
    services = parse_compose(harness["docker-compose.yml"])
    with tempfile.TemporaryDirectory(prefix="cvdp_") as td:
        root = Path(td)
        for sub in ("code", "src", "rundir/harness"):
            (root / sub).mkdir(parents=True)
        for name, text in {**row["input"]["context"], **outputs}.items():
            if not safe_name(name):
                return {"passed": False, "log": f"refused path {name}"}
            p = root / "code" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        for name, text in harness.items():
            if name.startswith("src/"):
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(_rewrite(text, root), encoding="utf-8")
        log, ok = [], True
        for name, svc in services.items():
            env = dict(os.environ)
            env_file = svc.get("env_file", "").replace("./", "", 1)
            if env_file and (root / env_file).is_file():
                env.update(parse_env((root / env_file).read_text(encoding="utf-8")))
            for item in svc.get("environment", []):
                if "=" in item:
                    k, v = item.split("=", 1)
                    env[k.strip()] = _rewrite(v.strip(), root)
            cmd = _rewrite(svc["command"], root)
            try:
                r = subprocess.run(cmd, shell=True, cwd=root / "code", env=env,
                                   capture_output=True, text=True, timeout=timeout)
                log.append(f"[{name}] exit {r.returncode}\n{r.stdout[-3000:]}{r.stderr[-1000:]}")
                ok = ok and r.returncode == 0
            except subprocess.TimeoutExpired:
                log.append(f"[{name}] TIMEOUT")
                ok = False
    return {"passed": ok, "log": "\n".join(log)[-6000:]}


def score(rows, texts, ks=(1,), workers=4, timeout=RUN_TIMEOUT_S):
    """Score ``texts[id] -> [response, ...]`` (same n for every id)."""
    ns = {len(texts[r["id"]]) for r in rows}
    if len(ns) != 1:
        raise CvdpError(f"uneven sample counts: {sorted(ns)}")
    n = ns.pop()
    jobs = [(r, t) for r in rows for t in texts[r["id"]]]

    def one(job):
        r, t = job
        outs = parse_response(t, list(r["output"]["context"]))
        if not outs:
            return r["id"], False
        return r["id"], run(r, outs, timeout)["passed"]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, jobs))
    correct = {r["id"]: 0 for r in rows}
    for rid, ok in results:
        correct[rid] += ok
    by_cat = {}
    for r in rows:
        by_cat.setdefault(category(r), []).append(correct[r["id"]])
    out = {"schema": SCHEMA_ID, "bench": "cvdp-v1.1.0/nonagentic-codegen-oss",
           "n_problems": len(rows), "n": n,
           "problems": [{"id": r["id"], "category": category(r),
                         "correct": correct[r["id"]]} for r in rows]}
    m = max(len(rows), 1)
    for k in ks:
        if k <= n:
            out[f"pass@{k}"] = sum(hb.pass_at_k(n, c, k) for c in correct.values()) / m
            out[f"pass@{k}_by_category"] = {
                f"cid{c:03d}": sum(hb.pass_at_k(n, x, k) for x in v) / len(v)
                for c, v in sorted(by_cat.items())}
    return out


def self_check(rows, workers=4):
    """Run the golden outputs (needs a ``*_with_solutions`` file)."""
    def one(r):
        res = run(r, r["output"]["context"])
        return r["id"], res["passed"]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(one, rows))
    fails = [rid for rid, ok in res if not ok]
    return len(rows) - len(fails), fails


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("fetch", "self-check", "list"))
    ap.add_argument("--data", default="data/hdl_bench/cvdp")
    ap.add_argument("--file", help="jsonl (default: <data>/" + FILE + ")")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        print(fetch(a.data))
        return 0
    rows, skipped = load(a.file or Path(a.data) / FILE)
    print(f"{len(rows)} datapoints runnable, {len(skipped)} skipped")
    if a.cmd == "list":
        for rid, why in sorted(skipped.items()):
            print(f"  skip {rid}: {why}")
        return 0
    if not shutil.which("pytest"):
        raise CvdpError("pytest not on PATH (pip install cocotb==2.0.1 pytest==8.3.2)")
    ok, fails = self_check(rows, a.workers)
    print(f"self-check: {ok}/{len(rows)} golden outputs pass")
    for rid in fails:
        print(f"  FAIL {rid}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())

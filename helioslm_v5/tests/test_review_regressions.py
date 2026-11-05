"""Regression oracles for the project review's P1/P2 findings."""
import fnmatch
import json
import shlex
import subprocess
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

try:
    import pytest
except ImportError:
    # v5.240: this file is also executed directly by the daily TESTS
    # suite (case_upstream_suites), where pytest is not installed. The
    # shim covers exactly the one pytest feature used below:
    # mark.parametrize (collection-time only; a no-op when the functions
    # are called directly with explicit arguments).
    class _Mark:
        @staticmethod
        def parametrize(*_args, **_kwargs):
            return lambda f: f

    class _PytestShim:
        mark = _Mark()

    pytest = _PytestShim()

import torch
import torch.nn.functional as F

from helioslm_v5.harness.web_ui_v2 import SessionStore
from helioslm_v5.hsh.session import Session
from helioslm_v5.src.inference.prefix_pool import PrefixPool
from helioslm_v5.src.training.pretrain import mixed_kd_loss, pilot_config, pretrain
from examples import cpt_hf


class _Source:
    def batch(self, batch, seq, *args):
        return (torch.ones(batch, seq, dtype=torch.long),
                torch.zeros(batch, seq, dtype=torch.long))


class _TrainLM(torch.nn.Module):
    def __init__(self, bad_loss=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))
        self.bad_loss = bad_loss

    def forward(self, input_ids):
        logits = self.weight.expand(*input_ids.shape, 3)
        if self.bad_loss:
            logits = logits * float("nan")
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("fault", ["loss", "nan_grad", "inf_grad", "remote"])
def test_cpt_rejects_batch_before_optimizer(fault, monkeypatch):
    model = _TrainLM(bad_loss=fault == "loss")
    before = model.weight.detach().clone()
    if fault.endswith("grad"):
        value = float("nan") if fault == "nan_grad" else float("inf")
        model.weight.register_hook(lambda g: torch.full_like(g, value))
    if fault == "remote":
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

        def remote_failure(flags, **kw):
            flags[0] = 1

        monkeypatch.setattr(torch.distributed, "all_reduce", remote_failure)

    def forbidden_step(*args, **kw):
        raise AssertionError("invalid batch reached optimizer.step")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_step)
    result = cpt_hf.train(
        model, {"main": _Source(), "decay": None}, steps=2,
        micro_batch=1, seq=2, grad_accum=2, lr=0.01, warmup=0,
        world=2 if fault == "remote" else 1, autocast=False)
    assert result["stopped"] == (
        "non_finite_loss" if fault in ("loss", "remote") else "non_finite_grad")
    assert result["steps"] == result["tokens"] == 0
    assert torch.equal(model.weight, before)
    assert model.weight.grad is None


def _pretrain_stub(fault):
    """HeliosLMv5 stand-in. ``fault`` is None, ``loss``, ``nan_grad`` or
    ``inf_grad``."""

    class Model(_TrainLM):
        def __init__(self, config):
            super().__init__(bad_loss=False)
            self.config = config
            if fault in ("nan_grad", "inf_grad"):
                value = float("nan") if fault == "nan_grad" else float("inf")
                self.weight.register_hook(
                    lambda g: torch.full_like(g, value))

        def forward(self, x):
            logits = super().forward(x).logits
            if fault == "loss":
                logits = logits * float("nan")
            return logits, None, None

    return Model


class _PretrainShard(_Source):
    def eval_windows(self, seq, max_windows):
        yield self.batch(1, seq)


@pytest.mark.parametrize("fault", ["loss", "nan_grad", "inf_grad"])
def test_pretrain_rejects_non_finite_before_step(fault, tmp_path, monkeypatch):
    """A non-finite loss or grad does not step, does not rewrite the
    checkpoint, and is reported. Matches cpt_hf's stop contract."""
    from helioslm_v5.src.training import pretrain as training

    live = []

    class Recording(_pretrain_stub(fault)):
        def __init__(self, config):
            super().__init__(config)
            live.append(self)

    monkeypatch.setattr(training, "HeliosLMv5", _pretrain_stub(None))
    cfg = pilot_config("dense", vocab_size=3)
    kw = dict(batch=2, seq=2, grad_accum=2, lr=0.01, warmup=0,
              eval_every=10, eval_windows=1, ckpt_every=1, seed=0,
              log=lambda *args: None)
    out = tmp_path / "run"
    pretrain(cfg, _PretrainShard(), _PretrainShard(), out, steps=1, **kw)
    before = torch.load(out / "ckpt.pt", weights_only=False)
    before_model = {k: v.clone() for k, v in before["model"].items()}
    assert before["step"] == 1
    assert not torch.equal(before_model["weight"], _TrainLM().weight)

    def forbidden_step(*args, **kwargs):
        raise AssertionError("non-finite batch reached optimizer.step")

    monkeypatch.setattr(training, "HeliosLMv5", Recording)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_step)
    result = pretrain(cfg, _PretrainShard(), _PretrainShard(), out,
                      steps=4, **kw)
    assert result["stopped"] == (
        "non_finite_loss" if fault == "loss" else "non_finite_grad")
    assert result["steps"] == 1
    assert result["tokens"] == 2 * 2 * 2
    assert result["last"]["stopped"] == result["stopped"]
    after = torch.load(out / "ckpt.pt", weights_only=False)
    assert after["step"] == 1
    for k, v in before_model.items():
        assert torch.equal(after["model"][k], v), k
    assert len(live) == 1
    assert torch.equal(live[0].weight.detach(), before_model["weight"])
    assert live[0].weight.grad is None
    rows = [json.loads(ln) for ln in (out / "log.jsonl").read_text().splitlines()
            if ln]
    assert rows[-1]["stopped"] == result["stopped"]
    assert rows[-1]["step"] == 2
    assert all("stopped" not in row for row in rows[:-1])
    if fault == "loss":
        assert rows[-1]["train_loss"] is None
    else:
        assert rows[-1]["train_loss"] is not None
        assert rows[-1]["grad_norm"] is None


class _PrefixLM:
    """Next token depends on the whole cached prefix, not just its tail."""
    def eval(self):
        return self

    def __call__(self, input_ids, past_key_values=None, **kw):
        previous = 0 if past_key_values is None else past_key_values[0][0]
        sums = input_ids.cumsum(1) + previous
        logits = F.one_hot(sums % 11, num_classes=11).float()
        return logits, None, [(sums[:, -1:].clone(),)]


@pytest.mark.parametrize("capacity", [0, 1, 2])
@pytest.mark.parametrize("length", [6, 7])
def test_prefix_eviction_does_not_break_generation(capacity, length):
    ids = list(range(1, length + 1))
    expected = list(ids)
    for _ in range(4):
        expected.append(sum(expected) % 11)
    pool = PrefixPool(_PrefixLM(), block_size=2, max_blocks=capacity)
    for _ in range(2):
        assert pool.generate(ids, max_new_tokens=4).tolist() == [expected]
        assert len(pool._pool) <= capacity
    # Public store() still returns the final complete block's logits.
    assert pool.store(ids).shape == (11,)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_mixed_kd_preserves_unaligned_and_masked_tokens(alpha):
    logits = torch.tensor([[[0.2, -0.1, 0.5], [0.4, 0.2, -0.3],
                            [0.1, 0.8, -0.2]]], requires_grad=True)
    y = torch.tensor([[1, 2, 0]])
    idx = torch.tensor([[[0], [0], [1]]])
    p = torch.tensor([[[1.0], [0.0], [1.0]]])
    mask = torch.tensor([[1, 1, 0]])
    loss, _ = mixed_kd_loss(logits, y, mask, idx, p, alpha)
    loss.backward()
    actual_grad = logits.grad.clone()
    reference = logits.detach().clone().requires_grad_()
    logp = F.log_softmax(reference, -1)
    expected = (-(1 - alpha) * logp[0, 0, 1]
                - alpha * logp[0, 0, 0] - logp[0, 1, 2]) / 2
    expected.backward()
    assert torch.allclose(loss, expected)
    assert torch.allclose(actual_grad, reference.grad)
    assert torch.equal(actual_grad[0, 2], torch.zeros(3))
    empty = torch.zeros_like(p)
    plain, kd = mixed_kd_loss(logits, y, mask, idx, empty, alpha)
    expected_ce = F.cross_entropy(logits[0, :2], y[0, :2])
    assert torch.allclose(plain, expected_ce) and kd.item() == 0


def test_pretrain_without_teacher_targets_matches_plain_ce(tmp_path, monkeypatch):
    from helioslm_v5.src.training import pretrain as training

    class Model(_TrainLM):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def forward(self, x):
            return super().forward(x).logits, None, None

    class Shard(_Source):
        def __init__(self, kd=False):
            self.kd = kd

        def batch(self, *args):
            x, y = super().batch(*args)
            if self.kd:
                return (x, y, torch.ones_like(y),
                        torch.zeros(*y.shape, 1, dtype=torch.long),
                        torch.zeros(*y.shape, 1))
            return x, y

        def eval_windows(self, seq, max_windows):
            yield self.batch(1, seq)

    monkeypatch.setattr(training, "HeliosLMv5", Model)
    cfg = pilot_config("dense", vocab_size=3)
    kw = dict(steps=3, batch=2, seq=2, grad_accum=2, lr=0.01,
              warmup=0, eval_every=3, eval_windows=1, ckpt_every=3,
              log=lambda *args: None)
    pretrain(cfg, Shard(), Shard(), tmp_path / "plain", **kw)
    pretrain(cfg, Shard(kd=True), Shard(), tmp_path / "kd",
             kd_alpha=1.0, **kw)
    plain = torch.load(tmp_path / "plain/ckpt.pt", weights_only=False)
    kd = torch.load(tmp_path / "kd/ckpt.pt", weights_only=False)
    assert torch.equal(plain["model"]["weight"], kd["model"]["weight"])
    assert not torch.equal(kd["model"]["weight"], _TrainLM().weight)


@pytest.mark.parametrize("second", ["append", "rename", "params"])
def test_session_store_concurrent_updates_are_atomic(tmp_path, second):
    barrier = threading.Barrier(2)

    class RacingStore(SessionStore):
        def get(self, sid):
            snapshot = super().get(sid)
            # Old code lets both threads read the old snapshot. Under the
            # transaction lock the first reader times out; the second then
            # sees the committed snapshot. No sleep/scheduler luck required.
            try:
                barrier.wait(timeout=1)
            except threading.BrokenBarrierError:
                pass
            return snapshot

    store = RacingStore(str(tmp_path))
    sid = store.create()["id"]
    first_msg = {"role": "user", "content": "A"}
    with ThreadPoolExecutor(max_workers=2) as workers:
        a = workers.submit(store.append_messages, sid, first_msg)
        if second == "append":
            b = workers.submit(store.append_messages, sid,
                               {"role": "assistant", "content": "B"})
        elif second == "rename":
            b = workers.submit(store.update, sid, title="renamed")
        else:
            b = workers.submit(store.update, sid, params={"max_tokens": 64})
        a.result(timeout=5)
        b.result(timeout=5)
    saved = SessionStore.get(store, sid)
    assert first_msg in saved["messages"]
    if second == "append":
        assert {m["content"] for m in saved["messages"]} == {"A", "B"}
    elif second == "rename":
        assert saved["title"] == "renamed"
    else:
        assert saved["params"]["max_tokens"] == 64


def test_session_append_and_each_listener_receive_detached_events():
    session = Session()
    seen = []

    def mutate(event):
        event["seq"] = 99
        event["data"]["nested"]["text"] = "rewritten"

    session.subscribe(mutate)
    session.subscribe(seen.append)
    returned = session.append("user/message", {"nested": {"text": "original"}})
    returned["data"]["nested"]["text"] = "changed by caller"
    assert seen[0]["seq"] == session.events[0]["seq"] == 0
    assert seen[0]["data"]["nested"]["text"] == "original"
    assert session.events[0]["data"]["nested"]["text"] == "original"
    assert Session.from_jsonl(session.to_jsonl()).events == session.events


def test_release_selection_imports(tmp_path):
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    command = workflow.split("zip -r helioslm-v5-release.zip", 1)[1]
    command = command.split("\n      - ", 1)[0].replace("\\\n", " ")
    args = shlex.split(command)
    split = args.index("-x")
    includes, excludes = args[:split], args[split + 1:]
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for entry in includes:
            path = root / entry
            for file in path.rglob("*") if path.is_dir() else [path]:
                name = file.relative_to(root).as_posix()
                if file.is_file() and not any(
                        fnmatch.fnmatch(name, pattern) for pattern in excludes):
                    z.write(file, name)
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as z:
        assert "helioslm_v5/configs/config_v5.py" in z.namelist()
        z.extractall(unpacked)
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from helioslm_v5.configs.config_v5 import HeliosLMv5Config; "
        "assert HeliosLMv5Config(size='lite').hidden_size > 0")
    proc = subprocess.run([sys.executable, "-I", "-c", code, str(unpacked)],
                          cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr

"""Disk-tier expert store with bit-exact streaming (v5.15, roadmap #6).

Frontier MoE weights are far larger than RAM, but each token only touches a
few experts — so routed experts are *data to be staged*, not state to be
resident (the colibri insight). This module offloads routed-expert weights
of a ``DeviceLimitedMoE`` to a single memory-mapped file and pages experts
in on routing demand with an LRU resident set bounded by ``budget_bytes``.

Correctness contract: weights are stored **raw, dtype and bits preserved**,
and restored verbatim into the same module objects, so the streaming
forward runs the *identical* GEMMs as the dense path — bit-exact under
fixed-shape expert blocks (the MoE dispatch already pads to fixed blocks
for bit-reproducibility). ``verify_streaming()`` asserts exactly this,
turning "did tiering change the answer?" into a test.

Router and shared experts stay resident (always hot, tiny — the same
division colibri uses); only the routed ``moe.experts`` are tiered.

Usage:
    dense = moe(x)
    store = attach_streaming_store(moe, "/tmp/experts.bin", budget_bytes=1<<20)
    stream = moe(x)                      # experts paged in on demand
    assert torch.equal(dense, stream)    # oracle
    ...
    detach_streaming_store(moe, store)   # restore resident weights

Limitations (honest): attach/detach are eval-mode operations (params are
stripped/reloaded, so don't attach mid-training or mid-optimizer-state);
expert modules are unregistered from ``state_dict`` while attached — detach
before saving checkpoints. Concurrency: single-process; the file is opened
read-only. ``detach_streaming_store`` temporarily expands the resident
budget to load every expert back without LRU eviction.
"""

import json
import os
import tempfile
from collections import OrderedDict

import numpy as np
import torch

__all__ = ["DiskExpertStore", "attach_streaming_store",
           "detach_streaming_store", "verify_streaming"]

_NP_DTYPE = {
    "torch.float32": "float32", "torch.float64": "float64",
    "torch.float16": "float16",
    "torch.int64": "int64", "torch.int32": "int32",
    "torch.uint8": "uint8", "torch.bool": "bool",
}


class DiskExpertStore:
    """Memory-mapped disk tier for routed experts.

    File layout: JSON header (index of tensor offsets/dtypes/shapes) +
    raw little-endian tensor bytes. The numpy memmap provides zero-copy
    slices; materialization copies into RAM, so correctness never depends
    on mmap aliasing. bf16 is stored as its uint16 bit pattern (numpy has
    no bf16) and reinterpreted on load.
    """

    def __init__(self, moe, path, budget_bytes):
        self.path = path
        # Set only when attach_streaming_store created the backing file
        # itself (path=None): the store then OWNS the file and unlinks it
        # on close. A caller-supplied path is never touched.
        self._owns_file = False
        self.budget_bytes = int(budget_bytes)
        self.experts = list(moe.experts)          # the real modules
        self.num_experts = len(self.experts)

        # ---- serialize all expert weights --------------------------
        index = {}   # eid -> {param_name: [offset, numel, dtype, shape]}
        blobs = []
        offset = 0
        for eid, mod in enumerate(self.experts):
            index[str(eid)] = {}
            for name, p in mod.state_dict().items():
                t = p.detach().cpu().contiguous()
                if t.dtype == torch.bfloat16:
                    raw = t.view(torch.uint16).numpy().tobytes(order="C")
                else:
                    raw = t.numpy().tobytes(order="C")
                index[str(eid)][name] = [offset, t.numel(), str(t.dtype),
                                         list(t.shape)]
                blobs.append(raw)
                offset += len(raw)
        self.total_bytes = offset

        header = json.dumps({"index": index}).encode("utf-8")
        with open(path, "wb") as f:
            f.write(len(header).to_bytes(8, "little"))
            f.write(header)
            for b in blobs:
                f.write(b)

        self._index = index
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")
        self._header_end = 8 + len(header)

        # ---- LRU resident set --------------------------------------
        self._resident = OrderedDict()   # eid -> bytes
        self.bytes_resident = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _torch_dtype(name):
        return getattr(torch, name.replace("torch.", ""))

    def _expert_bytes(self, eid):
        return sum(meta[1] * self._torch_dtype(meta[2]).itemsize
                   for meta in self._index[str(eid)].values())

    def _load_tensor(self, eid, name):
        off, numel, dtype, shape = self._index[str(eid)][name]
        dt = self._torch_dtype(dtype)
        start = self._header_end + off
        nbytes = numel * dt.itemsize
        raw = np.frombuffer(self._mm[start: start + nbytes].tobytes(),
                            dtype=np.uint8)
        if dt == torch.bfloat16:
            u16 = torch.from_numpy(raw.view(np.uint16).copy()).reshape(shape)
            return u16.view(torch.bfloat16)
        return torch.from_numpy(raw.view(_NP_DTYPE[str(dt)]).copy()) \
                   .reshape(shape)

    def _strip(self, eid):
        """Free an expert's RAM (weights remain on disk)."""
        mod = self.experts[eid]
        for p in mod.parameters():
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)

    def materialize(self, eid):
        """Ensure expert ``eid``'s weights are resident; evict LRU as needed."""
        if eid in self._resident:
            self.hits += 1
            self._resident.move_to_end(eid)
            return
        self.misses += 1
        need = self._expert_bytes(eid)
        while self._resident and self.bytes_resident + need > self.budget_bytes:
            old, _ = self._resident.popitem(last=False)
            self._strip(old)
            self.bytes_resident -= self._expert_bytes(old)
            self.evictions += 1
        mod = self.experts[eid]
        state = {name: self._load_tensor(eid, name)
                 for name in mod.state_dict()}
        # direct data assignment (params may be stripped to empty(0) —
        # load_state_dict would reject the shape)
        with torch.no_grad():
            for name, p in mod.named_parameters():
                p.data = state[name].to(dtype=p.dtype)
        self._resident[eid] = need
        self.bytes_resident += need

    def stats(self):
        reqs = self.hits + self.misses
        return {
            "hits": self.hits, "misses": self.misses,
            "hit_rate": self.hits / max(reqs, 1),
            "evictions": self.evictions,
            "bytes_resident": self.bytes_resident,
            "bytes_total": self.total_bytes,
            "ram_saving_vs_dense": 1.0 - self.bytes_resident / max(self.total_bytes, 1),
        }

    def close(self):
        if self._mm is not None:
            self._mm._mmap.close()
            self._mm = None
        if self._owns_file and self.path is not None:
            # We created this temp file (path=None at attach): remove it
            # or every attach leaks total_bytes in the system temp dir.
            # close() must stay idempotent, so clear the path after the
            # unlink (and swallow a raced/cleared file silently).
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            self.path = None


class _StreamingExpertList:
    """Duck-type replacement for ``moe.experts`` (nn.ModuleList).

    Indexing materializes the expert from the store, then returns the real
    module — so ``moe.forward`` runs unchanged.
    """

    def __init__(self, store):
        self.store = store

    def __getitem__(self, eid):
        self.store.materialize(eid)
        return self.store.experts[eid]

    def __len__(self):
        return self.store.num_experts

    def __iter__(self):
        for eid in range(len(self)):
            yield self[eid]


def attach_streaming_store(moe, path=None, budget_bytes=1 << 30):
    """Offload ``moe.experts`` to a disk tier and attach the streaming list.

    Returns the ``DiskExpertStore``. Router/shared weights stay resident.
    Eval mode only; see module docstring for the contract. When ``path``
    is None a temp file is created that the store OWNS (unlinked on
    ``close``/detach); a caller-supplied path is never removed.
    """
    if isinstance(moe.experts, _StreamingExpertList):
        raise RuntimeError(
            "attach_streaming_store: this MoE's experts are already "
            "managed by a streaming store — call detach_streaming_store "
            "first. Attaching a second store would serialize experts "
            "while the first store evicts them (writing empty(0) "
            "tensors) and leave stale LRU bookkeeping behind."
        )
    owns_file = False
    if path is None:
        fd, path = tempfile.mkstemp(prefix="helioslm_experts_", suffix=".bin")
        os.close(fd)
        owns_file = True
    store = DiskExpertStore(moe, path, budget_bytes)
    store._owns_file = owns_file
    for eid in range(store.num_experts):
        store._strip(eid)
    store._original_experts = moe.experts
    if "experts" in moe._modules:
        del moe._modules["experts"]              # unregister from module tree
    object.__setattr__(moe, "experts", _StreamingExpertList(store))
    return store


def detach_streaming_store(moe, store):
    """Restore resident weights and the original nn.ModuleList.

    Loads every expert back WITHOUT LRU eviction (budget temporarily
    expanded to the full serialized size — detach is not the hot path).
    """
    total = sum(store._expert_bytes(e) for e in range(store.num_experts))
    store.budget_bytes = max(store.budget_bytes, total)
    for eid in range(store.num_experts):
        store.materialize(eid)
    object.__setattr__(moe, "experts", store._original_experts)
    moe._modules["experts"] = store._original_experts   # re-register submodule
    store.close()


@torch.no_grad()
def verify_streaming(moe, hidden_states, budget_bytes=1 << 20, path=None):
    """Oracle: dense forward == streaming forward (bit-exact expected).

    Returns (dense_out, stream_out, report_dict).
    """
    moe.eval()
    dense = moe(hidden_states)
    store = attach_streaming_store(moe, path=path, budget_bytes=budget_bytes)
    try:
        stream = moe(hidden_states)
    finally:
        detach_streaming_store(moe, store)
    report = store.stats()
    report["max_abs_diff"] = (dense - stream).abs().max().item()
    report["bit_exact"] = report["max_abs_diff"] == 0.0
    return dense, stream, report

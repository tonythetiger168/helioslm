"""vLLM-style inference engine + GPU monitoring.

Engine architecture (M7/M25 dual-track design):
  - ``BlockManager`` is the paged-memory *accounting* layer: it owns block
    tables, per-block refcounts and copy-on-write bookkeeping per request
    (allocate on admission, append per generated token, free on finish).
  - The model's own per-layer ``past_key_values`` (returned by
    ``HeliosLMv5.forward``) carries the actual K/V tensors used by
    attention. The engine feeds only the *last* new token per running
    request each step — the full prefix is never re-concatenated.
  This keeps the engine model-agnostic about the internal cache layout
  while preserving the paged allocation/CoW semantics.

GPU monitoring is plain in-memory metric lists; Prometheus export is used
only when the optional ``prometheus_client`` package is installed.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

try:
    from helioslm_v5.src.inference.paged_attention import BlockManager
except ImportError:  # pragma: no cover - package-relative fallback
    from .paged_attention import BlockManager


@dataclass
class Request:
    """A single generation request (M26)."""
    request_id: int
    prompt_token_ids: List[int]
    generated_token_ids: List[int] = field(default_factory=list)
    max_new_tokens: int = 32
    temperature: float = 0.0
    eos_token_id: Optional[int] = None
    # runtime state (engine-internal)
    past_key_values: Optional[object] = None
    finished: bool = False
    # O2 (v5.4): set when the request failed mid-step (forward exception);
    # the request is retired and its blocks released instead of leaking.
    error: Optional[str] = None
    # Batched-stepping state (v5.2): ``prompt_pad`` left pad tokens were
    # prepended at prefill so that this request's stored cache length
    # (``cache_len``) matches the admission watermark; pad K/V entries sit at
    # the FRONT of the cache and are masked out of every batched decode.
    cache_len: int = 0
    prompt_pad: int = 0

    def is_done(self) -> bool:
        if self.finished:
            return True
        if len(self.generated_token_ids) >= self.max_new_tokens:
            return True
        if (self.eos_token_id is not None and self.generated_token_ids
                and self.generated_token_ids[-1] == self.eos_token_id):
            return True
        return False

    def get_input_ids(self) -> List[int]:
        """Full id sequence (prompt + generated)."""
        return self.prompt_token_ids + self.generated_token_ids

    def append_token(self, token_id: int):
        self.generated_token_ids.append(int(token_id))


class VLLMEngine:
    """
    vLLM-style inference engine with paged block accounting.

    Features:
      - Continuous batching (schedule/step over Request objects)
      - Paged KV-cache accounting with CoW (BlockManager)
      - Incremental decoding via the model's past_key_values
      - Batched stepping (v5.2): decode-phase requests that only need to feed
        ONE new token are merged into a single [B, 1] forward per step.
      - Batched prefill (v5.12): newly admitted requests are grouped by
        (prompt length, prompt_pad); equal-length unpadded rows share one
        [B, L] forward (exact — no mask, no position shift). Rows with a
        watermark pad prefix (prompt_pad > 0) cannot share one unmasked
        forward — each keeps the solo masked ``_prefill`` path. For
        recurrent-state (hybrid) models — which cannot use watermark pad
        prefixes — this grouping is the only batched prefill path;
        singleton groups keep the solo forward.

    Batched-decode alignment strategy (v5.2, watermark padding):
      The attention stack uses ``position_ids`` for BOTH RoPE and the causal
      mask, so ragged caches cannot be padded into one batch exactly through
      the public forward contract. Instead the engine keeps every running
      request's stored cache at a common *watermark* length: at admission a
      request's prefill is run on top of a prefix of
      ``watermark - len(prompt)`` pad-token cache entries (attention_mask=0
      — the pad K/V entries are masked out of every step, so they are
      exactly neutral), giving it ``cache_len == watermark``. All rows admitted under the same watermark then decode in
      lockstep (+1 token/step each) and share one batched forward per step
      with a per-row attention_mask covering the pad region. A request whose
      prompt is longer than the current watermark starts a new, larger
      watermark; older shorter rows then form a separate cache-length group
      and are decoded by a separate forward (correct, just less batched —
      they can never catch up because every row grows by 1 token per step).
      Requests with prompt_pad == 0 take the un-masked fast path, which is
      bitwise identical to a solo forward.
    """

    def __init__(self, model, config, block_size: Optional[int] = None,
                 max_num_blocks: Optional[int] = None,
                 max_batch_size: Optional[int] = None):
        self.model = model
        self.config = config
        # O5 (v5.4): block parameters default to config.paged_attention
        # (when the field exists); explicit constructor arguments win.
        # ``enabled`` is read for the record — the BlockManager is an
        # accounting-only ledger here (the model owns the real K/V), so
        # the engine keeps it active either way and there is no
        # non-paged decode mode to switch to.
        pa_cfg = getattr(config, "paged_attention", None)
        if block_size is None:
            block_size = getattr(pa_cfg, "block_size", 16)
        if max_num_blocks is None:
            max_num_blocks = getattr(pa_cfg, "num_blocks", 10000)
        self.paged_attention_enabled = bool(
            getattr(pa_cfg, "enabled", True))
        self.block_size = block_size

        device = next(model.parameters()).device
        # v5.5 (hybrid models): recurrent-state caches (GatedDeltaAttention,
        # marked via attention.is_recurrent_attention) CANNOT absorb the
        # neutral pad-token prefix used for watermark alignment — pad tokens
        # are folded into the fixed-size state multiplicatively and are not
        # maskable afterwards. For such models the engine falls back to
        # unpadded prefills; requests then group by their natural cache
        # length (correct, just less batched across unequal prompts).
        self._has_recurrent_state = any(
            getattr(getattr(layer, "attention", None),
                    "is_recurrent_attention", False)
            for layer in getattr(model, "layers", [])
        )
        self.block_manager = BlockManager(
            block_size=block_size,
            num_blocks=max_num_blocks,
            device=str(device),
        )
        # The BlockManager is used for block accounting / CoW only (the
        # model owns the real K/V in past_key_values), so cache tensors are
        # allocated with a minimal (1, 1) per-block shape to avoid holding a
        # second full-size KV pool.
        self._bm_num_heads = 1
        self._bm_head_dim = 1

        # O5 (v5.4): config.batch_size was removed from the config; the
        # engine batch cap defaults to 32 unless explicitly overridden.
        self.max_batch_size = max_batch_size if max_batch_size is not None else 32
        self._pad_token_id = getattr(config, "pad_token_id", 0)
        # Memoized per-length caches of pad-token prefixes (engine-internal;
        # see _prefill). Entries are finite garbage that every later
        # attention_mask permanently masks out.
        # F4: the memo is LRU-bounded — each entry is a full per-length KV
        # cache, so unbounded growth would pin memory in long-running
        # serving. Eviction is safe: running requests hold their own
        # sliced views, which keep the underlying storage alive via
        # refcount until the request finishes.
        self._pad_caches: Dict[int, object] = {}
        self._pad_caches_max: int = 64

        self.request_queue: List[Request] = []
        self.running_requests: List[Request] = []
        self.finished_requests: List[Request] = []
        self._next_request_id = 0

    # ------------------------------------------------------------------
    # request management
    # ------------------------------------------------------------------
    def add_request(self, prompt_token_ids: List[int], max_new_tokens: int = 32,
                    temperature: float = 0.0,
                    eos_token_id: Optional[int] = None) -> int:
        """Submit a new request; returns its request_id."""
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)
        req = Request(
            request_id=self._next_request_id,
            prompt_token_ids=[int(t) for t in prompt_token_ids],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=eos_token_id,
        )
        self._next_request_id += 1
        self.request_queue.append(req)
        return req.request_id

    def schedule(self):
        """Admit queued requests up to max_batch_size; retire finished ones."""
        self.running_requests = [r for r in self.running_requests if not r.is_done()]
        k = min(len(self.request_queue),
                self.max_batch_size - len(self.running_requests))
        if k <= 0:
            return
        # Watermark for this admission wave: every admitted request is
        # left-padded at prefill up to W = max(running cache lengths,
        # longest prompt in the wave), so same-watermark rows share one
        # cache length and can decode in a single batched forward.
        watermark = max([r.cache_len for r in self.running_requests]
                        + [len(self.request_queue[j].prompt_token_ids)
                           for j in range(k)])
        for _ in range(k):
            req = self.request_queue.pop(0)
            # Hybrid (recurrent-state) models: no pad prefix (see __init__).
            req.prompt_pad = 0 if self._has_recurrent_state else \
                watermark - len(req.prompt_token_ids)
            # Paged accounting (M25/M-I3): allocate blocks covering the
            # ACTUAL stored cache length (prompt_pad + prompt), not just
            # the prompt — the pad-prefix K/V occupies real memory, so
            # the block ledger must charge for it or OOM protection
            # drifts from reality as the watermark grows.
            self.block_manager.allocate(
                req.request_id, req.prompt_pad + len(req.prompt_token_ids),
                self._bm_num_heads, self._bm_head_dim,
            )
            self.running_requests.append(req)

    def _finish(self, req: Request, error: Optional[BaseException] = None):
        """Retire a request: mark finished, drop its KV cache and release
        its paged blocks. ``error`` (O2) records a mid-step failure on the
        request so callers can inspect why it terminated early."""
        req.finished = True
        req.past_key_values = None
        if error is not None:
            req.error = f"{type(error).__name__}: {error}"
        self.block_manager.free(req.request_id)  # release paged blocks
        if req in self.running_requests:
            self.running_requests.remove(req)
        self.finished_requests.append(req)

    # ------------------------------------------------------------------
    # decoding
    # ------------------------------------------------------------------
    @staticmethod
    def _batch_past(pasts):
        """Concatenate equal-length per-request caches along the batch dim
        (contract: per-layer tuples of tensors, sequence axis at dim 2)."""
        first = pasts[0]
        batched = []
        for li, layer in enumerate(first):
            entries = []
            for j, t in enumerate(layer):
                if torch.is_tensor(t):
                    entries.append(torch.cat([p[li][j] for p in pasts], dim=0))
                else:
                    entries.append(t)
            batched.append(tuple(entries))
        return batched

    @staticmethod
    def _row_past(past, row: int):
        """Extract one batch row from a batched cache (dim-2 intact)."""
        out = []
        for layer in past:
            entries = []
            for t in layer:
                if torch.is_tensor(t) and t.dim() >= 1 and t.shape[0] > row:
                    entries.append(t[row:row + 1].contiguous())
                else:
                    entries.append(t)
            out.append(tuple(entries))
        return out

    def _pad_prefix_cache(self, n_pad: int, device):
        """KV cache of ``n_pad`` pad tokens (positions 0..n_pad-1).

        Computed by a small standalone causal forward — the pad tokens
        attend to each other, so every value is FINITE garbage. (Naively
        prefilling ``[PAD]*n + prompt`` in one masked forward instead gives
        fully-masked pad query rows, whose softmax is NaN and which then
        corrupt real positions through 0*NaN in later layers' V matmuls.)
        The result depends only on ``n_pad`` (weights fixed at inference),
        so it is memoized and SHARED read-only across requests. Sharing is
        safe (O8): the model never mutates a passed-in ``past_key_values``
        in place — every layer concatenates past K/V with the new K/V via
        ``torch.cat`` (see ``MLA._forward_*``), which allocates fresh
        tensors, so a request's stored cache never aliases the memo entry's
        storage. The memo is LRU-bounded by ``self._pad_caches_max`` (F4):
        on a hit the entry is refreshed to most-recently-used; once the cap
        is reached the least-recently-used entry is evicted.
        """
        past = self._pad_caches.pop(n_pad, None)
        if past is not None:
            # LRU refresh: re-insert at the end (dicts keep insertion order).
            self._pad_caches[n_pad] = past
            return past
        ids = torch.full((1, n_pad), self._pad_token_id,
                         dtype=torch.long, device=device)
        _, _, past = self.model(input_ids=ids, use_cache=True)
        while len(self._pad_caches) >= self._pad_caches_max:
            # Evict the least-recently-used (front) entry.
            self._pad_caches.pop(next(iter(self._pad_caches)))
        self._pad_caches[n_pad] = past
        return past

    def _prefill(self, req: Request, device):
        """Run one request's prefill; returns the next-token logits [V].

        Watermark alignment: shorter prompts are prepended with
        ``req.prompt_pad`` pad-token KV entries (via ``_pad_prefix_cache``)
        so the stored cache length equals the admission watermark. The real
        prompt is then forwarded on top of that pad prefix with the pad
        region masked out (exactly neutral: masked keys get zero attention
        weight over finite values) and view-index positions. Pad entries sit
        at the FRONT of the stored cache and are masked out of every later
        batched decode step; real tokens' positions are shifted by a small
        constant ``prompt_pad`` (RoPE is relative, so attention scores are
        unchanged up to float rounding). With prompt_pad == 0 the unmasked
        fast path is bitwise identical to a solo forward.
        """
        n_pad = req.prompt_pad
        ids = req.prompt_token_ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        if n_pad > 0:
            pad_past = self._pad_prefix_cache(n_pad, device)
            attention_mask = torch.tensor(
                [[0] * n_pad + [1] * len(ids)], dtype=torch.long, device=device)
            position_ids = torch.arange(
                n_pad, n_pad + len(ids), device=device).unsqueeze(0)
            logits, _, past = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=pad_past,
                use_cache=True,
            )
            req.cache_len = n_pad + len(ids)
        else:
            logits, _, past = self.model(
                input_ids=input_ids, use_cache=True)
            req.cache_len = len(ids)
        req.past_key_values = past
        return logits[0, -1]

    def _prefill_group(self, reqs: List[Request], device):
        """Batched prefill for requests sharing prompt length AND pad
        (v5.12) — only ever called for prompt_pad == 0 groups (padded rows
        take the solo masked ``_prefill``; see ``step``).

        Rows of equal length need no padding mask and no position shifting:
        the causal forward scores each row independently, so one [B, L]
        forward replaces B solo forwards. This is the ONLY batched prefill
        path for recurrent-state (hybrid) models — pad prefixes are
        unavailable to them (see __init__), and equal-length groups are
        exact: unmasked, unshifted, and each row's cache is a batch slice of
        the shared forward (a view; the storage stays alive until the
        request finishes — the same lifetime contract as pad-prefix views).

        Returns the per-request next-token logits rows (list of [V]).
        """
        L = len(reqs[0].prompt_token_ids)
        assert all(len(r.prompt_token_ids) == L for r in reqs), \
            "batched prefill requires equal prompt lengths"
        assert all(r.prompt_pad == 0 for r in reqs), \
            "batched prefill does not support pad prefixes"
        ids = torch.tensor([r.prompt_token_ids for r in reqs],
                           dtype=torch.long, device=device)
        logits, _, past = self.model(input_ids=ids, use_cache=True)
        for i, req in enumerate(reqs):
            req.past_key_values = [
                tuple(t[i:i + 1] for t in layer) for layer in past
            ]
            req.cache_len = L
        return [logits[i, -1] for i in range(len(reqs))]

    def _decode_batch(self, group: List[Request], outputs: Dict[int, int],
                      device):
        """One batched decode forward for a group of requests whose stored
        caches all have the same length P (watermark-aligned).

        input_ids is [B, 1] (each row's newest token). The batched past is
        the per-layer batch-concat of the rows' caches; rows with
        ``prompt_pad > 0`` carry garbage pad K/V at cache positions
        [0, prompt_pad), masked out by the attention_mask. Position ids
        default to the cache index P, which equals each row's true next
        position because pads were baked in at prefill.
        """
        P = group[0].cache_len
        input_ids = torch.tensor(
            [[r.generated_token_ids[-1]] for r in group],
            dtype=torch.long, device=device)
        past = self._batch_past([r.past_key_values for r in group])
        if any(r.prompt_pad > 0 for r in group):
            attention_mask = torch.tensor(
                [[0] * r.prompt_pad + [1] * (P + 1 - r.prompt_pad)
                 for r in group],
                dtype=torch.long, device=device)
        else:
            # Unmasked fast path: bitwise identical to solo decode steps.
            attention_mask = None
        logits, _, past = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        for j, req in enumerate(group):
            token = self._sample(logits[j, -1], req.temperature)
            req.append_token(token)
            self.block_manager.append_tokens(req.request_id, 1)
            req.past_key_values = self._row_past(past, j)
            req.cache_len = P + 1
            outputs[req.request_id] = token

    def step(self) -> Dict[int, int]:
        """Execute one decoding step for all running requests.

        Each request is fed only its newest token once its past_key_values
        exist (prefill feeds the full prompt exactly once). All decode-phase
        requests sharing a cache length are merged into ONE batched forward
        per step (watermark alignment; see class docstring). Returns
        {request_id: token_id} for tokens produced this step.
        """
        # Retire requests that finished, then admit queued ones into freed
        # slots (continuous batching).
        for req in list(self.running_requests):
            if req.is_done():
                self._finish(req)
        self.schedule()
        # Retire requests that were admitted already-complete — e.g.
        # max_new_tokens == 0, where is_done() holds before any token is
        # generated. They must NOT run a prefill (which would emit one
        # token anyway); their blocks are released here, so step()
        # returns {} for them. The run() loop already retired such
        # requests on its own retire pass, so that path is unchanged.
        for req in list(self.running_requests):
            if req.is_done():
                self._finish(req)
        if not self.running_requests:
            return {}

        outputs: Dict[int, int] = {}
        device = next(self.model.parameters()).device
        with torch.no_grad():
            # 1) Prefill newly admitted requests, grouped by
            #    (prompt length, prompt_pad) (v5.12): equal-length UNPADDED
            #    rows batch into one forward (exact — no mask, no position
            #    shift); rows with a watermark pad prefix (prompt_pad > 0)
            #    each take the solo masked prefill — their per-row pad
            #    regions cannot be expressed in one unmasked [B, L] forward.
            #    For recurrent-state (hybrid) models prompt_pad is always 0,
            #    making this grouping the only batched prefill available
            #    (pad prefixes are unusable for them). A prefilled request
            #    joins decode batches next step.
            pending = [r for r in self.running_requests
                       if r.past_key_values is None]
            prefill_groups: Dict[Tuple[int, int], List[Request]] = {}
            for req in pending:
                prefill_groups.setdefault(
                    (len(req.prompt_token_ids), req.prompt_pad), []
                ).append(req)
            for (length, pad) in sorted(prefill_groups):
                group = prefill_groups[(length, pad)]
                # O2: a failed prefill must not hold its blocks forever —
                # retire the request(s) (error recorded, blocks freed) and
                # re-raise; the remaining requests keep their state and the
                # engine can continue serving them on the next step. A
                # batched-forward failure cannot be attributed to one row,
                # so the whole group is retired together.
                try:
                    if pad == 0 and len(group) >= 2:
                        logits_rows = self._prefill_group(group, device)
                    else:
                        logits_rows = [self._prefill(r, device)
                                       for r in group]
                except Exception as exc:
                    for req in group:
                        self._finish(req, error=exc)
                    raise
                for req, logits_row in zip(group, logits_rows):
                    token = self._sample(logits_row, req.temperature)
                    req.append_token(token)
                    self.block_manager.append_tokens(req.request_id, 1)
                    outputs[req.request_id] = token
                    if req.is_done():
                        self._finish(req)
            # 2) Batched decode: one forward per cache-length group. With
            #    watermark alignment there is normally a single group; a
            #    longer prompt admitted mid-run splits off a second group
            #    (correct fallback, just less batched).
            decode = [r for r in self.running_requests
                      if r.past_key_values is not None and not r.is_done()]
            groups: Dict[int, List[Request]] = {}
            for req in decode:
                groups.setdefault(req.cache_len, []).append(req)
            for cache_len in sorted(groups):
                group = groups[cache_len]
                # O2: the batched forward is shared by the whole group, so
                # a failure fails all of its rows — retire them all (blocks
                # freed, error recorded) and re-raise.
                try:
                    self._decode_batch(group, outputs, device)
                except Exception as exc:
                    for req in group:
                        self._finish(req, error=exc)
                    raise
                for req in group:
                    if req.is_done():
                        self._finish(req)
        return outputs

    def run(self, max_steps: Optional[int] = None) -> Dict[int, List[int]]:
        """Run until every queued/running request finishes (or max_steps).

        Returns {request_id: generated_token_ids}."""
        steps = 0
        while self.request_queue or self.running_requests:
            # Check the budget BEFORE consuming a step so that
            # max_steps=0 means exactly 0 steps (previously the check ran
            # only after self.step(), forcing at least one step).
            if max_steps is not None and steps >= max_steps:
                break
            if not self.running_requests:
                self.schedule()
            # Retire requests that are done before consuming a step
            for req in list(self.running_requests):
                if req.is_done():
                    self._finish(req)
            if not self.running_requests:
                if not self.request_queue:
                    break
                continue
            self.step()
            steps += 1
        return {r.request_id: list(r.generated_token_ids)
                for r in self.finished_requests}

    def generate(self, prompts: List[List[int]], max_new_tokens: int = 32,
                 temperature: float = 0.0,
                 eos_token_id: Optional[int] = None) -> List[List[int]]:
        """Convenience batch API: submit all prompts and run to completion."""
        ids = [self.add_request(p, max_new_tokens=max_new_tokens,
                                temperature=temperature, eos_token_id=eos_token_id)
               for p in prompts]
        results = self.run()
        return [results[i] for i in ids]

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float) -> int:
        """temperature <= 0 means greedy (argmax); otherwise multinomial."""
        if temperature is None or temperature <= 0:
            return int(logits.argmax(dim=-1).item())
        probs = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())


class GPUMonitor:
    """Lightweight GPU monitor.

    Metrics are stored in plain Python lists in memory (no Prometheus
    dependency). If the optional ``prometheus_client`` package is installed
    and ``export_prometheus=True``, the collected gauges are also exported
    to the default Prometheus registry.
    """

    def __init__(self, export_prometheus: bool = True):
        self.metrics = {
            "gpu_utilization": [],     # percent 0-100, per sample
            "gpu_memory_used": [],     # GiB
            "gpu_memory_total": [],    # GiB
            "gpu_memory_ratio": [],    # used / total, 0-1
        }
        self._prom = None
        if export_prometheus:
            try:
                from prometheus_client import Gauge
                self._prom = {
                    "gpu_utilization": Gauge("gpu_utilization", "GPU utilization %"),
                    "gpu_memory_ratio": Gauge("gpu_memory_ratio", "GPU memory used/total"),
                }
            except ImportError:
                self._prom = None  # optional dependency absent -> in-memory only

    def collect(self):
        """Collect one sample of GPU metrics (per device, if CUDA exists)."""
        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            util = float(torch.cuda.utilization(i))
            used = torch.cuda.memory_allocated(i) / 1024 ** 3
            total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            ratio = used / total if total > 0 else 0.0
            self.metrics["gpu_utilization"].append(util)
            self.metrics["gpu_memory_used"].append(used)
            self.metrics["gpu_memory_total"].append(total)
            self.metrics["gpu_memory_ratio"].append(ratio)
            if self._prom is not None:
                self._prom["gpu_utilization"].set(util)
                self._prom["gpu_memory_ratio"].set(ratio)

    def get_hpa_recommendation(self, current_replicas: int = 2,
                               target_utilization: float = 0.6) -> Dict:
        """Recommend replica count from *current utilization* (M27).

        Load is the max of mean GPU utilization (fraction) and mean memory
        ratio — unit-consistent 0..1 values compared against the 0.85 / 0.4
        thresholds. The desired replica count is derived from
        ``current_replicas * load / target_utilization``, never from the
        length of the metrics history.
        """
        if not self.metrics["gpu_utilization"]:
            return {"replicas": current_replicas, "action": "hold",
                    "reason": "no metrics collected"}

        avg_util = (sum(self.metrics["gpu_utilization"])
                    / len(self.metrics["gpu_utilization"])) / 100.0
        avg_mem = (sum(self.metrics["gpu_memory_ratio"])
                   / len(self.metrics["gpu_memory_ratio"]))
        load = max(avg_util, avg_mem)

        desired = max(1, min(10, math.ceil(current_replicas * load / target_utilization)))
        if load > 0.85:
            desired = max(desired, min(10, current_replicas + 1))
            return {"replicas": desired, "action": "scale_up", "load": load}
        if load < 0.4:
            desired = min(desired, max(1, current_replicas - 1))
            return {"replicas": desired, "action": "scale_down", "load": load}
        return {"replicas": current_replicas, "action": "hold", "load": load}

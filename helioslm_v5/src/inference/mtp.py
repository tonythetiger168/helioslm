"""Multi-Token Prediction (MTP) - DeepSeek-V3 Style

Predicts multiple future tokens simultaneously using shared transformer
layers, with real speculative verification against the main model.

Depth convention (DeepSeek-V3 style, per-module):
  - Module k (module_index=k, 0-based) predicts the token at position
    t + k + 2, using the hidden state at position t fused with the
    embedding of the token at position t + k + 1.
  - Hidden states are chained: module k's output hidden feeds module k+1.

Main-model contract (HeliosLMv5.forward):
  forward(input_ids, attention_mask=None, position_ids=None,
          past_key_values=None, use_cache=False, images=None,
          audio_features=None) -> (logits[B,L,V], hidden[B,L,H], past)

Note: the MTP *training loss* lives on the training side (out of scope
here); this module guarantees weight sharing (embed_tokens / lm_head are
the main model's own modules when passed in) so that such a loss trains
shared representations rather than private copies.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from helioslm_v5.src.attention.linear_attention import past_seq_len as _past_seq_len_impl
except ImportError:  # pragma: no cover - package-relative fallback
    from ..attention.linear_attention import past_seq_len as _past_seq_len_impl


def _derive_nhead(hidden_size: int, config) -> int:
    """Pick a num_heads that divides hidden_size (M22)."""
    nhead = getattr(getattr(config, "attention", None), "num_attention_heads", None)
    if isinstance(nhead, int) and nhead > 0 and hidden_size % nhead == 0:
        return nhead
    for h in (8, 4, 2):
        if hidden_size % h == 0:
            return h
    return 1


class MTPModule(nn.Module):
    """
    Multi-Token Prediction module.

    Module with ``module_index = k`` predicts the token at position
    t + k + 2 from the hidden state at t and the embedding of the token
    at t + k + 1 (see module docstring).
    """

    def __init__(self, config, module_index: int = 0,
                 embed_tokens: Optional[nn.Embedding] = None,
                 lm_head: Optional[nn.Linear] = None):
        super().__init__()
        self.module_index = module_index  # k: predicts t + k + 2
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # M24: share embedding / LM head with the main model (references,
        # not new parameters). Fallback to private copies only when the
        # caller does not provide them (e.g. standalone unit tests).
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        if lm_head is not None:
            self.lm_head = lm_head
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # MTP-specific transformer layer (causal-masked at runtime, M22)
        nhead = _derive_nhead(config.hidden_size, config)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=nhead,
            dim_feedforward=getattr(config, "intermediate_size", config.hidden_size * 4),
            batch_first=True,
            dropout=0.0,
        )

        # Projection to combine hidden state with next-token embedding
        self.fusion_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)

    # ------------------------------------------------------------------
    @staticmethod
    def _causal_mask(n: int, device, dtype) -> torch.Tensor:
        # M22: upper-triangular -inf mask => no future-token leakage
        return torch.triu(
            torch.full((n, n), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )

    def forward_with_hidden(
        self,
        main_hidden_states: torch.Tensor,
        next_token_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Two paths (C9):

        Training path (seq > 1): ``main_hidden_states`` [B, L, H] and
        ``next_token_ids`` [B, L] are the full sequence. With
        d = module_index + 1, position t fuses hidden[t] with
        emb(ids[t + d]) and the returned logits[t] predict ids[t + d + 1].
        Output length is L - d - 1.

        Inference path (seq == 1): fuse ``main_hidden_states[:, -1]`` with
        the embedding of the just-sampled token ``next_token_ids[:, -1]``;
        no sequence shift is applied.

        Returns:
            logits:      [B, L', vocab]
            hidden_out:  [B, L', hidden] — chained into the next MTP module
        """
        B, seq, H = main_hidden_states.shape

        if seq > 1:
            # ----- training path -----
            if next_token_ids is None:
                raise ValueError(
                    "Training path requires the full token sequence as "
                    "next_token_ids [B, L]"
                )
            d = self.module_index + 1
            usable = seq - d - 1
            if usable <= 0:
                raise ValueError(
                    f"Sequence length {seq} too short for MTP depth {d}"
                )
            h = main_hidden_states[:, :usable, :]          # hidden[t]
            tok = next_token_ids[:, d : d + usable]        # ids[t + d]
            emb = self.embed_tokens(tok)                   # [B, usable, H]
            fused = self.fusion_proj(torch.cat([h, emb], dim=-1))
            mask = self._causal_mask(usable, fused.device, fused.dtype)
            transformed = self.transformer(fused, src_mask=mask)
        else:
            # ----- inference path (single token, no shift) -----
            if next_token_ids is None:
                raise ValueError(
                    "Inference path requires the just-sampled token id "
                    "(next_token_ids [B, 1]); the zeros-embedding fallback "
                    "was removed (m3) because it produced meaningless logits."
                )
            h = main_hidden_states[:, -1:, :]                       # [B,1,H]
            emb = self.embed_tokens(next_token_ids[:, -1:])         # [B,1,H]
            fused = self.fusion_proj(torch.cat([h, emb], dim=-1))
            transformed = self.transformer(fused)  # length-1: mask is a no-op

        logits = self.lm_head(transformed)
        return logits, transformed

    def forward(self, main_hidden_states: torch.Tensor,
                next_token_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns logits only (see ``forward_with_hidden`` for semantics)."""
        return self.forward_with_hidden(main_hidden_states, next_token_ids)[0]

    @torch.no_grad()
    def generate(self, main_hidden_states: torch.Tensor, prev_token_ids: torch.Tensor,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        """Sample one predicted token per batch row (inference path).

        eval() mode is applied for the duration of the call and the
        original training/eval mode is restored on exit — on exceptions
        too (O1, v5.4)."""
        was_training = self.training
        self.eval()
        try:
            logits, _ = self.forward_with_hidden(main_hidden_states, prev_token_ids)
            next_logits = logits[:, -1, :]
            if temperature <= 0:
                return next_logits.argmax(dim=-1, keepdim=True)
            next_logits = next_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        finally:
            self.train(was_training)


def _filter_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering of a probability vector [V].

    Same semantics as ``HeliosLMv5.generate``'s non-MTP path: drop the
    smallest-probability tail whose cumulative mass exceeds ``top_p``,
    always keep the top token, then renormalize. ``top_p >= 1.0`` (or
    None) disables filtering.
    """
    if top_p is None or top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()  # keep the first token above the cut
    remove[0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    return filtered / filtered.sum().clamp(min=1e-12)


@dataclass
class MTPGenerateResult:
    """Result of MTPDecoder.generate."""
    sequences: torch.Tensor       # [B, prompt_len + num_new] token ids
    acceptance_rate: float        # accepted MTP draft tokens / drafted MTP tokens
    num_drafted: int              # MTP-proposed tokens verified (excl. main tokens)
    num_accepted: int             # MTP drafts accepted
    num_rounds: int               # speculative rounds executed


class MTPDecoder:
    """
    Multi-Token Prediction decoder combining main model + MTP modules.

    Strategy (real speculative decoding, C10):
      1. Main model produces the next-token distribution and hidden state
         (KV cache reused across rounds — no full-prefix recompute, m2).
      2. Main model samples token x0; MTP modules then draft x1..xn with
         chained hidden states (module k's output feeds module k+1, M23).
      3. Verification: ONE main-model forward over the drafted suffix
         [x0..xn] yields the main distribution at every drafted position.
         Greedy: a draft is accepted iff it equals the main argmax.
         Sampling: accepted with probability min(1, p_main(x)/p_draft(x)).
      4. The first rejection truncates the draft run; a correction token is
         resampled from norm((p_main - p_draft)+) at that position — the
         standard speculative-sampling rejection rule, so the emitted
         token distribution matches the (temperature- and top-p-filtered)
         main-model distribution exactly (M-I1: no double temperature).
         If all drafts are accepted, a bonus token is sampled from the
         main distribution one step beyond the last draft.
      5. KV-cache rollback (v5.2): the verification forward's cache covers
         the whole speculative suffix. Since the cache contract guarantees
         every per-layer cache tensor is sliceable along dim 2 (sequence
         axis), a rejection is rolled back by *truncating* the cache to the
         committed length per sequence — no prefix recompute (v5.1 used to
         re-forward the accepted prefix on every rejection).
         v5.5 (hybrid models): GatedDeltaAttention layers hold a fixed-size
         recurrent state (marked ``_is_recurrent_state``) that CANNOT be
         truncated. When the committed prefix does not cover the whole
         speculative suffix, the pre-speculation state is restored (the
         model never mutates input caches, so the round-start cache is the
         clone) and the committed prefix is replayed once
         (``_replay_committed``). Full-acceptance rounds still need no
         recompute: the verification state is already exact.

    Batched generation (v5.2):
      - ``input_ids`` is rectangular [B, L] (uniform prompt length) and
        ``max_new_tokens`` is a single scalar budget, so all rows start each
        round with the same committed length.
      - Each round, *active* rows are grouped by their cache length. Inside
        a group every row has an identical cache length, so the main-model
        forwards (round-start step + verification) concatenate the per-row
        caches along the batch dim and run ONE batched forward — per-row
        results are identical to running each row solo (no padding, no
        masking, batch-independent math).
      - Acceptance lengths differ per row, so groups re-form every round
        (rows that rejected shrink to shorter caches and regroup with
        equally-long rows). In the worst case (all rows diverge) a group
        degenerates to a single row — correct, just less batched.
      - The MTP drafting step is per-sequence independent by design
        (chained hidden states); the expensive main-model forwards are the
        batched ones.
      - EOS / budget: rows that finish leave the active set immediately and
        no longer participate in later rounds; the returned ``sequences``
        tensor is rectangular, right-padded with ``pad_token_id`` for rows
        that stopped early (mirrors ``HeliosLMv5.generate`` freezing).

    Reliability note: we deliberately do NOT pad ragged caches into one
    batch (RoPE positions and the causal mask share ``position_ids`` in the
    attention stack, so unequal-length cache padding cannot be made exact
    through the public forward contract). Length-grouped batching is exact.
    """

    def __init__(self, main_model, mtp_modules, config):
        self.main_model = main_model
        self.mtp_modules = nn.ModuleList(mtp_modules)
        self.num_mtp = len(mtp_modules)
        self.config = config
        # Optional torch.Generator for the speculative-acceptance draws
        # (O11). None -> the global default generator, reproducible via
        # torch.manual_seed. Assign a torch.Generator to isolate the
        # acceptance RNG from other consumers of the global stream.
        self.generator: Optional[torch.Generator] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _sample_from(logits: torch.Tensor, temperature: float,
                     top_p: float = 1.0):
        """logits: [V]. Returns (token_id:int, probs:Tensor[V]) where
        ``probs`` is the EXACT distribution the token was drawn from
        (temperature scaling + nucleus filter applied) — verification
        needs it as the sampling distribution q / p0."""
        if temperature is None or temperature <= 0:
            probs = F.softmax(logits, dim=-1)
            return int(logits.argmax(dim=-1).item()), probs
        probs = _filter_top_p(F.softmax(logits / temperature, dim=-1), top_p)
        return int(torch.multinomial(probs, 1).item()), probs

    @staticmethod
    def _residual_sample(p: torch.Tensor, q: torch.Tensor) -> int:
        """Strict speculative-sampling correction: sample from
        ``norm((p - q)+)`` — together with the min(1, p/q) acceptance
        rule this makes the emitted token distribution EXACTLY equal the
        target distribution p (M-I1). ``p`` is used as-is (it is already
        a probability vector; no second temperature/softmax is applied).
        Falls back to sampling p directly when the residual is
        numerically empty (p == q up to fp noise)."""
        residual = (p - q).clamp(min=0)
        total = float(residual.sum().item())
        if total <= 1e-12:
            return int(torch.multinomial(p, 1).item())
        return int(torch.multinomial(residual / total, 1).item())

    def _main_forward(self, ids: torch.Tensor, past,
                      attention_mask: Optional[torch.Tensor] = None):
        """Unpack the main model's (logits, hidden, past) triple (C8).

        ``attention_mask`` (optional) is passed straight through to
        ``HeliosLMv5.forward`` — the caller slices it to this forward's
        kv length (past + current).
        """
        logits, hidden, past = self.main_model(
            input_ids=ids, attention_mask=attention_mask,
            past_key_values=past, use_cache=True
        )
        return logits, hidden, past

    def _replay_committed(self, bpast, row: int, prefix_tokens, device,
                          attention_mask: Optional[torch.Tensor] = None):
        """Recurrent-state rollback for hybrid models (v5.5).

        A fixed-size recurrent state cannot be truncated to the committed
        prefix, so rollback = RESTORE the pre-speculation state (a
        batch-row slice of the round-start cache ``bpast`` — the model
        never mutates input caches in place, and ``_row_past`` copies via
        .contiguous(), so this slice IS the pre-speculation clone) and
        REPLAY the committed tokens except the last through the main model
        (the last committed token is fed by the next round's step A). The
        replay is exact: the recurrence and MLA cache appends are
        deterministic, so the rebuilt cache equals what a non-speculative
        decode would hold. Cost: one short forward, paid only on rounds
        where the committed prefix does not cover the whole speculative
        suffix (i.e. any rejection or a budget cut); pure-MLA models never
        take this path.
        """
        base = self._row_past(bpast, row)  # batch slice only, no truncation
        if not prefix_tokens:
            return base
        ids = torch.tensor([list(prefix_tokens)], dtype=torch.long,
                           device=device)
        _, _, past = self._main_forward(ids, base, attention_mask)
        return past

    # ------------------------------------------------------------------
    # batched-cache helpers (rely only on the cache contract: per-layer
    # tuples of tensors whose dim 2 is the sequence axis; recurrent-state
    # tensors are marked with ``_is_recurrent_state`` and have NO sequence
    # axis — they are batch-sliced but never dim-2-truncated)
    # ------------------------------------------------------------------
    @staticmethod
    def _is_state_tensor(t) -> bool:
        """Duck-typed marker check for recurrent-state cache tensors
        (GatedDeltaAttention StateTensor, v5.5)."""
        return bool(getattr(t, "_is_recurrent_state", False))

    @classmethod
    def _past_has_state(cls, past) -> bool:
        """True iff any per-layer cache holds a recurrent-state tensor."""
        if past is None:
            return False
        return any(cls._is_state_tensor(t) for layer in past for t in layer)

    @classmethod
    def _past_seq_len(cls, past) -> int:
        """Sequence length covered by a per-layer cache (0 for None).

        Recurrent-state tensors are skipped (fixed shape, no sequence
        axis). Under the hybrid interleave rule layer 0 is always MLA, so
        a sequence-length tensor always exists in a hybrid cache.
        """
        if past is None:
            return 0
        for layer in past:
            for t in layer:
                if torch.is_tensor(t) and t.dim() >= 3 \
                        and not cls._is_state_tensor(t):
                    return t.shape[2]
        return 0

    @staticmethod
    def _batch_past(pasts):
        """Concatenate same-length per-row caches along the batch dim."""
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

    @classmethod
    def _row_past(cls, past, row: int, length: Optional[int] = None):
        """Extract one batch row from a batched cache, optionally truncating
        the sequence axis (dim 2) to ``length`` — this truncation IS the
        speculative rollback (dim-2-sliceable cache contract); no recompute.

        Recurrent-state tensors (v5.5) are batch-sliced but NEVER dim-2
        truncated: the state has no sequence axis. Rolling a state cache
        back to an earlier commit point requires restoring the
        pre-speculation state and replaying the committed prefix, which is
        what the generate loop does via ``_replay_committed``.
        """
        out = []
        for layer in past:
            entries = []
            for t in layer:
                if torch.is_tensor(t) and t.dim() >= 1 and t.shape[0] > row:
                    r = t[row:row + 1]
                    if length is not None and r.dim() >= 3 \
                            and not cls._is_state_tensor(t):
                        # Slice beyond the size is a no-op, so this is safe
                        # even if a tensor's dim 2 were shorter than length.
                        r = r[:, :, :length]
                    entries.append(r.contiguous())
                else:
                    entries.append(t)
            out.append(tuple(entries))
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.7,
                 eos_token_id: Optional[int] = None,
                 top_p: float = 1.0,
                 attention_mask: Optional[torch.Tensor] = None
                 ) -> MTPGenerateResult:
        """Speculative decoding with the main model as verifier.

        ``top_p`` (< 1.0) enables nucleus filtering with the same
        semantics as ``HeliosLMv5.generate``'s non-MTP path: it is applied
        to the main model's distributions (x0, per-position verification,
        correction, bonus) AND to the draft distributions, so both sides
        of the accept ratio refer to the same filtered target. The default
        1.0 disables filtering (backward compatible).

        ``attention_mask`` follows the ``HeliosLMv5.generate`` contract:
        an optional [B, L] mask over the prompt (1 = real, 0 = pad).
        Generated positions are always real, so the decoder grows the mask
        with ones and slices it to each main-model forward's kv length
        (past + current) — the attention stack requires a mask that
        covers past + current exactly (its current-only shorthand would
        left-pad the whole cached prefix with ones, un-masking pad
        tokens). None (default) keeps every forward unmasked.

        All modules are switched to eval() for the duration of the call and
        their original training/eval modes are restored on exit — on
        exceptions too (O1, v5.4).
        """
        mtp_was_training = [m.training for m in self.mtp_modules]
        main_was_training = getattr(self.main_model, "training", None)
        for m in self.mtp_modules:
            m.eval()  # M22: no dropout etc. during generation
        if hasattr(self.main_model, "eval"):
            self.main_model.eval()
        try:
            return self._generate_impl(
                input_ids, max_new_tokens, temperature, eos_token_id, top_p,
                attention_mask)
        finally:
            for m, was_training in zip(self.mtp_modules, mtp_was_training):
                m.train(was_training)
            if main_was_training is not None and hasattr(self.main_model, "train"):
                self.main_model.train(main_was_training)

    @torch.no_grad()
    def _generate_impl(self, input_ids: torch.Tensor, max_new_tokens: int,
                       temperature: float, eos_token_id: Optional[int],
                       top_p: float,
                       attention_mask: Optional[torch.Tensor]
                       ) -> MTPGenerateResult:
        """Body of ``generate`` (semantics documented there); called with
        eval mode already applied and restored by the wrapper."""
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be [B, L], got shape {tuple(input_ids.shape)}"
            )
        B = input_ids.shape[0]
        if B == 0:
            # Empty batch: no rows to decode. Mirror the normal result
            # structure — an empty [0, L] sequences tensor (prompt length
            # preserved) with zeroed counters.
            return MTPGenerateResult(
                sequences=torch.empty((0, input_ids.shape[1]),
                                      dtype=input_ids.dtype,
                                      device=input_ids.device),
                acceptance_rate=0.0,
                num_drafted=0,
                num_accepted=0,
                num_rounds=0,
            )
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)
        pad_token_id = getattr(self.config, "pad_token_id", 0)

        device = input_ids.device
        greedy = temperature is None or temperature <= 0

        # attention_mask [B, L] over the prompt (see generate's docstring):
        # grown with ones for the generated positions so it can be sliced
        # to any forward's kv length (past + current) — the MLA contract
        # requires a mask covering past + current exactly.
        mask = attention_mask
        if mask is not None:
            if mask.dim() != 2 or mask.shape[0] != B \
                    or mask.shape[1] != input_ids.shape[1]:
                raise ValueError(
                    f"attention_mask must be [B, {input_ids.shape[1]}], got "
                    f"shape {tuple(mask.shape)}"
                )
            mask = torch.cat([
                mask,
                torch.ones(B, max(0, int(max_new_tokens)),
                           dtype=mask.dtype, device=mask.device),
            ], dim=1)

        def main_probs(logits_row: torch.Tensor) -> torch.Tensor:
            """Target distribution at a verification/bonus position:
            temperature scaling + nucleus filter — identical to the
            distribution x0 was sampled from, and to the non-MTP
            ``HeliosLMv5.generate`` semantics (F5)."""
            if greedy:
                return F.softmax(logits_row, dim=-1)
            return _filter_top_p(
                F.softmax(logits_row / temperature, dim=-1), top_p)

        # Per-row state. Cache invariant at the top of each round:
        # ``pasts[i]`` is None (first round) or covers exactly seqs[i][:-1].
        seqs: List[List[int]] = [[int(t) for t in row] for row in input_ids.tolist()]
        pasts: List[Optional[list]] = [None] * B
        remaining = [max(0, int(max_new_tokens))] * B
        num_drafted = 0
        num_accepted = 0
        num_rounds = 0
        active = [i for i in range(B) if remaining[i] > 0]

        while active:
            # O10 (v5.4): num_rounds counts actual speculative rounds
            # (outer-loop iterations), matching the MTPGenerateResult
            # field doc — NOT the number of per-group forwards executed
            # within a round (rows split into cache-length groups).
            num_rounds += 1
            # Group active rows by cache length: within a group every row
            # has the same committed length, hence the same remaining
            # budget (uniform prompt length + scalar max_new_tokens), so the
            # group shares one exact batched forward.
            groups: dict = {}
            for i in active:
                groups.setdefault(self._past_seq_len(pasts[i]), []).append(i)
            next_active: List[int] = []

            for clen in sorted(groups):
                rows = groups[clen]
                G = len(rows)
                seq_len = len(seqs[rows[0]])  # uniform within the group
                rem = remaining[rows[0]]      # uniform within the group

                # ---- Step A: one batched main-model forward -------------
                # First round (clen == 0) feeds the full prompt; later
                # rounds feed only the last committed token per row.
                if clen == 0:
                    step_ids = torch.tensor(
                        [seqs[i] for i in rows], dtype=torch.long, device=device)
                    bpast = None
                else:
                    step_ids = torch.tensor(
                        [[seqs[i][-1]] for i in rows], dtype=torch.long,
                        device=device)
                    bpast = self._batch_past([pasts[i] for i in rows])
                # Mask slice covering past + current (generated positions
                # are ones; see generate's docstring).
                step_mask = None if mask is None else \
                    mask[rows][:, :clen + step_ids.shape[1]]
                logits, hidden, bpast = self._main_forward(
                    step_ids, bpast, step_mask)
                # bpast now covers the FULL current sequence of every row.

                x0: List[int] = []
                p0: List[torch.Tensor] = []
                for r in range(G):
                    tok, prob = self._sample_from(logits[r, -1],
                                                  temperature, top_p)
                    x0.append(tok)
                    p0.append(prob)

                if rem == 1 or self.num_mtp == 0:
                    # m1: budget too small for a speculative round -> the
                    # whole group takes one plain (non-speculative) step.
                    for r, i in enumerate(rows):
                        seqs[i].append(x0[r])
                        remaining[i] -= 1
                        stop = (eos_token_id is not None
                                and x0[r] == eos_token_id)
                        if stop or remaining[i] == 0:
                            pasts[i] = None  # finished: drop the cache
                        else:
                            # bpast covers seqs[i] == new seq[:-1] exactly.
                            pasts[i] = self._row_past(bpast, r)
                            next_active.append(i)
                    continue

                # ---- Step B: draft x1..xn per row (independent chains) ---
                n_draft = min(self.num_mtp, rem - 1)
                draft_ids: List[List[int]] = []
                draft_probs: List[List[Optional[torch.Tensor]]] = []
                for r in range(G):
                    ids_r = [x0[r]]
                    probs_r: List[Optional[torch.Tensor]] = [p0[r]]
                    h_prev = hidden[r:r + 1, -1:, :]              # [1,1,H]
                    prev_tok = torch.tensor([[x0[r]]], device=device)
                    for module in self.mtp_modules[:n_draft]:
                        mtp_logits, h_prev = module.forward_with_hidden(
                            h_prev, prev_tok)
                        xk, qk = self._sample_from(mtp_logits[0, -1],
                                                   temperature, top_p)
                        ids_r.append(xk)
                        probs_r.append(qk)
                        prev_tok = torch.tensor([[xk]], device=device)
                    draft_ids.append(ids_r)
                    draft_probs.append(probs_r)

                # ---- Step C: one batched verification forward -----------
                suffix = torch.tensor(draft_ids, dtype=torch.long,
                                      device=device)  # [G, n+1]
                v_mask = None if mask is None else \
                    mask[rows][:, :seq_len + suffix.shape[1]]
                v_logits, _, v_past = self._main_forward(suffix, bpast, v_mask)
                # v_logits[r, j] = main distribution for the token following
                # draft_ids[r][j]; draft_ids[r][j+1] is checked against it.
                # The last row validates the final draft / sources the bonus.
                has_state = self._past_has_state(v_past)

                # ---- Step D+E: per-row accept/reject, commit, rollback --
                for r, i in enumerate(rows):
                    ids_r = draft_ids[r]
                    accepted: List[int] = []
                    rejected = False
                    for j, tok in enumerate(ids_r):
                        if j == 0:
                            p = p0[r]  # x0 came from this exact distribution
                        else:
                            p = main_probs(v_logits[r, j - 1])
                        if greedy:
                            ok = int(p.argmax(dim=-1).item()) == tok
                        else:
                            q = draft_probs[r][j]
                            p_tok = float(p[tok].item())
                            q_tok = max(float(q[tok].item()), 1e-12)
                            # O11 (v5.4): acceptance draw uses torch's RNG
                            # (self.generator, defaulting to the global
                            # generator) instead of Python's ``random`` —
                            # reproducible via torch.manual_seed.
                            u = float(torch.rand(
                                (), generator=self.generator).item())
                            ok = u < min(1.0, p_tok / q_tok)
                        if j > 0:
                            num_drafted += 1
                        if ok:
                            accepted.append(tok)
                            if j > 0:
                                num_accepted += 1
                        else:
                            # First rejection: truncate here, then commit a
                            # correction token. Greedy: main argmax.
                            # Sampling: strict speculative resampling from
                            # norm((p - q)+) — p is already the (filtered)
                            # target probability vector and is used
                            # directly, no second temperature (M-I1).
                            if greedy:
                                corr = int(p.argmax(dim=-1).item())
                            else:
                                corr = self._residual_sample(
                                    p, draft_probs[r][j])
                            accepted.append(corr)
                            rejected = True
                            break

                    if not rejected and len(accepted) < remaining[i]:
                        # All drafts accepted: bonus token one step past the
                        # last draft from the main distribution.
                        p_bonus = main_probs(v_logits[r, len(ids_r) - 1])
                        if greedy:
                            bonus = int(p_bonus.argmax(dim=-1).item())
                        else:
                            bonus = int(torch.multinomial(p_bonus, 1).item())
                        accepted.append(bonus)

                    # Commit with per-row budget + EOS early stop.
                    committed: List[int] = []
                    stop = False
                    for tok in accepted:
                        if remaining[i] == 0:
                            break
                        committed.append(tok)
                        remaining[i] -= 1
                        if eos_token_id is not None and tok == eos_token_id:
                            stop = True
                            break
                    seqs[i].extend(committed)

                    if stop or remaining[i] == 0:
                        pasts[i] = None  # finished: drop the cache
                    else:
                        # Rollback. v_past covers bpast + the whole
                        # speculative suffix (len(ids_r) tokens); the cache
                        # must end up covering new seq[:-1], i.e. bpast +
                        # committed[:-1] (len(committed) - 1 tokens).
                        if has_state and len(committed) - 1 != len(ids_r):
                            # Hybrid model: recurrent state cannot be
                            # truncated -> restore the pre-speculation
                            # state and replay the committed prefix.
                            replay_mask = None if mask is None else \
                                mask[i:i + 1][:, :seq_len + len(committed) - 1]
                            pasts[i] = self._replay_committed(
                                bpast, r, committed[:-1], device,
                                replay_mask)
                        else:
                            # Rollback by truncation: slice dim 2 down to
                            # the committed prefix (new seq[:-1] = seq_len +
                            # c - 1). The rejected tail is discarded WITHOUT
                            # recompute. (With state tensors, this branch is
                            # only taken when the committed prefix covers
                            # the whole suffix, so the state is exact and
                            # the dim-2 slice is a no-op.)
                            pasts[i] = self._row_past(
                                v_past, r, seq_len + len(committed) - 1)
                        next_active.append(i)

            active = next_active

        # Rectangular output: rows that stopped early are right-padded with
        # pad_token_id (same convention as HeliosLMv5.generate's freezing).
        max_len = max(len(s) for s in seqs)
        sequences = torch.full((B, max_len), pad_token_id,
                               dtype=input_ids.dtype, device=device)
        for i, s in enumerate(seqs):
            sequences[i, :len(s)] = torch.tensor(
                s, dtype=input_ids.dtype, device=device)

        acceptance_rate = (num_accepted / num_drafted) if num_drafted else 0.0
        return MTPGenerateResult(
            sequences=sequences,
            acceptance_rate=acceptance_rate,
            num_drafted=num_drafted,
            num_accepted=num_accepted,
            num_rounds=num_rounds,
        )

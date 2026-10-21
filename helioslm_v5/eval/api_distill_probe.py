"""API-teacher distillation probe (v5.82).

Task: the MOPD probe's ``rev`` domain (reverse four digits, then EOS).
The teacher is a char-level model trained with cross-entropy, stopped
early so it is imperfect, as real teachers are. It is only reachable
through two simulated APIs:

  TextAPI   one sampled answer per call (temperature 1);
  TopKAPI   the top-k next teacher tokens with log-probs, where a teacher
            token is TWO characters (a different tokenizer from the
            student's): p(ab) = p(a) p(b | a) from the char model, so the
            simulated API is exact up to the top-k cut.

A fresh student is trained on the same prompts in every arm:

  ground_truth  cross-entropy on the true answers (ceiling);
  seq_kd        cross-entropy on one teacher sample per prompt;
  seq_kd_verified  the same, keeping only samples a verifier accepts;
  topk_kd       sparse KD on the teacher's greedy path, targets aligned
                from 2-char top-k tokens to characters.
"""
import math

import torch

from helioslm_v5.eval.mopd_probe import (
    ANSWER_LEN, DIGIT0, EOS, N_DIGITS, HeliosLMv5, answer, config,
    exact_match, make_batch, train_ce)
from helioslm_v5.src.training.api_distill import (
    collect_sequences, first_char_targets, next_char_targets, sparse_kd_loss)

ALPHABET = [str(d) for d in range(N_DIGITS)] + ["$"]       # "$" = EOS


_OTHER = 0xE000      # any other token id i is carried as chr(_OTHER + i)


def char_id(c):
    if c == "$":
        return EOS
    if c.isdigit():
        return DIGIT0 + int(c)
    return ord(c) - _OTHER


def id_char(i):
    """Answer characters; tokens outside the digits and EOS (a sampled
    teacher can emit them) round-trip through a private-use character, so
    a wrong sample is trained on as exactly what the teacher said."""
    if i == EOS:
        return "$"
    if 0 <= i - DIGIT0 < N_DIGITS:
        return str(i - DIGIT0)
    return chr(_OTHER + i)


def train_teacher(steps, seed=0):
    torch.manual_seed(seed)
    return train_ce(HeliosLMv5(config()), lambda n, g: ["rev"] * n, steps,
                    seed)


class TextAPI:
    def __init__(self, model, temperature=1.0):
        self.model, self.t, self.calls = model, temperature, 0

    @torch.no_grad()
    def sample(self, prompt):
        self.calls += 1
        out = self.model.generate(prompt.unsqueeze(0),
                                  max_new_tokens=ANSWER_LEN,
                                  temperature=self.t, top_p=1.0)[0]
        s = "".join(id_char(i) for i in out[prompt.shape[0]:].tolist())
        return s.split("$")[0] + "$" if "$" in s else s


class TopKAPI:
    def __init__(self, model, k=5, chunk=2):
        self.model, self.k, self.chunk, self.calls = model, k, chunk, 0

    @torch.no_grad()
    def _next_char_logprobs(self, prompt, prefix):
        ids = torch.cat([prompt, torch.tensor([char_id(c) for c in prefix],
                                              dtype=torch.long)])
        logits = self.model(ids.unsqueeze(0))[0][0, -1].float()
        lp = torch.log_softmax(logits, -1)
        return {c: lp[char_id(c)].item() for c in ALPHABET}

    def topk(self, prompt, prefix):
        """Top-k next teacher tokens (up to ``chunk`` chars; a token ends
        early at EOS) after ``prefix``; log-probs are exact joints."""
        self.calls += 1
        cands = [("", 0.0)]
        for _ in range(self.chunk):
            nxt = []
            for tok, lp in cands:
                if tok.endswith("$"):
                    nxt.append((tok, lp))
                    continue
                for c, l in self._next_char_logprobs(prompt,
                                                     prefix + tok).items():
                    nxt.append((tok + c, lp + l))
            cands = nxt
        cands.sort(key=lambda x: -x[1])
        return cands[:self.k]


def topk_targets(api, prompt):
    """The teacher's greedy path (top-1 token at each step) and, per answer
    position, the student-token target distribution, from one top-k call
    per teacher token."""
    path, targets = "", []
    while len(path) < ANSWER_LEN and not path.endswith("$"):
        tk = api.topk(prompt, path)
        tok = tk[0][0]
        targets.append({char_id(c): p
                        for c, p in first_char_targets(tk).items()})
        for j in range(1, len(tok)):
            nxt = next_char_targets(tk, tok[:j])
            targets.append({char_id(c): p for c, p in nxt.items()})
        path += tok
    return path[:ANSWER_LEN], targets[:ANSWER_LEN]


def verify_rev(prompt, resp):
    digits = [i - DIGIT0 for i in prompt[2:].tolist()]
    return resp == "".join(str(d) for d in answer("rev", digits)) + "$"


def _pad(seq, n):
    return seq + [EOS] * (n - len(seq))


def train_student(examples, steps, seed, batch=64, lr=3e-3, soft=False):
    """examples: list of (prompt ids, answer char string) for hard targets,
    or (prompt ids, path, [target dicts]) when soft."""
    torch.manual_seed(1000 + seed)
    model = HeliosLMv5(config())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(2000 + seed)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(examples), (batch,), generator=gen).tolist()
        rows, tgt_rows = [], []
        for i in idx:
            ex = examples[i]
            path = ex[1]
            ans_ids = _pad([char_id(c) for c in path], ANSWER_LEN)[:ANSWER_LEN]
            rows.append(torch.cat([ex[0], torch.tensor(ans_ids)]))
            # train only on the answer's own positions (no pad targets)
            tg = ex[2] if soft else [{t: 1.0} for t in ans_ids]
            tgt_rows.append(tg[:min(len(path), ANSWER_LEN)])
        ids = torch.stack(rows)
        P = ids.shape[1] - ANSWER_LEN
        logits = model(ids)[0][:, P - 1:-1]
        flat, flat_t = [], []
        for b, tg in enumerate(tgt_rows):
            for j, t in enumerate(tg):
                flat.append(logits[b, j])
                flat_t.append(t)
        loss = sparse_kd_loss(torch.stack(flat), flat_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def prompt_pool(n, seed):
    gen = torch.Generator().manual_seed(5000 + seed)
    p, t = make_batch(["rev"] * n, gen)
    return list(p), ["".join(id_char(i) for i in row.tolist()) for row in t]


def topk_coverage(api, prompts):
    """Mean teacher probability mass the top-k tokens cover."""
    tot = 0.0
    for p in prompts:
        tot += sum(math.exp(lp) for _, lp in api.topk(p, ""))
    return tot / len(prompts)


__all__ = ["TextAPI", "TopKAPI", "collect_sequences", "exact_match",
           "prompt_pool", "topk_coverage", "topk_targets", "train_student",
           "train_teacher", "verify_rev"]

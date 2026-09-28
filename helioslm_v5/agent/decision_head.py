"""HeliosLM v5.32 — non-autoregressive DecisionHead (Jev architecture).

Jev's core engineering claim: decisions don't need token-by-token
generation — encode the state ONCE, answer K typed questions in a single
parallel pass. This is our reference version at toy scale:

- Encoder: mean-pooled char embeddings of the state text. Deviation
  recorded: Jev presumably uses a full LLM encoder; at 8.5M-char toy
  scale a bag-of-char-embeddings encoder is the honest minimum, and the
  interface (encode(state) -> d-vector) is swappable for our model's
  hidden states later.
- Heads: one per registered question. choice -> softmax over options;
  noul -> softmax over yes/no/unknown; score -> sigmoid. Confidence =
  max softmax prob — a CALIBRATION CLAIM.
- Training: cross-entropy on the answer + explicit Brier penalty tying
  confidence to outcome correctness (RLCD direction: probabilities answer
  to outcomes, not preferences). ECE/accuracy reported, never assumed.

The same head answers the pi-warden preset (5 questions) in ONE forward —
the K-questions-for-the-price-of-one property Jev demos.
"""
import torch
import torch.nn as nn

from decision import NOUL_NO, NOUL_UNKNOWN, NOUL_YES

_KIND_DIMS = {"choice": None, "noul": 3, "score": 1}
_NOUL_IDX = {NOUL_YES: 0, NOUL_NO: 1, NOUL_UNKNOWN: 2}


class DecisionHead(nn.Module):
    def __init__(self, questions: dict, d: int = 64, vocab: int = 1024,
                 brier_lambda: float = 1.0):
        """questions: {name: (kind, options_or_None)}. For 'choice',
        options is a tuple of answer strings; for noul/score, None."""
        super().__init__()
        self.qnames = list(questions)
        self.qmeta = questions
        self.brier_lambda = brier_lambda
        self.emb = nn.Embedding(vocab, d, padding_idx=0)
        self.body = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.heads = nn.ModuleDict()
        for name, (kind, options) in questions.items():
            if kind == "choice":
                out = len(options)
            elif kind == "noul":
                out = 3
            else:
                out = 1
            self.heads[name] = nn.Linear(d, out)

    def encode(self, text_ids):
        x = self.emb(text_ids)
        return self.body(x.mean(dim=0))

    def forward(self, text_ids):
        z = self.encode(text_ids)
        out = {}
        for name in self.qnames:
            kind, options = self.qmeta[name]
            logits = self.heads[name](z)
            if kind == "score":
                out[name] = (float(torch.sigmoid(logits)[0]), None)
            else:
                probs = torch.softmax(logits, dim=-1)
                conf, idx = float(probs.max()), int(probs.argmax())
                if kind == "choice":
                    out[name] = (options[idx], conf)
                else:
                    out[name] = ({0: NOUL_YES, 1: NOUL_NO,
                                  2: NOUL_UNKNOWN}[idx], conf)
        return out

    def decide(self, name, text_ids):
        """Single-question convenience returning (answer, confidence)."""
        return self.forward(text_ids)[name]

    # --- training ---------------------------------------------------------

    def _targets(self, records):
        """records: list of {ids, answers: {name: answer_str}}."""
        zs, labels = [], {n: [] for n in self.qnames}
        for r in records:
            with torch.no_grad():
                zs.append(self.encode(r["ids"]))
            for n in self.qnames:
                kind, options = self.qmeta[n]
                a = r["answers"].get(n)
                if kind == "choice":
                    labels[n].append(options.index(a))
                elif kind == "noul":
                    labels[n].append(_NOUL_IDX[a])
                else:
                    labels[n].append(float(a))
        return torch.stack(zs), {n: torch.tensor(v) for n, v in labels.items()}

    def fit(self, records, epochs=200, lr=1e-2, verbose=False):
        """CE on answers + Brier(conf, target).

        FINDING (2026-09-28, caught by T28 A/B): with target == train
        label correctness, the Brier term is REDUNDANT with CE — both are
        proper scoring rules converging to the posterior on iid data
        (identical confidences observed). The term only bites when
        `target` is an OUTCOME signal that differs from the label —
        exactly the RLCD setting (P2): records carry a replay-verified
        `target_conf` (e.g. env.verify result under the routed action),
        which the softmax cannot memorize away. Default target = train
        correctness (documented no-op)."""
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        zs, labels = self._targets(records)
        for ep in range(epochs):
            opt.zero_grad()
            loss = 0.0
            for n in self.qnames:
                kind, _ = self.qmeta[n]
                logits = self.heads[n](zs)
                if kind == "score":
                    pred = torch.sigmoid(logits)[:, 0]
                    loss = loss + torch.nn.functional.mse_loss(pred, labels[n])
                else:
                    loss = loss + torch.nn.functional.cross_entropy(
                        logits, labels[n])
                    with torch.no_grad():
                        correct = (logits.argmax(-1) == labels[n]).float()
                        tgt = torch.stack([torch.as_tensor(
                            r["answers"].get(f"{n}__target_conf",
                                             float(c)), dtype=torch.float32)
                            for r, c in zip(records, correct)])
                    conf = torch.softmax(logits, dim=-1).max(dim=-1).values
                    loss = loss + self.brier_lambda * torch.mean(
                        (conf - tgt) ** 2)
            loss.backward()
            opt.step()
            if verbose and ep % 50 == 0:
                print(f"  ep{ep} loss {float(loss):.4f}", flush=True)
        return self

    def evaluate(self, records):
        """Returns (accuracy_per_question, ece_per_question)."""
        zs, labels = self._targets(records)
        acc, ece = {}, {}
        for n in self.qnames:
            kind, _ = self.qmeta[n]
            logits = self.heads[n](zs)
            if kind == "score":
                pred = torch.sigmoid(logits)[:, 0]
                acc[n] = float((pred.round() == labels[n].round()).float().mean())
                ece[n] = None
                continue
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            correct = (pred == labels[n]).float()
            acc[n] = float(correct.mean())
            # ECE with 10 bins
            e = 0.0
            for lo in torch.linspace(0, 0.9, 10):
                m = (conf >= lo) & (conf < lo + 0.1)
                if m.any():
                    e += float(m.float().mean() * (conf[m].mean()
                                                   - correct[m].mean()).abs())
            ece[n] = e
        return acc, ece

"""lm-evaluation-harness adapter for HeliosLM (plan E1).

Registers the model type ``helioslm`` with lm-eval, so any harness task
(ARC, HellaSwag, MMLU, TMMLU+, ...) can score a pilot checkpoint:

    lm_eval --model helioslm \\
        --model_args run=runs/dense,data=data/pilot,device=cuda \\
        --tasks arc_easy,hellaswag --batch_size 1

or from Python with an already-built model and tokenizer:

    HeliosHarnessLM(model=model, tokenizer=tok, max_length=1024)

``run`` is a ``pilot_100m.py`` output directory (``run.json`` + ``ckpt.pt``);
the tokenizer is ``<data>/tokenizer.json`` from ``pilot_prepare_data.py``.

The three request types follow lm-eval's HF implementation:

  loglikelihood          sum of log p(continuation | context) and whether
                         every continuation token is the greedy choice; the
                         context is cut from the left to fit max_length.
  loglikelihood_rolling  log-likelihood of a whole document in disjoint
                         windows of max_length (lm_eval.utils rolling windows),
                         each window conditioned on the tokens before it.
  generate_until         greedy decoding until a stop string or max_gen_toks.

Scoring is exact float32 log-softmax of the model's logits; requests are
scored one at a time (correctness first; a 100M model is fast enough).
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    from lm_eval.utils import get_rolling_token_windows, make_disjoint_window
    HAVE_LM_EVAL = True
except ImportError:      # optional dependency: the module still imports
    HAVE_LM_EVAL = False
    LM = object

    def register_model(name):
        return lambda cls: cls


def load_pilot(run, data=None, device="cpu", checkpoint="ckpt.pt"):
    """(model, tokenizer, max_length) from a pilot_100m.py run directory."""
    from tokenizers import Tokenizer
    from helioslm_v5.src.model_v5 import HeliosLMv5
    from helioslm_v5.src.training.pretrain import pilot_config
    run = Path(run)
    meta = json.loads((run / "run.json").read_text())
    data = Path(data or meta["data"])
    data_meta = json.loads((data / "meta.json").read_text())
    if data_meta["tokenizer_sha256"] != meta["tokenizer_sha256"]:
        raise ValueError("tokenizer in --data is not the one this run trained on")
    cfg = pilot_config(meta["arch"], vocab_size=data_meta["vocab_size"],
                       hidden_size=meta["hidden"], num_layers=meta["layers"],
                       intermediate_size=meta["intermediate"],
                       max_seq=meta["seq"], balance=meta["balance"],
                       moe_dispatch=meta.get("moe_dispatch", "blocked"),
                       kda_chunk=meta.get("kda_chunk", 0))
    model = HeliosLMv5(cfg)
    state = torch.load(run / checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    tok = Tokenizer.from_file(str(data / "tokenizer.json"))
    return model.to(device).eval(), tok, meta["seq"]


@register_model("helioslm")
class HeliosHarnessLM(LM):
    def __init__(self, model=None, tokenizer=None, max_length=1024, run=None,
                 data=None, device="cpu", checkpoint="ckpt.pt", eot_token=None,
                 batch_size=1, **kwargs):
        if not HAVE_LM_EVAL:
            raise ImportError("HeliosHarnessLM needs lm-evaluation-harness: "
                              "pip install lm-eval")
        super().__init__()
        if model is None:
            if run is None:
                raise ValueError("pass model= and tokenizer=, or run=")
            model, tokenizer, max_length = load_pilot(run, data, device, checkpoint)
        self.model = model.eval()
        self.tok = tokenizer
        self.max_length = int(max_length)
        self._device = next(model.parameters()).device
        eot = eot_token if eot_token is not None else tokenizer.token_to_id("<|endoftext|>")
        self.eot_token_id = eot if eot is not None else tokenizer.get_vocab_size() - 1

    @property
    def device(self):
        return self._device

    @property
    def tokenizer_name(self):
        return "helioslm-bpe"

    # ------------------------------------------------------------ helpers
    def tok_encode(self, text):
        return self.tok.encode(text).ids

    def tok_decode(self, ids):
        return self.tok.decode(list(ids))

    @torch.no_grad()
    def _logprobs(self, ids):
        """log-softmax over the vocabulary at every position of ids (1-D)."""
        x = torch.tensor([ids], dtype=torch.long, device=self._device)
        logits = self.model(x)[0][0].float()
        return F.log_softmax(logits, dim=-1)

    def _score(self, context_ids, cont_ids):
        """(sum log p(cont | context), all-greedy) with left truncation."""
        if not cont_ids:
            return 0.0, True
        ids = (context_ids + cont_ids)[-(self.max_length + 1):]
        logp = self._logprobs(ids[:-1])
        n = len(cont_ids)
        target = torch.tensor(ids[-n:], device=self._device)
        lp = logp[-n:]
        total = lp.gather(-1, target[:, None]).sum().item()
        greedy = bool((lp.argmax(-1) == target).all().item())
        return total, greedy

    # ------------------------------------------------------------ lm-eval API
    def loglikelihood(self, requests, disable_tqdm=False):
        out = []
        for req in requests:
            context, continuation = req.args
            if context == "":
                ctx_ids = [self.eot_token_id]
                cont_ids = self.tok_encode(continuation)
            else:
                # split on the joined string so a leading space stays with
                # the continuation, as lm-eval's HF model does
                n_space = len(context) - len(context.rstrip())
                if n_space:
                    continuation = context[-n_space:] + continuation
                    context = context[:-n_space]
                whole = self.tok_encode(context + continuation)
                ctx_ids = self.tok_encode(context)
                cont_ids = whole[len(ctx_ids):]
            out.append(self._score(ctx_ids, cont_ids))
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        out = []
        for req in requests:
            (text,) = req.args
            windows = map(make_disjoint_window, get_rolling_token_windows(
                token_list=self.tok_encode(text), prefix_token=self.eot_token_id,
                max_seq_len=self.max_length, context_len=1))
            out.append(sum(self._score(list(c), list(t))[0] for c, t in windows))
        return out

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm=False):
        out = []
        for req in requests:
            context, kwargs = req.args
            until = kwargs.get("until") or []
            if isinstance(until, str):
                until = [until]
            max_new = int(kwargs.get("max_gen_toks", 256))
            ids = self.tok_encode(context)[-(self.max_length - max_new):] or [self.eot_token_id]
            gen = []
            for _ in range(max_new):
                logp = self._logprobs(ids + gen)
                nxt = int(logp[-1].argmax())
                if nxt == self.eot_token_id:
                    break
                gen.append(nxt)
                text = self.tok_decode(gen)
                if any(s in text for s in until):
                    break
            text = self.tok_decode(gen)
            for s in until:
                text = text.split(s)[0]
            out.append(text)
        return out

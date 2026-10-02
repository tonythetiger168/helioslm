"""Shared harness for the mid eval/training scripts (v5.37).

The three agentic evals (eval_mid_agent_v2/v4/v5) grew identical
preambles: paired-tokenizer load + checkpoint load + a model_fn closure
capturing per-call softmax max confidence. One copy now lives here;
the scripts pass max_new/FEWSHOT and get (model, tok, make_fn).
"""
import os
from pathlib import Path

import torch


def load_model_tok(device=None, ckpt_dir=None):
    """Paired artifact loading (v5.33 P3-P5 discipline): the tokenizer
    travels WITH the checkpoint; never retrain at eval time."""
    from helioslm_v5.configs.config_v5 import HeliosLMv5Config
    from helioslm_v5.src.model_v5 import HeliosLMv5
    from helioslm_v5.src.tokenizer.bpe import BOS, HeliosBPE
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = Path(ckpt_dir or os.environ.get("HELIOS_CKPT_DIR", "checkpoints"))
    tok = HeliosBPE.load(str(ck / "mid_sft_v5.33.tok.json"))
    print("loaded paired tokenizer from checkpoint", flush=True)
    model = HeliosLMv5(HeliosLMv5Config(size="mid")).to(device)
    model.load_state_dict(torch.load(ck / "mid_sft_v5.33.pt",
                                     map_location=device))
    model.eval()
    return model, tok, device


def make_model_fn(model, tok, device, max_new=128, fewshot="",
                  state=None):
    """Greedy decode closure; records softmax-max confidence per call in
    state['conf'] (a dict the caller owns)."""
    from helioslm_v5.src.tokenizer.bpe import BOS
    bos = tok.vocab[BOS]

    def model_fn(prompt, seed, step):
        ids = [bos] + tok.encode(fewshot + prompt)
        input_ids = torch.tensor([ids]).to(device)
        with torch.no_grad():
            for _ in range(max_new):
                logits, _, _ = model(input_ids,
                                     attention_mask=torch.ones_like(input_ids))
                probs = torch.softmax(logits[0, -1], dim=-1)
                conf, nxt = float(probs.max()), int(probs.argmax())
                if nxt == tok.vocab["<eos>"]:
                    break
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[nxt]]).to(device)], dim=1)
        state["conf"] = conf
        return tok.decode(input_ids[0, len(ids):].tolist())

    return model_fn

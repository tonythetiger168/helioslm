"""alignbench_mid_encode.py - encode AlignBench states with mid (360M HeliosLM) hidden states.

Prereqs (local):
  1. checkpoints/mid_sft_v5.33.pt + checkpoints/mid_sft_v5.33.tok.json
     (from https://huggingface.co/chienhsinlin/helioslm -- paired artifacts)
  2. all.jsonl: RLCDAlignBench data (gated)
  3. GPU or CPU (~15 min GPU / ~1.5h CPU)

Usage:
  python examples/alignbench_mid_encode.py
  -> writes mid_feats.npz
"""
import json, sys, time
import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.tokenizer.bpe import HeliosBPE

items = [json.loads(l) for l in open("all.jsonl", encoding="utf-8") if l.strip()]
SUBSET = ["sycophancy_eval", "faith_mt_grounded", "confaide", "injecagent",
          "open_prompt_injection", "abstentionbench", "machiavelli_reward",
          "instrumental_eval", "sycon_fp", "privaci_bench",
          "verbalized_ece", "harmbench", "reward_hacking_freeform",
          "bias_scene_heldout", "paiq", "tensor_trust_hijack"]

def render(it):
    st = json.loads(it["state"])
    return "\n".join(f"[{k}] {str(v)}" for k, v in sorted(st.items()))

keep = [(i, it) for i, it in enumerate(items) if it["benchmark"] in SUBSET]
print(f"subset: {len(keep)} states", flush=True)

tok = HeliosBPE.load("checkpoints/mid_sft_v5.33.tok.json")
cfg = HeliosLMv5Config(size="mid")
model = HeliosLMv5(cfg)
model.load_state_dict(torch.load("checkpoints/mid_sft_v5.33.pt",
                                 map_location="cpu"))
model.eval()

captured = {}
def hook(mod, inp, out):
    captured["h"] = out[0] if isinstance(out, tuple) else out
model.layers[-1].register_forward_hook(hook)

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(dev)
print(f"device: {dev}", flush=True)

FEAT = cfg.hidden_size
feats = np.zeros((len(keep), FEAT), dtype=np.float32)
labels = np.zeros(len(keep), dtype=np.int8)
bench = []
t0 = time.time()
with torch.no_grad():
    for j, (i, it) in enumerate(keep):
        ids = torch.tensor([tok.encode(render(it))[:384]]).to(dev)
        model(ids, attention_mask=torch.ones_like(ids))
        feats[j] = captured["h"][0, :ids.shape[1]].float().mean(0).cpu().numpy()
        labels[j] = int(it["label"])
        bench.append(it["benchmark"])
        if j % 200 == 0:
            el = time.time() - t0
            print(f"{j}/{len(keep)} {el:.0f}s", flush=True)
np.savez("mid_feats.npz", feats=feats, labels=labels, bench=np.array(bench))
print("MID_ENCODE_DONE", flush=True)

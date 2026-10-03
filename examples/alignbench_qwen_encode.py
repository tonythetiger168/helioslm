"""alignbench_qwen_encode.py - encode AlignBench states with Qwen3-0.6B hidden states (GPU).

Uses transformers (AutoModel) -- the hand-rolled runner from the sandbox
benchmark is not in this repo, and the official path is cleaner anyway.

Prereqs (local):
  1. Qwen3-0.6B weights via huggingface-cli or git-lfs into ./qwen/
     (config.json / tokenizer.json / model.safetensors)
  2. all.jsonl (RLCDAlignBench, gated)
  3. GPU: ~5 min on RTX 4060

Usage:
  python examples/alignbench_qwen_encode.py
  -> writes qwen_feats.npz
"""
import json, time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

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

tok = AutoTokenizer.from_pretrained("qwen")
model = AutoModel.from_pretrained("qwen").eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(dev)
print(f"device: {dev}", flush=True)

FEAT = model.config.hidden_size
feats = np.zeros((len(keep), FEAT), dtype=np.float32)
labels = np.zeros(len(keep), dtype=np.int8)
bench = []
t0 = time.time()
with torch.no_grad():
    for j, (i, it) in enumerate(keep):
        enc = tok(render(it), return_tensors="pt", truncation=True,
                  max_length=384).to(dev)
        out = model(**enc)
        n = enc["attention_mask"].sum().item()
        feats[j] = out.last_hidden_state[0, :n].float().mean(0).cpu().numpy()
        labels[j] = int(it["label"])
        bench.append(it["benchmark"])
        if j % 200 == 0:
            el = time.time() - t0
            print(f"{j}/{len(keep)} {el:.0f}s", flush=True)
np.savez("qwen_feats.npz", feats=feats, labels=labels, bench=np.array(bench))
print("QWEN_ENCODE_DONE", flush=True)

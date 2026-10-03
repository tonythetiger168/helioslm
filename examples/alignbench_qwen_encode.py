"""alignbench_qwen_encode.py - encode AlignBench states with Qwen3-0.6B hidden states (GPU).

Prereqs (local):
  1. Qwen3-0.6B weights: config.json / tokenizer.json / model.safetensors
     in ./qwen/  (from https://huggingface.co/Qwen/Qwen3-0.6B)
  2. all.jsonl: RLCDAlignBench data (gated; request access at
     https://huggingface.co/datasets/sumleo/RLCDAlignBench)
  3. GPU: ~5 min on RTX 4060; ~2h on CPU

Usage:
  python examples/alignbench_qwen_encode.py
  -> writes qwen_feats.npz  (upload to the sandbox or feed to alignbench_readout.py)
"""
import json, sys, time
import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from qwen_runner import load_safetensors, BPETokenizer, Qwen3   # noqa: E402

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

cfg = json.load(open("qwen/config.json"))
w = load_safetensors("qwen/model.safetensors")
tok = BPETokenizer("qwen/tokenizer.json")
model = Qwen3(w, cfg)

FEAT = cfg["hidden_size"]
feats = np.zeros((len(keep), FEAT), dtype=np.float32)
labels = np.zeros(len(keep), dtype=np.int8)
bench = []
t0 = time.time()
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev}", flush=True)
with torch.no_grad():
    for j, (i, it) in enumerate(keep):
        ids = tok.encode(render(it))[:384]
        model.forward(ids)
        feats[j] = model.last_hidden[:len(ids)].float().mean(0).numpy()
        labels[j] = int(it["label"])
        bench.append(it["benchmark"])
        if j % 200 == 0:
            el = time.time() - t0
            print(f"{j}/{len(keep)} {el:.0f}s", flush=True)
np.savez("qwen_feats.npz", feats=feats, labels=labels, bench=np.array(bench))
print("QWEN_ENCODE_DONE", flush=True)

"""T3b: calibrate sampling-noise baseline, then isolate the MTP sampling bug."""
import sys
sys.path.insert(0, "/mnt/agents/output")
import torch, random
from collections import Counter
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5

torch.manual_seed(0)
cfg = HeliosLMv5Config(size="lite")
model = HeliosLMv5(cfg).eval()
PLEN = 4
PROMPT = torch.tensor([[11, 22, 33, 44]])

def collect(use_mtp, n, seed0):
    c = [Counter() for _ in range(3)]
    used = 0
    for s in range(n):
        torch.manual_seed(seed0 + s)
        random.seed(seed0 + s)
        out = model.generate(PROMPT, max_new_tokens=3, temperature=1.0, top_p=1.0,
                             use_mtp=use_mtp)
        if out.shape[1] < PLEN + 3:
            continue
        used += 1
        for j in range(3):
            c[j][int(out[0, PLEN + j])] += 1
    return c, used

def tv(a, b, n):
    keys = set(a) | set(b)
    return sum(abs(a[k] - b[k]) for k in keys) / (2 * n)

N = 600
# noise floor: two independent non-MTP runs
c1, u1 = collect(False, N, 10_000)
c2, u2 = collect(False, N, 20_000)
print("non-MTP vs non-MTP (noise floor):")
for j in range(3):
    print(f"  pos {j}: TV = {tv(c1[j], c2[j], min(u1,u2)):.4f}")

c3, u3 = collect(True, N, 10_000)   # same seeds as c1
print(f"MTP vs non-MTP same seeds (used ref={u1}, mtp={u3}):")
for j in range(3):
    print(f"  pos {j}: TV = {tv(c1[j], c3[j], min(u1,u3)):.4f}")

# top-token frequency comparison at pos 1
top_ref = c1[1].most_common(5)
print("ref pos1 top5:", [(k, v/u1) for k, v in top_ref])
print("mtp pos1 top5:", [(k, v/u3) for k, v in c3[1].most_common(5)])

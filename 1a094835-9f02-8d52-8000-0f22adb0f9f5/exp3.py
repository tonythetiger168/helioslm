import sys, torch
sys.path.insert(0, "/mnt/agents/output")
torch.manual_seed(0)
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.attention.mla import MLA

cfg = HeliosLMv5Config(size="lite")
for mode in ("absorbed", "expanded"):
    cfg.attention.use_absorption = (mode == "absorbed")
    mla = MLA(cfg).eval()
    L = 6
    h = torch.randn(1, L, cfg.hidden_size)
    pos = torch.tensor([[0,1,2,0,1,2]])
    with torch.no_grad():
        out_packed, _ = mla(h, position_ids=pos)
        # solo reference for doc B (idx 3,4,5) with positions 0,1,2
        out_soloB, _ = mla(h[:, 3:6], position_ids=torch.tensor([[0,1,2]]))
        # perturb doc A only; doc B outputs should be invariant if mask is correct
        h2 = h.clone(); h2[:, :3] += 10.0
        out_packed2, _ = mla(h2, position_ids=pos)
    leak = (out_packed[0,3:] - out_packed2[0,3:]).abs().max().item()
    d_solo = (out_packed[0,3:] - out_soloB[0]).abs().max().item()
    print(f"{mode}: docB output change when docA perturbed = {leak:.3e} (0 expected); "
          f"packed-vs-solo diff = {d_solo:.3e}")

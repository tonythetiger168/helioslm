import sys, torch
sys.path.insert(0, "/mnt/agents/output")
torch.manual_seed(0)
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.vision.navit import NaViTEncoder

cfg = HeliosLMv5Config(size="lite")
cfg.multimodal.enabled = True
cfg.multimodal.vision_hidden_size = 64
cfg.multimodal.vision_num_layers = 1
cfg.multimodal.vision_num_heads = 4
enc = NaViTEncoder(cfg).eval()

# tiny image smaller than patch size
try:
    enc(torch.randn(1, 3, 7, 7))
    print("tiny image: no error??")
except Exception as e:
    print("tiny image raises:", type(e).__name__, str(e)[:100])
# non-divisible size
try:
    out = enc(torch.randn(1, 3, 30, 30))
    print("30x30 (non-divisible by 14):", tuple(out.shape), "(floor to 2x2 grid; silent)")
except Exception as e:
    print("30x30 raises:", type(e).__name__)
# 0-size batch
try:
    out = enc(torch.randn(0, 3, 28, 28))
    print("B=0:", tuple(out.shape))
except Exception as e:
    print("B=0 raises:", type(e).__name__, str(e)[:80])
# forward_packed with empty list
try:
    enc.forward_packed([])
    print("empty packed: no error??")
except Exception as e:
    print("forward_packed([]) raises:", type(e).__name__)

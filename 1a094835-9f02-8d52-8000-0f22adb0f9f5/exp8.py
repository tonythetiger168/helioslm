import sys, torch
sys.path.insert(0, "/mnt/agents/output")
torch.manual_seed(0)
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
import helioslm_v5.src.model_v5 as m5

cfg = HeliosLMv5Config(size="lite")
model = m5.HeliosLMv5(cfg)
model.train()
assert model.training
ids = torch.randint(3, cfg.vocab_size, (1, 5))
model.generate(ids, max_new_tokens=3)
print("generate() leaves model.training =", model.training, "(v5.3 known limitation: not restored)")

# MTPDecoder.generate eval restore
from helioslm_v5.src.inference.mtp import MTPDecoder
model.train()
dec = MTPDecoder(model, model.mtp_modules, cfg)
dec.generate(ids, max_new_tokens=4)
print("MTPDecoder.generate leaves model.training =", model.training)

# engine LRU
from helioslm_v5.src.inference.vllm_engine import VLLMEngine
eng = VLLMEngine(model, cfg, max_num_blocks=2000)
eng._pad_caches_max = 4
for n in range(1, 8):
    eng._pad_prefix_cache(n, torch.device("cpu"))
print("pad cache size after 7 inserts (cap 4):", len(eng._pad_caches), "keys:", sorted(eng._pad_caches))
# LRU refresh: touch oldest surviving
eng._pad_prefix_cache(4, torch.device("cpu"))
eng._pad_prefix_cache(9, torch.device("cpu"))
print("after touching 4 then inserting 9, evicted key should be 5:", sorted(eng._pad_caches))

# engine block accounting includes pad prefix (M-I3)
eng2 = VLLMEngine(model, cfg, max_num_blocks=1000)
r1 = eng2.add_request([1,2,3], max_new_tokens=2)
r2 = eng2.add_request([1,2,3,4,5,6,7,8], max_new_tokens=2)
eng2.schedule()
reqs = {r.request_id: r for r in eng2.running_requests}
bm = eng2.block_manager
for rid, r in reqs.items():
    print(f"req{rid}: prompt={len(r.prompt_token_ids)} pad={r.prompt_pad} table_tokens={bm.get_context_length(rid)}")

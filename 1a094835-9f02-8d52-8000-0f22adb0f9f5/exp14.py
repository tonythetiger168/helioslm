import sys, torch
sys.path.insert(0, "/mnt/agents/output")
torch.manual_seed(0)
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
import helioslm_v5.src.model_v5 as m5
from helioslm_v5.src.inference.vllm_engine import VLLMEngine

cfg = HeliosLMv5Config(size="lite")
model = m5.HeliosLMv5(cfg).eval()
eng = VLLMEngine(model, cfg, max_num_blocks=100)

# check whether a request's cache shares storage with the memoized pad cache
pad_past = eng._pad_prefix_cache(4, torch.device("cpu"))
rid = eng.add_request([5,6,7], max_new_tokens=2)
eng.schedule()
req = eng.running_requests[0]
req.prompt_pad = 4
eng._prefill(req, torch.device("cpu"))
memo_t = eng._pad_caches[4][0][0]
req_t = req.past_key_values[0][0]
print("memo vs request cache share storage:", memo_t.data_ptr() == req_t.data_ptr(),
      "| same object:", memo_t is req_t)
# mutate request cache; does memo entry change?
before = memo_t.clone()
req_t.mul_(0)
print("memo entry affected by request mutation:", not torch.equal(before, memo_t))

# MTPModule.generate leaves module in eval
from helioslm_v5.src.inference.mtp import MTPModule
m = MTPModule(cfg, 0)
m.train()
h = torch.randn(1, 1, cfg.hidden_size)
tok = torch.randint(0, cfg.vocab_size, (1, 1))
m.generate(h, tok)
print("MTPModule.generate leaves training =", m.training)

import sys, torch
sys.path.insert(0, "/mnt/agents/output")
torch.manual_seed(0)
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
import helioslm_v5.src.model_v5 as m5
from helioslm_v5.src.inference.vllm_engine import VLLMEngine

cfg = HeliosLMv5Config(size="lite")
model = m5.HeliosLMv5(cfg).eval()

# Engine: exception during forward -> blocks leak?
eng = VLLMEngine(model, cfg, max_num_blocks=100)
rid = eng.add_request([1,2,3,4], max_new_tokens=4)
eng.schedule()
free0 = eng.block_manager.num_free_blocks()

orig = model.forward
def boom(*a, **k):
    raise RuntimeError("simulated OOM")
model.forward = boom
try:
    eng.step()
except RuntimeError as e:
    print("step raised:", e)
model.forward = orig
free1 = eng.block_manager.num_free_blocks()
print(f"blocks free before={free0} after_exception={free1} | running={len(eng.running_requests)}")
# blocks still owned by the request (not lost from ledger) but request never retried/removed:
# subsequent successful step continues
out = eng.run(max_steps=6)
print("recovery run outputs:", out, "| free after finish:", eng.block_manager.num_free_blocks())

"""T4: VLLMEngine boundary tests."""
import sys
sys.path.insert(0, "/mnt/agents/output")
import torch
from helioslm_v5.configs.config_v5 import HeliosLMv5Config
from helioslm_v5.src.model_v5 import HeliosLMv5
from helioslm_v5.src.inference.vllm_engine import VLLMEngine

torch.manual_seed(0)
cfg = HeliosLMv5Config(size="lite")
model = HeliosLMv5(cfg).eval()

def fresh():
    return VLLMEngine(model, cfg, block_size=4, max_num_blocks=2000)

def banner(s): print(f"\n=== {s} ===")

banner("1. single request")
eng = fresh()
out = eng.generate([[3, 4, 5]], max_new_tokens=8, temperature=0.0)
ref = model.generate(torch.tensor([[3, 4, 5]]), max_new_tokens=8, temperature=0.0)
print("engine:", out[0], "\nref:   ", ref[0, 3:].tolist(), "MATCH:", out[0] == ref[0, 3:].tolist())

banner("2. eight unequal-length prompts, greedy == solo")
torch.manual_seed(5)
prompts = [torch.randint(0, 1024, (L,)).tolist() for L in (1, 2, 3, 5, 8, 13, 21, 4)]
eng = fresh()
outs = eng.generate(prompts, max_new_tokens=10, temperature=0.0)
allok = True
for p, o in zip(prompts, outs):
    ref = model.generate(torch.tensor([p]), max_new_tokens=10, temperature=0.0)[0, len(p):].tolist()
    if o != ref:
        allok = False
        print(f"  MISMATCH prompt_len={len(p)}: engine {o} vs ref {ref}")
print("all 8 rows match solo greedy:", allok)

banner("3. max_new_tokens=0")
eng = fresh()
rid = eng.add_request([1, 2, 3], max_new_tokens=0)
res = eng.run()
print("run() result for max_new=0:", res, "(expect empty list)")

# manual step() usage
eng = fresh()
rid = eng.add_request([1, 2, 3], max_new_tokens=0)
outs = eng.step()
print("step() once with max_new=0: outputs =", outs, "(expect {} — no token should be generated)")
req = eng.finished_requests[0] if eng.finished_requests else None
print("finished generated_token_ids:", req.generated_token_ids if req else "not finished")

banner("4. long prompt joins mid-run (watermark growth)")
torch.manual_seed(6)
eng = fresh()
r1 = eng.add_request([7, 8, 9], max_new_tokens=12, temperature=0.0)
eng.step(); eng.step()   # r1 running, cache_len=4
long_prompt = torch.randint(0, 1024, (40,)).tolist()
r2 = eng.add_request(long_prompt, max_new_tokens=12, temperature=0.0)
res = eng.run()
ref1 = model.generate(torch.tensor([[7, 8, 9]]), max_new_tokens=12, temperature=0.0)[0, 3:].tolist()
ref2 = model.generate(torch.tensor([long_prompt]), max_new_tokens=12, temperature=0.0)[0, 40:].tolist()
print("r1 match:", res[r1] == ref1, " r2 match:", res[r2] == ref2)

banner("5. two waves (run -> add -> run)")
eng = fresh()
o1 = eng.generate([[1, 2]], max_new_tokens=5, temperature=0.0)
o2 = eng.generate([[3, 4, 5, 6]], max_new_tokens=5, temperature=0.0)
r1 = model.generate(torch.tensor([[1, 2]]), max_new_tokens=5, temperature=0.0)[0, 2:].tolist()
r2 = model.generate(torch.tensor([[3, 4, 5, 6]]), max_new_tokens=5, temperature=0.0)[0, 4:].tolist()
print("wave1 match:", o1[0] == r1, " wave2 match:", o2[0] == r2)
print("blocks freed after runs:", eng.block_manager.num_free_blocks(), "of", eng.block_manager.num_blocks)

banner("6. all requests EOS immediately (forced)")
orig_forward = model.forward
def forced_forward(input_ids, **kw):
    logits, h, past = orig_forward(input_ids, **kw)
    logits = logits.clone()
    logits[..., cfg.eos_token_id] = 1e6
    return logits, h, past
model.forward = forced_forward
eng = fresh()
outs = eng.generate([[1, 2, 3], [4, 5], [6]], max_new_tokens=10, temperature=0.0)
print("outputs:", outs, "(expect each == [EOS])")
print("all == [EOS]:", all(o == [cfg.eos_token_id] for o in outs))
model.forward = orig_forward

banner("7. block accounting: no leak after mixed run")
eng = fresh()
outs = eng.generate(prompts, max_new_tokens=10, temperature=0.0)
print("free blocks:", eng.block_manager.num_free_blocks(), "/", eng.block_manager.num_blocks,
      "LEAK" if eng.block_manager.num_free_blocks() != eng.block_manager.num_blocks else "OK")

banner("8. engine max_batch_size=1 serializes correctly")
eng = VLLMEngine(model, cfg, block_size=4, max_num_blocks=2000, max_batch_size=1)
outs = eng.generate([[1, 2, 3], [9, 9]], max_new_tokens=6, temperature=0.0)
r1 = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=6, temperature=0.0)[0, 3:].tolist()
r2 = model.generate(torch.tensor([[9, 9]]), max_new_tokens=6, temperature=0.0)[0, 2:].tolist()
print("serial match:", outs[0] == r1 and outs[1] == r2)

banner("9. empty prompt to engine")
eng = fresh()
try:
    outs = eng.generate([[]], max_new_tokens=3)
    print("empty prompt ->", outs)
except Exception as e:
    print(f"empty prompt EXC: {type(e).__name__}: {e}")
print("DONE")

import sys, types, torch
sys.path.insert(0, "/mnt/agents/output")
from helioslm_v5.src.training.grpo import GRPOTrainer
class Dummy:
    _extract_final_answer = GRPOTrainer._extract_final_answer  # classmethod
    _normalize_answer = staticmethod(GRPOTrainer._normalize_answer)
chk = types.MethodType(GRPOTrainer._check_correctness, Dummy())
cases = [("The answer is 25","2",False),("The answer is 2","2",True),
         (r"... \boxed{42}","42",True),(r"... \boxed{42}","43",False),
         ("result: 42.0","42",True),("<think>blah 99</think> Final: 7","7",True),
         ("","2",False),("Answer: $1,000.","1000",True),
         ("thinking 25 then 3","3",True), ("x = -4","-4",True),
         ("Answer is 2 but wait 25","25",True)]
for resp, gold, expect in cases:
    got = chk(resp, gold)
    print("OK" if got==expect else "FAIL", repr(resp), gold, "->", got)

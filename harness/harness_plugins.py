"""helios-harness plugins - snap existing modules into the Context."""
import sys
_AGENT = __file__.rsplit("/", 2)[0] + "/helioslm_v5/agent"
if _AGENT not in sys.path:
    sys.path.insert(0, _AGENT)

class ModelPlugin:
    def __init__(self, name, model_fn):
        self.name, self.model_fn = name, model_fn
    def apply(self, ctx):
        ctx.register(f"model.{self.name}", self.model_fn)

class ToolPlugin:
    def apply(self, ctx):
        from tools import build_default_registry
        reg, impls = build_default_registry()
        ctx.register("tools.registry", reg)
        ctx.register("tools.impls", impls)

class SessionPlugin:
    def apply(self, ctx):
        from trajectory import Trajectory
        ctx.register("session.new", lambda: Trajectory(
            task="", seed=0, steps=[], final_answer=None))
        ctx.effect("session_store", {"note": "append-only"})

class DecisionPlugin:
    def __init__(self, gate=None, prefetcher=None):
        self.gate, self.prefetcher = gate, prefetcher
    def apply(self, ctx):
        from gate import FixedGate, Route
        from grounding import GroundingGate
        from trust_gate import TrustGate
        from decision_head import DecisionHead
        base = self.gate or FixedGate(Route.DIRECT)
        ctx.register("decision.grounding", GroundingGate(base))
        ctx.register("decision.trust", TrustGate(
            DecisionHead({"trust": ("noul", None)}), inner=base))
        if self.prefetcher:
            ctx.register("decision.prefetcher", self.prefetcher)

class LoopPlugin:
    def apply(self, ctx):
        from loop import AgentLoop
        def make_loop(model_fn, max_steps=8):
            return AgentLoop(
                model_fn,
                ctx.tools.registry, ctx.tools.impls,
                ctx.decision.grounding,
                max_steps=max_steps)
        ctx.register("loop.make", make_loop)

class StatePlugin:
    """Workflow state: a dict-backed store for multi-step pipelines.
    Registered as ctx.state, accessible across plugin boundaries."""

    def apply(self, ctx):
        ctx.register("state", {})


class PresetPlugin:
    """Named presets wiring the standard plugin set.
    minimal: model+tools only. full: + session+decision+loop+state."""

    PRESETS = {"minimal", "full"}

    def __init__(self, name, model_fn=None):
        assert name in self.PRESETS
        self.name, self.model_fn = name, model_fn

    def apply(self, ctx):
        ctx.register("preset", self.name)
        if self.model_fn:
            ctx.use(ModelPlugin(self.name, self.model_fn))
        ctx.use(ToolPlugin())
        if self.name == "full":
            ctx.use(SessionPlugin())
            ctx.use(DecisionPlugin())
            ctx.use(LoopPlugin())
            ctx.use(StatePlugin())

class AgentWorkflowPlugin:
    """v1.2: ReAct workflow on the harness -- think -> act -> observe
    -> ... -> answer, with per-step effects (auditable). Unlike
    AgentLoop (tool-call only), this allows interleaved reasoning and
    action, and records an effect per step (not just the final one)."""

    def apply(self, ctx):
        def run(task_text, model_fn, max_steps=12, seed=0):
            ctx.state["workflow"] = {"task": task_text, "steps": [],
                                     "final": None}
            transcript = f"Task: {task_text}"
            for step in range(max_steps):
                raw = model_fn(transcript, seed, step)
                ctx.state["workflow"]["steps"].append({"raw": raw})
                ctx.effect(f"step_{step}", {"raw": raw[:80]},
                           replay_fn=lambda p: True)
                # parse: think (continue) / tool (execute) / text (answer)
                from schema import ToolCallError, parse_chat_turn
                try:
                    kind, payload = parse_chat_turn(raw, ctx.tools.registry)
                except ToolCallError:
                    transcript += f"\nPARSE_ERROR"
                    continue
                if kind == "text":
                    ctx.state["workflow"]["final"] = payload
                    break
                call = payload[0]
                from tools import execute
                if call.name not in ctx.tools.registry._specs:
                    obs = f"TOOL_ERROR: unknown tool {call.name}"
                else:
                    try:
                        obs = execute(call, ctx.tools.registry,
                                      ctx.tools.impls)
                    except ToolCallError as e:
                        obs = f"TOOL_ERROR: {e}"
                ctx.state["workflow"]["steps"][-1]["obs"] = obs
                transcript += f"\n{raw}\nstep {step}: {obs}"
            return ctx.state["workflow"]
        ctx.register("agent.run", run)

class MidModelPlugin:
    """v1.3: real model_fn from the mid 360M checkpoint. Loads paired
    tokenizer + weights (v5.33 pairing discipline). For harness runs
    where grounding handles content; model provides the tool sequence."""

    def __init__(self, ckpt_dir="checkpoints"):
        self.ckpt_dir = ckpt_dir

    def apply(self, ctx):
        import os, torch
        from helioslm_v5.configs.config_v5 import HeliosLMv5Config
        from helioslm_v5.src.model_v5 import HeliosLMv5
        from helioslm_v5.src.tokenizer.bpe import BOS, HeliosBPE
        ck = self.ckpt_dir
        tok = HeliosBPE.load(os.path.join(ck, "mid_sft_v5.33.tok.json"))
        model = HeliosLMv5(HeliosLMv5Config(size="mid"))
        model.load_state_dict(torch.load(
            os.path.join(ck, "mid_sft_v5.33.pt"), map_location="cpu"))
        model.eval()
        bos = tok.vocab[BOS]

        def model_fn(prompt, seed, step, max_new=96):
            ids = [bos] + tok.encode(prompt)
            input_ids = torch.tensor([ids])
            with torch.no_grad():
                for _ in range(max_new):
                    logits, _, _ = model(input_ids,
                                         attention_mask=torch.ones_like(input_ids))
                    probs = torch.softmax(logits[0, -1], dim=-1)
                    nxt = int(probs.argmax())
                    if nxt == tok.vocab["<eos>"]:
                        break
                    input_ids = torch.cat([input_ids,
                                           torch.tensor([[nxt]])], dim=1)
            return tok.decode(input_ids[0, len(ids):].tolist())

        ctx.register("model.mid", model_fn)

class QwenModelPlugin:
    """v1.4: Qwen3-0.6B as a harness model_fn. Uses transformers
    AutoModelForCausalLM (the hand-rolled runner stays for benchmarks).
    Qwen3-0.6B frozen scored 0.807 on AlignBench -- usable as a real
    decision-capable model in the harness."""

    def __init__(self, model_dir="qwen"):
        self.model_dir = model_dir

    def apply(self, ctx):
        import os
        if not os.path.exists(self.model_dir):
            ctx.register("model.qwen", None)
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_dir, torch_dtype=torch.bfloat16).eval()

        def model_fn(prompt, seed, step, max_new=96):
            enc = tok(prompt, return_tensors="pt", truncation=True,
                      max_length=900)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
            return tok.decode(out[0][enc["input_ids"].shape[1]:],
                              skip_special_tokens=True)

        ctx.register("model.qwen", model_fn)


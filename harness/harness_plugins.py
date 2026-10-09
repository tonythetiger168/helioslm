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

class MathPlugin:
    """v1.5: math augmentation -- symbolic verify + CoT prompt scaffold.
    Verifies numeric answers by re-eval (not string match), and prepends
    a chain-of-thought scaffold to the prompt. The scaffold is OPTIONAL:
    models that do not reason still work, models that do get a boost.
    """

    def apply(self, ctx):
        def verify_numeric(want, got, tol=1e-9):
            try:
                return abs(float(want) - float(got)) < tol
            except (ValueError, TypeError):
                return False
        ctx.register("math.verify", verify_numeric)
        ctx.register("math.scaffold",
                     "Let me solve this step by step.\n")


class MathWorkflowPlugin:
    """Math-specific workflow: uses math.verify for correctness instead
    of env.verify. For benchmarks where the answer is numeric and the
    env's string-match is too brittle."""

    def __init__(self, scaffold=True):
        self.scaffold = scaffold

    def apply(self, ctx):
        def run_math(task_text, model_fn, max_steps=12, seed=0):
            if self.scaffold:
                task_text = ctx.math.scaffold + task_text
            ctx.state["math"] = {"task": task_text, "results": []}
            wf = ctx.agent.run(task_text, model_fn, max_steps=max_steps,
                               seed=seed)
            final = wf["final"]
            ok = ctx.math.verify(
                task_text.split(":")[-1].strip(), final) if final else False
            ctx.state["math"]["results"].append(ok)
            ctx.effect("math_done", {"ok": ok}, replay_fn=lambda p: True)
            return {"workflow": wf, "numeric_ok": ok}
        ctx.register("math.run", run_math)

class CodePlugin:
    """v1.6: code execution + verification sandbox. Uses our existing
    file_env + tools (calc, str_op, file_read/write) as a minimal code
    execution environment. The model generates tool calls (code), the
    sandbox executes them, and file_env.verify checks the output.
    """

    def apply(self, ctx):
        def run_code(task_text, model_fn, max_steps=12, seed=0):
            ctx.state["code"] = {"task": task_text, "steps": []}
            from envs import make_long_envs
            from grounding import GroundingGate
            from gate import FixedGate, Route
            from envs.file_env import FileLongHorizonEnv
            env = FileLongHorizonEnv()
            task = env.sample(__import__("random").Random(seed))
            # grounding ensures args are correct even if model confabulates
            ctx.register("decision.grounding", GroundingGate(
                FixedGate(Route.DIRECT)))
            wf = ctx.agent.run(task.text, model_fn, max_steps=max_steps,
                               seed=seed)
            final = wf["final"]
            ok = env.verify(task, final) if final else False
            ctx.state["code"]["ok"] = ok
            ctx.effect("code_done", {"ok": ok}, replay_fn=lambda p: True)
            return {"workflow": wf, "file_env_ok": ok}
        ctx.register("code.run", run_code)

class ICPlugin:
    """v1.7: IC design + verification support. RTL generation (Verilog/
    SystemVerilog), syntax lint (no external simulator), UVM testbench
    scaffold, and coverage closure tracking. All through the harness
    workflow with per-step audit effects. NO simulator dependency --
    the lint is structural (module/port/always blocks), not simulation."""

    def apply(self, ctx):
        def gen_rtl(spec_text, model_fn, max_steps=8):
            ctx.state["ic"] = {"spec": spec_text, "rtl": None, "uvm": None}
            prompt = f"Generate synthesizable Verilog for: {spec_text}.\nOutput ONLY the module, no explanation."
            rtl = model_fn(prompt, 0, 0)
            ctx.state["ic"]["rtl"] = rtl
            ctx.effect("rtl_gen", {"rtl": rtl[:100]},
                       replay_fn=lambda p: True)
            return rtl

        def lint_rtl(rtl_text):
            errors = []
            if "module" not in rtl_text:
                errors.append("no module declaration")
            if "endmodule" not in rtl_text:
                errors.append("no endmodule")
            if rtl_text.count("(") != rtl_text.count(")"):
                errors.append("unbalanced parentheses")
            if rtl_text.count("begin") != rtl_text.count("end"):
                errors.append("unbalanced begin/end")
            return {"ok": len(errors) == 0, "errors": errors}

        def gen_uvm(rtl_text, model_fn):
            prompt = f"Generate a UVM testbench scaffold for:\n{rtl_text[:500]}\nOutput ONLY the class definitions."
            uvm = model_fn(prompt, 0, 0)
            ctx.state["ic"]["uvm"] = uvm
            ctx.effect("uvm_gen", {"uvm": uvm[:100]},
                       replay_fn=lambda p: True)
            return uvm

        def coverage(goal, hit):
            return {"goal": goal, "hit": hit,
                    "pct": round(100.0 * hit / goal, 1) if goal else 0.0}

        ctx.register("ic.gen_rtl", gen_rtl)
        ctx.register("ic.lint", lint_rtl)
        ctx.register("ic.gen_uvm", gen_uvm)
        ctx.register("ic.coverage", coverage)


class ICWorkflowPlugin:
    """IC workflow: spec -> RTL -> lint -> UVM -> coverage, all audited."""

    def apply(self, ctx):
        def run_ic(spec_text, model_fn, max_steps=12):
            ctx.state["ic_workflow"] = {"spec": spec_text}
            rtl = ctx.ic.gen_rtl(spec_text, model_fn)
            lint = ctx.ic.lint(rtl)
            uvm = ctx.ic.gen_uvm(rtl, model_fn) if lint["ok"] else None
            ctx.state["ic_workflow"].update({
                "rtl_ok": lint["ok"], "lint_errors": lint["errors"],
                "uvm": uvm is not None})
            ctx.effect("ic_done",
                       {"rtl_ok": lint["ok"], "has_uvm": uvm is not None},
                       replay_fn=lambda p: True)
            return {"rtl": rtl, "lint": lint, "uvm": uvm}
        ctx.register("ic.run", run_ic)

class MemoryPlugin:
    """DeepSeek-harness style: persistent memory across sessions.
    Wraps ExperienceStore (think.py) as a harness service."""

    def apply(self, ctx):
        try:
            from think import ExperienceStore
            store = ExperienceStore()
        except ImportError:
            store = None
        def remember(key, thought, outcome):
            if store:
                store.remember(key, thought, outcome)
            ctx.effect("memory_remember", {"key": key[:40]},
                       replay_fn=lambda p: True)
        def recall(key):
            if store:
                return store.recall(key)
            return None
        ctx.register("memory.remember", remember)
        ctx.register("memory.recall", recall)


class TerminalPlugin:
    """DeepSeek-harness style: local command execution. DANGEROUS ops
    are gated by TrustGate (if wired)."""

    def apply(self, ctx):
        import subprocess
        def run(cmd, timeout=30, gated=True):
            if gated:
                from gate import Route
                from schema import ToolCall
                # simulate TrustGate decision (if wired)
                decision = ctx._services.get("decision.trust")
                if decision:
                    route = decision.decide(ToolCall("terminal", {"cmd": cmd}),
                                            {"task": cmd})
                    if route.value == "ESCALATE":
                        return {"error": "blocked by TrustGate", "cmd": cmd}
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True,
                                   text=True, timeout=timeout)
                ctx.effect("terminal", {"cmd": cmd[:60]},
                           replay_fn=lambda p: True)
                return {"stdout": r.stdout[:2000], "stderr": r.stderr[:500],
                        "rc": r.returncode}
            except Exception as e:
                return {"error": str(e)[:200]}
        ctx.register("terminal.run", run)


class FetchPlugin:
    """DeepSeek-harness style: HTTP fetch (probe_api.py is the basis)."""

    def apply(self, ctx):
        import urllib.request
        def fetch(url, timeout=30):
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "helios-harness/1.0"})
                data = urllib.request.urlopen(req, timeout=timeout).read()
                ctx.effect("fetch", {"url": url[:60]},
                           replay_fn=lambda p: True)
                return {"status": 200, "body": data[:5000].decode(
                    errors="replace"), "bytes": len(data)}
            except Exception as e:
                return {"error": str(e)[:200], "url": url}
        ctx.register("fetch.get", fetch)


class FilesystemPlugin:
    """DeepSeek-harness style: local file ops beyond toy file_env.
    Read/write any file under a root dir (default cwd)."""

    def __init__(self, root="."):
        self.root = root

    def apply(self, ctx):
        import os
        root = os.path.abspath(self.root)
        def read(path):
            full = os.path.join(root, path)
            if not full.startswith(root):
                return {"error": "path escape blocked"}
            try:
                with open(full, encoding="utf-8", errors="replace") as f:
                    ctx.effect("fs_read", {"path": path[:60]},
                               replay_fn=lambda p: True)
                    return {"content": f.read()[:5000]}
            except Exception as e:
                return {"error": str(e)[:200]}
        def write(path, content):
            full = os.path.join(root, path)
            if not full.startswith(root):
                return {"error": "path escape blocked"}
            try:
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)
                ctx.effect("fs_write", {"path": path[:60]},
                           replay_fn=lambda p: True)
                return {"ok": True, "bytes": len(content)}
            except Exception as e:
                return {"error": str(e)[:200]}
        ctx.register("fs.read", read)
        ctx.register("fs.write", write)


class TimePlugin:
    """DeepSeek-harness style: time utilities."""

    def apply(self, ctx):
        import datetime
        def now():
            ctx.effect("time_now", {}, replay_fn=lambda p: True)
            return datetime.datetime.now().isoformat()
        def utc():
            return datetime.datetime.utcnow().isoformat() + "Z"
        ctx.register("time.now", now)
        ctx.register("time.utc", utc)


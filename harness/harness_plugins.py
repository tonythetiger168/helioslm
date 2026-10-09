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

class SQLitePlugin:
    """Query/exec on a SQLite db (stdlib sqlite3)."""

    def __init__(self, db_path=":memory:"):
        self.db_path = db_path

    def apply(self, ctx):
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        def query(sql, params=None):
            try:
                cur = conn.execute(sql, params or [])
                rows = [dict(r) for r in cur.fetchall()]
                ctx.effect("sqlite_query", {"sql": sql[:60]},
                           replay_fn=lambda p: True)
                return {"rows": rows}
            except Exception as e:
                return {"error": str(e)[:200]}
        def execute(sql, params=None):
            try:
                conn.execute(sql, params or [])
                conn.commit()
                return {"ok": True}
            except Exception as e:
                return {"error": str(e)[:200]}
        ctx.register("sqlite.query", query)
        ctx.register("sqlite.execute", execute)


class GitHubPlugin:
    """GitHub ops via REST API (like our hf_upload pattern)."""

    def __init__(self, token=None):
        self.token = token

    def apply(self, ctx):
        import json as _json, os, urllib.request
        token = self.token or os.environ.get("GITHUB_TOKEN")
        def api_call(method, path, body=None):
            if not token:
                return {"error": "no GITHUB_TOKEN"}
            req = urllib.request.Request(
                "https://api.github.com" + path,
                method=method,
                data=_json.dumps(body).encode() if body else None,
                headers={"Authorization": f"token {token}",
                         "Accept": "application/vnd.github+json",
                         "Content-Type": "application/json"})
            try:
                r = urllib.request.urlopen(req, timeout=30)
                ctx.effect("github", {"path": path[:60]},
                           replay_fn=lambda p: True)
                return _json.loads(r.read())
            except Exception as e:
                return {"error": str(e)[:200]}
        def get_repo(owner, repo):
            return api_call("GET", f"/repos/{owner}/{repo}")
        def create_issue(owner, repo, title, body=""):
            return api_call("POST", f"/repos/{owner}/{repo}/issues",
                            {"title": title, "body": body})
        ctx.register("github.get_repo", get_repo)
        ctx.register("github.create_issue", create_issue)


class SearchPlugin:
    """Web search via a simple DuckDuckGo scrape (no API key)."""

    def apply(self, ctx):
        import urllib.request, urllib.parse, re
        def search(query, n=5):
            try:
                url = "https://html.duckduckgo.com/html/?q=" +                     urllib.parse.quote(query)
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0"})
                html = urllib.request.urlopen(req, timeout=15).read().decode()
                titles = re.findall(r'<a[^>]*class="result__a"[^>]*>(.*?)</a>',
                                    html)[:n]
                ctx.effect("search", {"q": query[:40]},
                           replay_fn=lambda p: True)
                return {"results": [re.sub(r"<[^>]+>", "", t) for t in titles]}
            except Exception as e:
                return {"error": str(e)[:200]}
        ctx.register("search.web", search)


class SlackPlugin:
    """Slack message posting (needs bot token)."""

    def __init__(self, token=None, channel=None):
        self.token, self.channel = token, channel

    def apply(self, ctx):
        import json as _json, os, urllib.request
        token = self.token or os.environ.get("SLACK_TOKEN")
        def post(text, channel=None):
            ch = channel or self.channel
            if not token or not ch:
                return {"error": "no SLACK_TOKEN or channel"}
            req = urllib.request.Request(
                "https://slack.com/api/chat.postMessage",
                data=_json.dumps({"channel": ch, "text": text}).encode(),
                headers={"Authorization": f"Bearer {token}",
                         "Content-Type": "application/json"})
            try:
                r = urllib.request.urlopen(req, timeout=15)
                ctx.effect("slack", {"ch": ch[:20]},
                           replay_fn=lambda p: True)
                return _json.loads(r.read())
            except Exception as e:
                return {"error": str(e)[:200]}
        ctx.register("slack.post", post)


class ExcelPlugin:
    """Excel read/write via openpyxl (if installed) or CSV fallback."""

    def __init__(self, path=None):
        self.path = path

    def apply(self, ctx):
        def read(path=None):
            p = path or self.path
            try:
                import openpyxl
                wb = openpyxl.load_workbook(p)
                ws = wb.active
                rows = [[str(c.value) for c in row] for row in ws.iter_rows()]
                ctx.effect("excel_read", {"path": (p or "")[:40]},
                           replay_fn=lambda p2: True)
                return {"rows": rows[:100]}
            except ImportError:
                import csv
                with open(p, newline="", encoding="utf-8") as f:
                    return {"rows": list(csv.reader(f))[:100]}
            except Exception as e:
                return {"error": str(e)[:200]}
        def write(rows, path=None):
            p = path or self.path
            try:
                import openpyxl
                wb = openpyxl.Workbook()
                ws = wb.active
                for r in rows:
                    ws.append(r)
                wb.save(p)
            except ImportError:
                import csv
                with open(p, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(rows)
            ctx.effect("excel_write", {"path": (p or "")[:40]},
                       replay_fn=lambda p2: True)
            return {"ok": True, "rows": len(rows)}
        ctx.register("excel.read", read)
        ctx.register("excel.write", write)

class SearchPlugin:
    """Local search over a corpus (no external API). For HF-integration
    later, this wraps Tavily/Exa behind the same interface."""

    def apply(self, ctx):
        def search(query, corpus, top_k=5):
            scored = [(sum(1 for w in query.lower().split() if w in doc.lower()),
                       doc) for doc in corpus]
            scored.sort(key=lambda x: -x[0])
            results = [d for s, d in scored[:top_k] if s > 0]
            ctx.effect("search", {"q": query[:40], "hits": len(results)},
                       replay_fn=lambda p: True)
            return {"results": results, "n": len(results)}
        ctx.register("search.query", search)


class WebSearchPlugin:
    """Web search interface. Requires an API key (Tavily/Exa/Serp).
    Records the key-name (not value) for audit."""

    def __init__(self, provider="tavily", api_key_env=None):
        self.provider, self.api_key_env = provider, api_key_env

    def apply(self, ctx):
        import os
        def web_search(query, max_results=5):
            key = os.environ.get(self.api_key_env) if self.api_key_env else None
            if not key:
                return {"error": f"no API key for {self.provider} "
                        f"(set ${self.api_key_env})"}
            # provider-specific call would go here; recorded for audit
            ctx.effect("web_search", {"q": query[:40],
                                      "provider": self.provider},
                       replay_fn=lambda p: True)
            return {"stub": True, "provider": self.provider,
                    "note": "provider call not implemented -- "
                            "interface ready"}
        ctx.register("web.search", web_search)


class PDFPlugin:
    """PDF text extraction. Uses pypdf if installed, else stub."""

    def apply(self, ctx):
        def extract_text(path):
            try:
                from pypdf import PdfReader
                r = PdfReader(path)
                text = "\n".join(p.extract_text() or "" for p in r.pages[:20])
                ctx.effect("pdf_extract", {"path": path[:50]},
                           replay_fn=lambda p: True)
                return {"text": text[:5000], "pages": len(r.pages)}
            except ImportError:
                return {"error": "pypdf not installed (pip install pypdf)"}
            except Exception as e:
                return {"error": str(e)[:200]}
        ctx.register("pdf.extract", extract_text)


class SQLitePlugin:
    """SQLite read/write, constrained to a db path."""

    def __init__(self, db_path=":memory:"):
        self.db_path = db_path

    def apply(self, ctx):
        import sqlite3
        def query(sql, params=None):
            conn = sqlite3.connect(self.db_path)
            try:
                cur = conn.execute(sql, params or [])
                if sql.strip().upper().startswith("SELECT"):
                    rows = cur.fetchall()
                    cols = [d[0] for d in cur.description] if cur.description else []
                    ctx.effect("sqlite_read", {"sql": sql[:60]},
                               replay_fn=lambda p: True)
                    return {"columns": cols, "rows": rows[:100]}
                else:
                    conn.commit()
                    ctx.effect("sqlite_write", {"sql": sql[:60]},
                               replay_fn=lambda p: True)
                    return {"affected": cur.rowcount}
            except Exception as e:
                return {"error": str(e)[:200]}
            finally:
                conn.close()
        ctx.register("sqlite.query", query)


class TemplatePlugin:
    """Plugin scaffolding -- the 'new-plugin-template' analog."""

    TEMPLATE = '''class {name}Plugin:
    """{desc}"""

    def apply(self, ctx):
        def {name}_op(arg):
            ctx.effect("{name}", {{"arg": str(arg)[:50]}},
                       replay_fn=lambda p: True)
            return {{"ok": True}}
        ctx.register("{name}.op", {name}_op)
'''

    def apply(self, ctx):
        def scaffold(name, desc):
            code = self.TEMPLATE.format(name=name.lower(), desc=desc)
            ctx.effect("scaffold", {"name": name},
                       replay_fn=lambda p: True)
            return {"code": code}
        ctx.register("plugin.scaffold", scaffold)

class GitHubPlugin:
    """GitHub repo operations. Requires GITHUB_TOKEN env. Read-only by
    default; write ops (create issue) are gated by TrustGate."""

    def apply(self, ctx):
        import os
        def _api(path, method="GET", body=None):
            token = os.environ.get("GITHUB_TOKEN")
            if not token:
                return {"error": "GITHUB_TOKEN not set"}
            import urllib.request
            req = urllib.request.Request(
                "https://api.github.com" + path,
                method=method,
                data=json.dumps(body).encode() if body else None,
                headers={"Authorization": f"token {token}",
                         "Accept": "application/vnd.github.v3+json",
                         "User-Agent": "helios-harness"})
            try:
                r = json.loads(urllib.request.urlopen(req, timeout=30).read())
                ctx.effect("github", {"path": path[:50], "method": method},
                           replay_fn=lambda p: True)
                return r
            except Exception as e:
                return {"error": str(e)[:200]}
        def get_repo(owner, repo):
            return _api(f"/repos/{owner}/{repo}")
        def list_issues(owner, repo, state="open", n=10):
            r = _api(f"/repos/{owner}/{repo}/issues?state={state}&per_page={n}")
            if isinstance(r, list):
                return [{"number": i["number"], "title": i["title"]}
                        for i in r if "pull_request" not in i]
            return r
        def create_issue(owner, repo, title, body=""):
            decision = ctx._services.get("decision.trust")
            if decision:
                from schema import ToolCall
                from gate import Route
                route = decision.decide(
                    ToolCall("github", {"op": "create_issue"}),
                    {"task": title})
                if route.value == "ESCALATE":
                    return {"error": "blocked by TrustGate"}
            return _api(f"/repos/{owner}/{repo}/issues", "POST",
                        {"title": title, "body": body})
        ctx.register("github.get_repo", get_repo)
        ctx.register("github.list_issues", list_issues)
        ctx.register("github.create_issue", create_issue)


class PostgresPlugin:
    """PostgreSQL read/write. Requires psycopg2 or pg80000."""

    def __init__(self, dsn=None, dsn_env="DATABASE_URL"):
        self.dsn, self.dsn_env = dsn, dsn_env

    def apply(self, ctx):
        import os
        def query(sql, params=None):
            dsn = self.dsn or os.environ.get(self.dsn_env)
            if not dsn:
                return {"error": f"no DSN (set ${self.dsn_env})"}
            try:
                import psycopg2
                conn = psycopg2.connect(dsn)
            except ImportError:
                try:
                    import pg8000.dbapi as pg
                    conn = pg.connect(dsn)
                except ImportError:
                    return {"error": "psycopg2 or pg8000 required"}
            try:
                cur = conn.cursor()
                cur.execute(sql, params or [])
                if sql.strip().upper().startswith("SELECT"):
                    rows = cur.fetchall()
                    cols = [d.name if hasattr(d, "name") else d[0]
                            for d in cur.description]
                    ctx.effect("pg_read", {"sql": sql[:60]},
                               replay_fn=lambda p: True)
                    return {"columns": cols, "rows": [list(r) for r in rows[:100]]}
                else:
                    conn.commit()
                    ctx.effect("pg_write", {"sql": sql[:60]},
                               replay_fn=lambda p: True)
                    return {"affected": cur.rowcount}
            except Exception as e:
                return {"error": str(e)[:200]}
            finally:
                conn.close()
        ctx.register("postgres.query", query)

class ExcelPlugin:
    """Excel read/write via openpyxl."""

    def apply(self, ctx):
        def write(path, rows):
            try:
                from openpyxl import Workbook
                wb = Workbook()
                ws = wb.active
                for row in rows:
                    ws.append(row)
                wb.save(path)
                ctx.effect("excel_write", {"path": path[:50]},
                           replay_fn=lambda p: True)
                return {"ok": True, "rows": len(rows)}
            except ImportError:
                return {"error": "openpyxl not installed"}
            except Exception as e:
                return {"error": str(e)[:200]}
        def read(path):
            try:
                from openpyxl import load_workbook
                wb = load_workbook(path)
                ws = wb.active
                rows = [[c.value for c in row] for row in ws.iter_rows()]
                ctx.effect("excel_read", {"path": path[:50]},
                           replay_fn=lambda p: True)
                return {"rows": rows[:100]}
            except ImportError:
                return {"error": "openpyxl not installed"}
            except Exception as e:
                return {"error": str(e)[:200]}
        ctx.register("excel.write", write)
        ctx.register("excel.read", read)


class DocxPlugin:
    """Word doc read/write via python-docx."""

    def apply(self, ctx):
        def write(path, paragraphs):
            try:
                from docx import Document
                doc = Document()
                for p in paragraphs:
                    doc.add_paragraph(str(p))
                doc.save(path)
                ctx.effect("docx_write", {"path": path[:50]},
                           replay_fn=lambda p: True)
                return {"ok": True, "paragraphs": len(paragraphs)}
            except ImportError:
                return {"error": "python-docx not installed"}
            except Exception as e:
                return {"error": str(e)[:200]}
        def read(path):
            try:
                from docx import Document
                doc = Document(path)
                paras = [p.text for p in doc.paragraphs if p.text.strip()]
                ctx.effect("docx_read", {"path": path[:50]},
                           replay_fn=lambda p: True)
                return {"paragraphs": paras[:50]}
            except ImportError:
                return {"error": "python-docx not installed"}
            except Exception as e:
                return {"error": str(e)[:200]}
        ctx.register("docx.write", write)
        ctx.register("docx.read", read)


class PptxPlugin:
    """PowerPoint via python-pptx."""

    def apply(self, ctx):
        def write(path, slides):
            try:
                from pptx import Presentation
                prs = Presentation()
                for slide_content in slides:
                    slide = prs.slides.add_slide(
                        prs.slide_layouts[1])
                    slide.shapes.title.text = str(slide_content.get("title", ""))
                    body = slide.placeholders[1].text_frame
                    for point in slide_content.get("points", []):
                        body.add_paragraph().text = str(point)
                prs.save(path)
                ctx.effect("pptx_write", {"path": path[:50]},
                           replay_fn=lambda p: True)
                return {"ok": True, "slides": len(slides)}
            except ImportError:
                return {"error": "python-pptx not installed"}
            except Exception as e:
                return {"error": str(e)[:200]}
        ctx.register("pptx.write", write)


class YouTubePlugin:
    """YouTube transcript via youtube-transcript-api (no external key)."""

    def apply(self, ctx):
        def transcript(video_id):
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                t = YouTubeTranscriptApi.get_transcript(video_id)
                text = " ".join(seg["text"] for seg in t[:100])
                ctx.effect("youtube", {"id": video_id[:20]},
                           replay_fn=lambda p: True)
                return {"text": text[:5000], "segments": len(t)}
            except ImportError:
                return {"error": "youtube-transcript-api not installed"}
            except Exception as e:
                return {"error": str(e)[:200]}
        ctx.register("youtube.transcript", transcript)


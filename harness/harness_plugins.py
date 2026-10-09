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

class MCPWizardPlugin:
    """v1.12: MCP (Model Context Protocol) integration wizard. Wraps any
    MCP server as a harness plugin. Requires mcp package."""

    def apply(self, ctx):
        def wrap(server_name, command, args=None):
            try:
                import mcp
            except ImportError:
                return {"error": "mcp package not installed"}
            # registration would go here; recorded for audit
            ctx.effect("mcp_wrap", {"server": server_name[:40]},
                       replay_fn=lambda p: True)
            return {"wrapped": server_name, "command": command,
                    "note": "MCP server wrapping requires the mcp package"}
        ctx.register("mcp.wrap", wrap)


class TmuxPlugin:
    """v1.12: tmux session manager."""

    def apply(self, ctx):
        import subprocess
        def new_session(name):
            try:
                subprocess.run(["tmux", "new-session", "-d", "-s", name],
                               capture_output=True, timeout=10)
                ctx.effect("tmux_new", {"name": name[:30]},
                           replay_fn=lambda p: True)
                return {"ok": True, "session": name}
            except FileNotFoundError:
                return {"error": "tmux not installed"}
            except Exception as e:
                return {"error": str(e)[:200]}
        def list_sessions():
            try:
                r = subprocess.run(["tmux", "ls"], capture_output=True,
                                   text=True, timeout=10)
                return {"sessions": r.stdout.strip().split("\n") if r.returncode == 0 else []}
            except FileNotFoundError:
                return {"error": "tmux not installed"}
        def send_keys(session, keys):
            try:
                subprocess.run(["tmux", "send-keys", "-t", session, keys, "Enter"],
                               capture_output=True, timeout=10)
                ctx.effect("tmux_send", {"session": session[:30]},
                           replay_fn=lambda p: True)
                return {"ok": True}
            except FileNotFoundError:
                return {"error": "tmux not installed"}
        ctx.register("tmux.new", new_session)
        ctx.register("tmux.list", list_sessions)
        ctx.register("tmux.send", send_keys)


class EverythingPlugin:
    """v1.12: local file content search (like 'everything' but pure
    Python, no external indexer)."""

    def apply(self, ctx):
        import os
        def search(root, pattern, max_results=20):
            hits = []
            for dirpath, _, files in os.walk(root):
                for f in files:
                    full = os.path.join(dirpath, f)
                    try:
                        if pattern.lower() in f.lower():
                            hits.append(full)
                            if len(hits) >= max_results:
                                break
                        elif os.path.getsize(full) < 100_000:
                            with open(full, errors="replace") as fh:
                                if pattern.lower() in fh.read().lower():
                                    hits.append(full)
                                    if len(hits) >= max_results:
                                        break
                    except Exception:
                        pass
                if len(hits) >= max_results:
                    break
            ctx.effect("everything", {"pattern": pattern[:30],
                                      "hits": len(hits)},
                       replay_fn=lambda p: True)
            return {"hits": hits, "n": len(hits)}
        ctx.register("everything.search", search)

class VideoPlugin:
    """Video generation interface. Wraps external APIs (Runway/Pika/
    Sora-class) behind a common harness interface. No local fallback
    (video gen is compute-prohibitive locally)."""

    def __init__(self, provider="stub", api_key_env=None):
        self.provider, self.api_key_env = provider, api_key_env

    def apply(self, ctx):
        import os
        def generate(prompt, duration_s=5, resolution="720p"):
            key = os.environ.get(self.api_key_env) if self.api_key_env else None
            if not key:
                return {"error": f"no API key for {self.provider} "
                        f"(set ${self.api_key_env})"}
            ctx.effect("video_gen", {"prompt": prompt[:50],
                                     "provider": self.provider},
                       replay_fn=lambda p: True)
            return {"stub": True, "provider": self.provider,
                    "note": "provider call not implemented"}
        ctx.register("video.generate", generate)


class AudioPlugin:
    """Audio/TTS generation. Local fallback: pyttsx3 (offline TTS).
    External: ElevenLabs/OpenAI TTS (api_key_env)."""

    def __init__(self, provider="auto", api_key_env=None):
        self.provider, self.api_key_env = provider, api_key_env

    def apply(self, ctx):
        import os
        def tts(text, out_path=None):
            # external first
            key = os.environ.get(self.api_key_env) if self.api_key_env else None
            if key and self.provider != "local":
                ctx.effect("tts_external", {"chars": len(text)},
                           replay_fn=lambda p: True)
                return {"stub": True, "provider": self.provider,
                        "note": "external TTS not implemented"}
            # local fallback
            try:
                import pyttsx3
                engine = pyttsx3.init()
                if out_path:
                    engine.save_to_file(text, out_path)
                    engine.runAndWait()
                    ctx.effect("tts_local", {"chars": len(text)},
                               replay_fn=lambda p: True)
                    return {"ok": True, "path": out_path}
                engine.say(text)
                engine.runAndWait()
                return {"ok": True, "spoken": True}
            except ImportError:
                return {"error": "pyttsx3 not installed and no external key"}
            except Exception as e:
                return {"error": str(e)[:200]}
        def stt(audio_path):
            try:
                import speech_recognition as sr
                r = sr.Recognizer()
                with sr.AudioFile(audio_path) as source:
                    audio = r.record(source)
                text = r.recognize_google(audio)
                ctx.effect("stt", {"path": audio_path[:40]},
                           replay_fn=lambda p: True)
                return {"text": text}
            except ImportError:
                return {"error": "speech_recognition not installed"}
            except Exception as e:
                return {"error": str(e)[:200]}
        ctx.register("audio.tts", tts)
        ctx.register("audio.stt", stt)


class NanoVideoPlugin:
    """Lightweight video from images + audio (ffmpeg slideshow).
    The 'nano-pdf' analog for video -- minimal local generation."""

    def apply(self, ctx):
        import subprocess, os
        def from_images(image_paths, audio_path=None, out_path="out.mp4",
                        duration_per_image=2):
            try:
                if not image_paths:
                    return {"error": "no images"}
                # ffmpeg concat demuxer
                list_file = "/tmp/ffmpeg_list.txt"
                with open(list_file, "w") as f:
                    for img in image_paths:
                        f.write(f"file '{img}'\n")
                        f.write(f"duration {duration_per_image}\n")
                cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                       "-i", list_file, "-vsync", "vfr", "-pix_fmt", "yuv420p"]
                if audio_path:
                    cmd += ["-i", audio_path, "-c:a", "aac", "-shortest"]
                cmd += [out_path]
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=120)
                if r.returncode != 0:
                    return {"error": r.stderr[-200:]}
                ctx.effect("nano_video", {"images": len(image_paths)},
                           replay_fn=lambda p: True)
                return {"ok": True, "path": out_path,
                        "size": os.path.getsize(out_path)}
            except FileNotFoundError:
                return {"error": "ffmpeg not installed"}
            except Exception as e:
                return {"error": str(e)[:200]}
        ctx.register("nanovideo.from_images", from_images)

class BlenderPlugin:
    """Blender 3D modeling via headless blender python. Requires blender
    binary (not pip-installable). Generates .py scripts for blender to
    execute, or runs blender --background --python."""

    def apply(self, ctx):
        import subprocess, os
        def run_script(script_path, blend_path=None):
            if not os.path.exists(script_path):
                return {"error": "script not found"}
            try:
                cmd = ["blender", "--background", "--python", script_path]
                if blend_path:
                    cmd += [blend_path]
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=300)
                ctx.effect("blender", {"script": script_path[:40]},
                           replay_fn=lambda p: True)
                return {"rc": r.returncode,
                        "stdout": r.stdout[-500:],
                        "stderr": r.stderr[-300:]}
            except FileNotFoundError:
                return {"error": "blender binary not installed"}
            except Exception as e:
                return {"error": str(e)[:200]}
        def gen_script(desc, out_path):
            # minimal cube scene as placeholder
            code = f"""import bpy
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
cube = bpy.context.active_object
cube.name = "gen_{desc[:20]}"
bpy.ops.wm.save_as_mainfile(filepath="{out_path}")
"""
            with open(out_path + ".py", "w") as f:
                f.write(code)
            ctx.effect("blender_gen", {"desc": desc[:40]},
                       replay_fn=lambda p: True)
            return {"script": out_path + ".py"}
        ctx.register("blender.run", run_script)
        ctx.register("blender.gen", gen_script)


class OmiPlugin:
    """Omi wearable interface. Bluetooth Low Energy requires bleak.
    The omi device streams audio/transcriptions over BLE."""

    def apply(self, ctx):
        def connect(device_name="Omi"):
            try:
                import bleak
            except ImportError:
                return {"error": "bleak not installed (pip install bleak)"}
            ctx.effect("omi_connect", {"device": device_name[:30]},
                       replay_fn=lambda p: True)
            return {"stub": True, "device": device_name,
                    "note": "BLE connection requires bleak + hardware"}
        def transcribe():
            return {"error": "not connected -- use omi.connect first"}
        ctx.register("omi.connect", connect)
        ctx.register("omi.transcribe", transcribe)

class BenchmarkPlugin:
    """v1.15: benchmark suite -- the paper's numbers as a runnable
    plugin. Loads RLCDAlignBench artifacts and reports medians. Requires
    benchmarks/*.json in the repo."""

    def apply(self, ctx):
        import json, os
        def _load(name):
            for cand in (f"benchmarks/{name}", f"../benchmarks/{name}"):
                if os.path.exists(cand):
                    return json.load(open(cand))
            return None
        def alignbench_summary():
            out = {}
            for name, key in [("alignbench_char_results.json", "char"),
                              ("alignbench_tfidf_results.json", "tfidf"),
                              ("alignbench_jev_zero_shot_auroc.json", "jev")]:
                d = _load(name)
                if d:
                    vals = [v["auroc"] for v in d.values()
                            if isinstance(v, dict) and "auroc" in v]                            if isinstance(d, dict) and any(
                               isinstance(v, dict) for v in d.values())                            else list(d.values())
                    vals = [v for v in vals if isinstance(v, (int, float))]
                    if vals:
                        vals.sort()
                        out[key] = round(vals[len(vals)//2], 3)
            ctx.effect("benchmark", {"n": len(out)}, replay_fn=lambda p: True)
            return out
        def api_probe_summary():
            d = _load("api_probe_summary_2026-10-04.json")
            if d:
                ctx.effect("api_probe", {"models": len(d.get("models", {}))},
                           replay_fn=lambda p: True)
            return d or {"error": "api_probe_summary not found"}
        ctx.register("benchmark.alignbench", alignbench_summary)
        ctx.register("benchmark.api_probe", api_probe_summary)


class ReportPlugin:
    """v1.15: generate a markdown report from benchmark results."""

    def apply(self, ctx):
        def gen_report():
            ab = ctx.benchmark.alignbench()
            lines = ["# HeliosLM Benchmark Report", "",
                     "| Leg | Median AUROC |", "|---|---|"]
            for k, v in ab.items():
                lines.append(f"| {k} | {v} |")
            md = "\n".join(lines)
            ctx.effect("report", {"legs": len(ab)}, replay_fn=lambda p: True)
            return md
        ctx.register("report.gen", gen_report)

class RobotControlPlugin:
    """Robot control: mobile base (cmd_vel), arm (joint angles), gripper.
    Two modes:
    - mock (default): a kinematic simulator, no hardware
    - ros2: interface to a real ROS2 node (requires rclpy)
    All movements are TrustGate-gated and effect-audited."""

    def __init__(self, mode="mock"):
        self.mode = mode

    def apply(self, ctx):
        state = {"x": 0.0, "y": 0.0, "theta": 0.0,
                 "joints": [0.0] * 6, "gripper": 0.0}
        def _gate(op, params):
            decision = ctx._services.get("decision.trust")
            if decision:
                from schema import ToolCall
                route = decision.decide(
                    ToolCall("robot", {"op": op, **params}),
                    {"task": f"robot.{op}"})
                if route.value == "ESCALATE":
                    return {"error": f"robot.{op} blocked by TrustGate"}
            return None
        def move_base(vx, vy, omega, duration_s=1.0):
            g = _gate("move_base", {"vx": vx, "vy": vy})
            if g:
                return g
            if self.mode == "mock":
                state["x"] += vx * duration_s
                state["y"] += vy * duration_s
                state["theta"] += omega * duration_s
            ctx.effect("robot_move", {"vx": vx, "vy": vy},
                       replay_fn=lambda p: True)
            return {"ok": True, "state": dict(state)}
        def move_arm(joint_angles):
            g = _gate("move_arm", {"n": len(joint_angles)})
            if g:
                return g
            if self.mode == "mock":
                state["joints"] = list(joint_angles)[:6]
            ctx.effect("robot_arm", {"n": len(joint_angles)},
                       replay_fn=lambda p: True)
            return {"ok": True, "joints": list(joint_angles)[:6]}
        def gripper(position):
            g = _gate("gripper", {"pos": position})
            if g:
                return g
            if self.mode == "mock":
                state["gripper"] = max(0.0, min(1.0, position))
            ctx.effect("robot_gripper", {"pos": position},
                       replay_fn=lambda p: True)
            return {"ok": True, "gripper": state["gripper"]}
        def get_state():
            return dict(state)
        ctx.register("robot.move_base", move_base)
        ctx.register("robot.move_arm", move_arm)
        ctx.register("robot.gripper", gripper)
        ctx.register("robot.state", get_state)

class AgentSwarmPlugin:
    """v1.17: multi-agent swarm. Leader decomposes a task, delegates to
    workers, aggregates results. Blackboard for shared state. Each
    worker runs through the same harness (effects audited per agent)."""

    def __init__(self, n_workers=3):
        self.n_workers = n_workers

    def apply(self, ctx):
        def spawn_swarm(task_text, model_fns):
            # model_fns: dict {worker_name: model_fn}
            ctx.state["swarm"] = {
                "task": task_text,
                "blackboard": {},
                "workers": {},
                "results": []}
            n = min(self.n_workers, len(model_fns))
            workers = dict(list(model_fns.items())[:n])
            for name, fn in workers.items():
                ctx.state["swarm"]["workers"][name] = {"status": "idle"}
            ctx.effect("swarm_spawn", {"n": n}, replay_fn=lambda p: True)
            return {"workers": list(workers.keys()), "n": n}

        def delegate(worker_name, sub_task, model_fn):
            ctx.state["swarm"]["workers"][worker_name] = {"status": "busy",
                                                           "task": sub_task}
            ctx.effect("swarm_delegate",
                       {"worker": worker_name, "task": sub_task[:40]},
                       replay_fn=lambda p: True)
            # worker runs the task through the harness
            wf = ctx.agent.run(sub_task, model_fn, max_steps=6)
            result = wf["final"]
            ctx.state["swarm"]["workers"][worker_name] = {"status": "done",
                                                           "result": result}
            ctx.state["swarm"]["results"].append(
                {"worker": worker_name, "result": result})
            ctx.state["swarm"]["blackboard"][worker_name] = result
            ctx.effect("swarm_done",
                       {"worker": worker_name, "ok": result is not None},
                       replay_fn=lambda p: True)
            return result

        def aggregate():
            results = ctx.state["swarm"]["results"]
            # simple majority/concat aggregation
            finals = [r["result"] for r in results if r["result"]]
            if not finals:
                return None
            # if all same, return it; else return list
            if len(set(finals)) == 1:
                return finals[0]
            return finals
        def blackboard():
            return ctx.state["swarm"]["blackboard"]

        ctx.register("swarm.spawn", spawn_swarm)
        ctx.register("swarm.delegate", delegate)
        ctx.register("swarm.aggregate", aggregate)
        ctx.register("swarm.blackboard", blackboard)

class MultiRobotPlugin:
    """v1.18: multi-robot coordination. Fleet management, task allocation
    (nearest-capable), formation control. Built on AgentSwarmPlugin +
    RobotControlPlugin patterns."""

    def apply(self, ctx):
        fleet = {}
        def register(robot_id, caps=None):
            fleet[robot_id] = {"caps": caps or [], "pos": (0.0, 0.0),
                               "busy": False}
            ctx.effect("fleet_reg", {"id": robot_id}, replay_fn=lambda p: True)
            return {"fleet": list(fleet.keys())}
        def allocate(task, required_caps=None):
            # nearest capable idle robot
            cands = [r for r, s in fleet.items()
                     if not s["busy"]
                     and (not required_caps
                          or all(c in s["caps"] for c in required_caps))]
            if not cands:
                return {"error": "no capable robot"}
            chosen = cands[0]
            fleet[chosen]["busy"] = True
            ctx.effect("fleet_alloc", {"task": task[:30], "robot": chosen},
                       replay_fn=lambda p: True)
            return {"robot": chosen}
        def release(robot_id):
            fleet[robot_id]["busy"] = False
        def formation(points):
            # assign robots to formation points (greedy nearest)
            assignments = {}
            free = [r for r, s in fleet.items() if not s["busy"]]
            for pt in points[:len(free)]:
                r = free.pop(0)
                assignments[r] = pt
                fleet[r]["pos"] = pt
            ctx.effect("fleet_formation", {"n": len(assignments)},
                       replay_fn=lambda p: True)
            return assignments
        def fleet_state():
            return dict(fleet)
        ctx.register("fleet.register", register)
        ctx.register("fleet.allocate", allocate)
        ctx.register("fleet.release", release)
        ctx.register("fleet.formation", formation)
        ctx.register("fleet.state", fleet_state)


class SLAMPlugin:
    """v1.18: occupancy-grid SLAM (toy-scale, no external deps). Simulated
    lidar scans against a known map; integrates poses into a grid."""

    def apply(self, ctx):
        def create_map(w=20, h=20):
            return {"w": w, "h": h, "grid": [[0.5] * w for _ in range(h)],
                    "pose": (0.0, 0.0, 0.0)}
        def scan(sim_map, pose, n_beams=8, max_range=5.0):
            # simulated: returns ranges (mock obstacles at walls)
            px, py, th = pose
            ranges = []
            for i in range(n_beams):
                a = th + 2 * 3.14159 * i / n_beams
                r = max_range
                # wall at x=10 or y=10
                if abs(a) < 0.1:
                    r = min(r, 10 - px)
                if abs(abs(a) - 3.14159) < 0.1:
                    r = min(r, px)
                if abs(abs(a) - 1.5708) < 0.1:
                    r = min(r, 10 - py)
                if abs(abs(a) + 1.5708) < 0.1 or abs(abs(a) - 4.712) < 0.1:
                    r = min(r, py)
                ranges.append(round(max(0.0, r), 2))
            ctx.effect("slam_scan", {"n": n_beams}, replay_fn=lambda p: True)
            return ranges
        def integrate(sim_map, pose, ranges):
            # mark cells along beams as free, endpoints as occupied
            px, py, th = pose
            grid = sim_map["grid"]
            for i, r in enumerate(ranges):
                a = th + 2 * 3.14159 * i / len(ranges)
                steps = int(r * 2)
                for s in range(steps):
                    x = int(px + s / 2 * __import__("math").cos(a))
                    y = int(py + s / 2 * __import__("math").sin(a))
                    if 0 <= x < sim_map["w"] and 0 <= y < sim_map["h"]:
                        grid[y][x] = max(0.0, grid[y][x] - 0.1)
                ex, ey = int(px + r * __import__("math").cos(a)), \
                         int(py + r * __import__("math").sin(a))
                if 0 <= ex < sim_map["w"] and 0 <= ey < sim_map["h"]:
                    grid[ey][ex] = min(1.0, grid[ey][ex] + 0.3)
            sim_map["pose"] = pose
            ctx.effect("slam_integrate", {"beams": len(ranges)},
                       replay_fn=lambda p: True)
            return sim_map
        def frontier(sim_map):
            # cells with 0.5 (unknown) adjacent to free (<0.3)
            grid = sim_map["grid"]
            fr = []
            for y in range(1, sim_map["h"] - 1):
                for x in range(1, sim_map["w"] - 1):
                    if grid[y][x] == 0.5:
                        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
                            if grid[y+dy][x+dx] < 0.3:
                                fr.append((x, y))
                                break
            return fr
        ctx.register("slam.create_map", create_map)
        ctx.register("slam.scan", scan)
        ctx.register("slam.integrate", integrate)
        ctx.register("slam.frontier", frontier)


class VisionPlugin:
    """v1.18: computer vision. Object detection interface (YOLO-class
    stub), OCR (pytesseract or stub), image captioning (transformers
    or stub). All effects audited."""

    def apply(self, ctx):
        def detect(image_path, model="yolov8n.pt"):
            try:
                from ultralytics import YOLO
                m = YOLO(model)
                r = m(image_path)[0]
                boxes = [{"cls": int(b.cls), "conf": float(b.conf)}
                         for b in r.boxes]
                ctx.effect("vision_detect", {"n": len(boxes)},
                           replay_fn=lambda p: True)
                return {"boxes": boxes}
            except ImportError:
                return {"error": "ultralytics not installed"}
            except Exception as e:
                return {"error": str(e)[:200]}
        def ocr(image_path):
            try:
                import pytesseract
                from PIL import Image
                text = pytesseract.image_to_string(Image.open(image_path))
                ctx.effect("vision_ocr", {"chars": len(text)},
                           replay_fn=lambda p: True)
                return {"text": text}
            except ImportError:
                return {"error": "pytesseract not installed"}
            except Exception as e:
                return {"error": str(e)[:200]}
        def caption(image_path):
            try:
                from transformers import pipeline
                cap = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
                r = cap(image_path)
                ctx.effect("vision_caption", {}, replay_fn=lambda p: True)
                return {"caption": r[0]["generated_text"]}
            except ImportError:
                return {"error": "transformers not installed"}
            except Exception as e:
                return {"error": str(e)[:200]}
        ctx.register("vision.detect", detect)
        ctx.register("vision.ocr", ocr)
        ctx.register("vision.caption", caption)

class VoiceDialogPlugin:
    """v1.19: voice dialog loop -- STT -> LLM -> TTS, with barge-in
    (user can interrupt). Uses AudioPlugin services."""

    def apply(self, ctx):
        def dialog_turn(user_speech_text, model_fn, audio=None):
            # STT if audio given, else assume text
            text = user_speech_text
            if audio:
                stt = ctx._services.get("audio.stt")
                if stt:
                    r = ctx.audio.stt(audio)
                    text = r.get("text", text)
            # LLM response
            response = model_fn(text, 0, 0)
            # TTS
            tts = ctx._services.get("audio.tts")
            spoken = None
            if tts:
                tts(ctx.audio.tts(response) if False else response)
                spoken = True
            ctx.effect("voice_turn", {"chars": len(text)},
                       replay_fn=lambda p: True)
            return {"user": text, "response": response, "spoken": spoken}
        ctx.register("voice.turn", dialog_turn)


class TerminalUIPlugin:
    """v1.19: terminal UI -- progress bars, menus, status lines. For
    long-running harness workflows."""

    def apply(self, ctx):
        def progress(current, total, label=""):
            pct = 100 * current / max(1, total)
            bar = "#" * int(pct // 5) + "-" * (20 - int(pct // 5))
            line = f"\r[{bar}] {pct:.0f}% {label}"
            print(line, end="", flush=True)
            if current >= total:
                print()
            ctx.effect("ui_progress", {"pct": pct}, replay_fn=lambda p: True)
        def menu(options, title="Select:"):
            print(title)
            for i, o in enumerate(options):
                print(f"  {i+1}. {o}")
            ctx.effect("ui_menu", {"n": len(options)}, replay_fn=lambda p: True)
            return options
        ctx.register("ui.progress", progress)
        ctx.register("ui.menu", menu)


class AutoDrivePlugin:
    """v1.19: toy autonomous driving -- lane keeping + obstacle avoidance
    on a simulated road. No external simulator (Carla/Donkey) required."""

    def apply(self, ctx):
        def simulate(road, n_steps=100, speed=1.0):
            state = {"lane_offset": 0.0, "obstacles": 0, "collisions": 0,
                     "path": []}
            for step in range(n_steps):
                # lane keeping: P controller
                state["lane_offset"] *= 0.9
                # obstacle every 20 steps
                if step % 20 == 0 and step > 0:
                    state["obstacles"] += 1
                    # simple avoidance: swerve
                    state["lane_offset"] += 0.5
                state["path"].append(state["lane_offset"])
                if abs(state["lane_offset"]) > 1.0:
                    state["collisions"] += 1
            ctx.effect("autodrive", {"steps": n_steps,
                                     "collisions": state["collisions"]},
                       replay_fn=lambda p: True)
            return state
        ctx.register("autodrive.simulate", simulate)


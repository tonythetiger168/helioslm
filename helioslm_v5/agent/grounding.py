"""HeliosLM v5.34 — deterministic grounding (System One: policy in code).

The mid autopsy (docs/scale_lineage_2026-09-29.md) named the disease:
the model decides tool SEQUENCES fluently but confabulates CONTENT at
both ends of the pipeline (task->args, and sometimes obs->finish), all
at 0.9999 confidence. This module is the therapy, deployed at inference
time with the existing gate:

- GroundingGate wraps any gate. On Route.DIRECT it rewrites the call's
  ARGS in place (ToolCall.args is a dict; the frozen dataclass blocks
  reassignment, not content mutation) with values PARSED FROM THE TASK
  TEXT, and rewrites finish answers with the DETERMINISTICALLY PREDICTED
  tool-chain output. The model keeps exactly one job: the tool sequence.
- Prediction is sound because our tools are deterministic: tools.calc /
  tools.str_op replayed over the grounded args produce exactly the obs
  the executor will return. The gate replays the same forward function.

Boundaries (recorded, not hidden):
- ESCALATE / oracle paths are NEVER touched.
- Grounding covers the v5.23-v5.31 task grammar (calc/str/compose + the
  three file-env families). Unknown task shapes pass through UNGROUNDED;
  coverage is queryable via groundable().
- Replay deviation (precise): the trajectory records the GROUNDED call
  (args are mutated in place on the parsed-call object), so both the
  recorded call and its observation come from grounded execution while
  verify_chat_replay re-derives the RAW call from generated text --
  replay mismatches on the call comparison by design. Grounded runs
  verify by determinism (T33); a grounding-aware replay is future work.
- write_transform file_read redirection (documented): any file_read in
  a write_transform task is redirected to the result file b, on the
  assumption that reading the raw intermediate is never the intended
  semantics of this grammar. Canonical sequences only read b.
- _state grows per task text and is not thread-safe: single-session
  loops only (documented, matches AgentLoop/ChatSession semantics).
"""
import re
from dataclasses import dataclass, field

try:
    from .gate import Gate, Route
    from .task_grammar import parse as parse_task
    from .tools import calc, str_op
except ImportError:
    from gate import Gate, Route
    from task_grammar import parse as parse_task
    from tools import calc, str_op


@dataclass
class GroundingGate(Gate):
    inner: Gate
    # per-task state: task_text -> {"obs": [predicted obs, ...],
    #                               "files": {path: content},
    #                               "reads": [...], "writes": 0, "last"}
    _state: dict = field(default_factory=dict)

    def _parse(self, task):
        # v5.35: the grammar now lives in ONE place (task_grammar.py);
        # the v5.34 regex table -- and its whole debug chain -- was
        # deleted in favor of the shared table that envs also render
        # through. Drift between producer and consumer is now
        # structurally impossible (T34 guards the roundtrip).
        return parse_task(task)

    def groundable(self, task_text):
        return self._parse(task_text) is not None

    # -- the wrapper -----------------------------------------------------

    def decide(self, call, context):
        route = self.inner.decide(call, context)
        if route != Route.DIRECT:
            return route
        task = context.get("task", "")
        spec = self._parse(task)
        if spec is None:
            return route
        st = self._state.setdefault(task, {"obs": [], "files": {},
                                           "reads": [], "writes": 0,
                                           "last": None})
        k = spec["kind"]
        try:
            if call.name == "calc":
                if k == "accumulate":
                    if not st["obs"]:
                        expr = spec["e1"]
                    elif len(st["obs"]) == 1:
                        expr = spec["e2"]
                    else:  # the closing sum over both stored results
                        expr = f"({st['obs'][0]}) + ({st['obs'][1]})"
                else:
                    expr = spec.get("expr")
                if expr is None:
                    return route
                call.args["expr"] = expr
                out = calc(expr)
                st["obs"].append(out)
                st["last"] = ("obs", out)
            elif call.name == "str_op":
                if k == "str":
                    call.args.update({"s": spec["s"], "op": spec["op"],
                                      "n": spec["n"]})
                    out = str_op(spec["s"], spec["op"], spec["n"])
                    st["obs"].append(out)
                    st["last"] = ("obs", out)
                elif k == "compose" and st["obs"]:
                    call.args.update({"s": st["obs"][-1], "op": spec["op"],
                                      "n": 0})
                    out = str_op(st["obs"][-1], spec["op"], 0)
                    st["obs"].append(out)
                    st["last"] = ("obs", out)
                elif k == "write_transform":
                    # transform the ORIGINAL string from the task text
                    call.args.update({"s": spec["s"], "op": spec["op"],
                                      "n": spec.get("n", 0)})
                    out = str_op(spec["s"], spec["op"], spec.get("n", 0))
                    st["obs"].append(out)
                    st["last"] = ("obs", out)
            elif call.name == "file_write":
                if k == "write_transform":
                    path = spec["a"] if st["writes"] == 0 else spec["b"]
                    content = spec["s"] if st["writes"] == 0 \
                        else st["obs"][-1]
                elif k == "write_read":
                    path, content = spec["path"], st["obs"][-1]
                elif k == "accumulate":
                    path = spec["p1"] if st["writes"] == 0 else spec["p2"]
                    content = st["obs"][st["writes"]]
                else:
                    return route
                call.args.update({"path": path, "content": content})
                st["files"][path] = content
                st["writes"] += 1
            elif call.name == "file_read":
                if k == "write_transform":
                    path = spec["b"]
                elif k == "write_read":
                    path = spec["path"]
                elif k == "accumulate":
                    path = spec["p1"] if not st["reads"] else spec["p2"]
                else:
                    return route
                call.args["path"] = path
                content = st["files"].get(path, "")
                st["reads"].append(content)
                st["last"] = ("read", content)
            elif call.name == "finish":
                # Semantic finish grounding (code-review fix, 2026-09-29):
                # the answer anchors to the TASK's definition, not to the
                # sequence's last event -- a mid-sequence deviation must
                # not produce a grounded-but-wrong answer (the exact
                # disease this module exists to cure). For accumulate, two
                # observations are enough to compute the sum even if the
                # closing calc never happened.
                if k == "accumulate" and len(st["obs"]) >= 2:
                    call.args["answer"] = calc(
                        f"({st['obs'][0]}) + ({st['obs'][1]})")
                elif st["last"] is not None:
                    call.args["answer"] = st["last"][1]
        except Exception:
            pass  # best-effort: never crash the loop
        return route

    def escalate(self, call, context):
        return self.inner.escalate(call, context)

    def ask(self, call, context, kind, question):
        return self.inner.ask(call, context, kind, question)


def verify_grounded_replay(traj, registry, impls) -> None:
    """Grounding-aware replay (v5.37): the recorded trajectory's calls
    were GROUNDED (mutated in place), so raw-text replay mismatches by
    design. This verifier re-derives the RAW call from the generated
    text, applies a FRESH GroundingGate over it (same task grammar, so
    the grounding reproduces exactly), executes, and compares
    observations. Raises ReplayMismatch on any divergence.
    """
    try:
        from .gate import FixedGate, Route
        from .schema import ToolCallError, parse_tool_call
        from .tools import execute
        from .trajectory import ReplayMismatch, ids_to_text
    except ImportError:
        from gate import FixedGate, Route
        from schema import ToolCallError, parse_tool_call
        from tools import execute
        from trajectory import ReplayMismatch, ids_to_text

    gate = GroundingGate(FixedGate(Route.DIRECT))
    for s in traj.steps:
        text = ids_to_text(s.generated_ids)
        try:
            calls = parse_tool_call(text, registry)
        except ToolCallError as e:
            obs = f"PARSE_ERROR: {e}"
        else:
            call = calls[0]
            gate.decide(call, {"task": traj.task, "step": s.index})
            try:
                obs = execute(call, registry, impls)
            except ToolCallError as e:
                obs = f"TOOL_ERROR: {e}"
        if obs != s.observation:
            raise ReplayMismatch(
                f"step {s.index}: grounded replay observation differs:\n"
                f"  replay  : {obs!r}\n  recorded: {s.observation!r}")

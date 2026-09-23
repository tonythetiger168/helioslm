try:
    from .gate import Route
    from .schema import ToolCallError, ToolRegistry, parse_tool_call
    from .tools import execute
    from .trajectory import Step, Trajectory, text_to_ids
except ImportError:
    from gate import Route
    from schema import ToolCallError, ToolRegistry, parse_tool_call
    from tools import execute
    from trajectory import Step, Trajectory, text_to_ids

SYSTEM = ("You are an agent. Answer by emitting exactly one tool block of "
          "the form @@tool@@{\"calls\":[{\"name\":...,\"args\":{...}}]}@@end@@. "
          "Available tools: %%TOOLS%%.")


def render_tool_docs(registry: ToolRegistry) -> str:
    return "; ".join(
        f"{n}({', '.join(f'{k}: {t.__name__}' for k, t in s.params.items())})"
        f" — {s.description}" for n, s in registry.specs.items())


class AgentLoop:
    def __init__(self, model_fn, registry: ToolRegistry, impls: dict,
                 gate, max_steps: int = 8, confidence_fn=None):
        self.model_fn, self.registry, self.impls = model_fn, registry, impls
        self.gate, self.max_steps, self.confidence_fn = \
            gate, max_steps, confidence_fn

    def build_prompt(self, task_text: str, traj: Trajectory) -> str:
        parts = [SYSTEM.replace("%%TOOLS%%", render_tool_docs(self.registry)),
                 f"Task: {task_text}"]
        parts += [f"step {s.index}: {s.observation}" for s in traj.steps]
        return "\n".join(parts)

    def run(self, task_text: str, seed: int = 0) -> Trajectory:
        traj = Trajectory(task=task_text, seed=seed)
        for i in range(self.max_steps):
            prompt = self.build_prompt(task_text, traj)
            gen_text = self.model_fn(prompt, seed, i)
            try:
                call, perr = parse_tool_call(gen_text, self.registry)[0], None
            except ToolCallError as e:
                call, perr = None, str(e)
            if call is None:
                route_s, obs = "-", f"PARSE_ERROR: {perr}"
            else:
                ctx = {"step": i, "task": task_text,
                       "confidence": (self.confidence_fn(prompt, i)
                                      if self.confidence_fn else None)}
                route = self.gate.decide(call, ctx)
                route_s = route.value
                obs = (self.gate.escalate(call, ctx)
                       if route == Route.ESCALATE
                       else execute(call, self.registry, self.impls))
            traj.steps.append(Step(i, text_to_ids(prompt),
                                   text_to_ids(gen_text), call, perr,
                                   route_s, obs))
            if call is not None and call.name == "finish":
                traj.final_answer = (
                    call.args["answer"] if route_s == "DIRECT"
                    else obs.removeprefix("FINISH: "))
                break
        return traj

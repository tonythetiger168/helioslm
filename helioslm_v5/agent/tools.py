import ast
import operator
from pathlib import Path

try:
    from .schema import ToolCall, ToolCallError, ToolRegistry, ToolSpec
except ImportError:
    from schema import ToolCall, ToolCallError, ToolRegistry, ToolSpec

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
           ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_POW_EXP, _MAX_BITS = 64, 4096


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a, b = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if not isinstance(b, int) or abs(b) > _MAX_POW_EXP:
                raise ToolCallError("calc: exponent must be int "
                                    f"with |exp| <= {_MAX_POW_EXP}")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and b == 0:
            raise ToolCallError("calc: division by zero")
        r = _BINOPS[type(node.op)](a, b)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        r = _UNARYOPS[type(node.op)](_eval_node(node.operand))
    else:
        raise ToolCallError(f"calc: disallowed syntax {type(node).__name__}")
    if isinstance(r, int) and r.bit_length() > _MAX_BITS:
        raise ToolCallError("calc: result too large")
    return r


def calc(expr: str) -> str:
    if not isinstance(expr, str) or not expr.strip():
        raise ToolCallError("calc: empty expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ToolCallError(f"calc: syntax error: {e}") from e
    return repr(_eval_node(tree))


def str_op(s: str, op: str, n: int = 0) -> str:
    if not isinstance(s, str):
        raise ToolCallError("str_op: s must be a string")
    if op == "upper":
        return s.upper()
    if op == "lower":
        return s.lower()
    if op == "reverse":
        return s[::-1]
    if op == "repeat":
        if type(n) is not int or n < 0 or n > 10_000:
            raise ToolCallError("str_op: n must be int in [0, 10000]")
        return s * n
    raise ToolCallError(f"str_op: unknown op {op!r}")


def finish(answer: str) -> str:
    return f"FINISH: {answer}"


def build_default_registry(root="/tmp/helioslm_agent"):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def file_read(path: str) -> str:
        p = (root / path).resolve()
        if not str(p).startswith(str(root)):
            raise ToolCallError("file_read: path escapes sandbox")
        if not p.is_file():
            raise ToolCallError(f"file_read: no such file: {path}")
        return p.read_text(encoding="utf-8")

    def file_write(path: str, content: str) -> str:
        p = (root / path).resolve()
        if not str(p).startswith(str(root)):
            raise ToolCallError("file_write: path escapes sandbox")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"WROTE {len(content)} chars -> {path}"

    specs = [
        ToolSpec("calc", "evaluate an arithmetic expression", {"expr": str}),
        ToolSpec("str_op", "string transform: upper/lower/reverse/repeat",
                 {"s": str, "op": str, "n": int}),
        ToolSpec("file_read", "read a file inside the sandbox", {"path": str}),
        ToolSpec("file_write", "write a file inside the sandbox",
                 {"path": str, "content": str}),
        ToolSpec("finish", "submit the final answer and stop", {"answer": str}),
    ]
    impls = {"calc": calc, "str_op": str_op, "file_read": file_read,
             "file_write": file_write, "finish": finish}
    return ToolRegistry(specs), impls


def execute(call, registry: ToolRegistry, impls: dict) -> str:
    registry.spec(call.name)
    return str(impls[call.name](**call.args))

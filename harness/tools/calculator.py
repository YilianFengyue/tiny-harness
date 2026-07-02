"""calculator：基于 ast 白名单的安全求值。

不用 eval()：eval 的沙箱化是已被反复证伪的方向（dunder 链、builtins 注入），
ast 白名单从根上只允许算术结构存在。
"""
from __future__ import annotations

import ast
import math

from .registry import ToolContext, ToolError, tool

_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: _safe_pow(a, b),
}
_UNARYOPS = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}
_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def validate_calculator_input(ctx: ToolContext, arguments: dict) -> None:
    expression = arguments.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise ToolError("expression must be a non-empty arithmetic expression string")


def _safe_pow(a, b):
    if abs(b) > 10_000 or (abs(a) > 1 and abs(b) * math.log10(abs(a)) > 308):
        raise ToolError("exponent too large; result would overflow")
    return a ** b


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ToolError(f"only numbers allowed, got {type(node.value).__name__}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ToolError(f"unknown name '{node.id}'; allowed constants: {', '.join(_CONSTS)}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ToolError(f"unknown function; allowed: {', '.join(sorted(_FUNCS))}")
        if node.keywords:
            raise ToolError("keyword arguments are not supported")
        return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):  # 供 sum([...]) / min(...) 使用
        return [_eval_node(e) for e in node.elts]
    raise ToolError(f"unsupported syntax: {type(node).__name__}; "
                    "only arithmetic expressions are allowed")


@tool(
    name="calculator",
    description=(
        "Evaluate a single arithmetic expression and return the numeric result. "
        "Supports + - * / // % **, parentheses, constants pi/e, and functions: "
        "abs, round, min, max, sum (over a list literal like sum([1,2,3])), sqrt, "
        "log, log10, log2, exp, floor, ceil, sin, cos, tan. "
        "Use this for any numeric computation instead of doing mental arithmetic, "
        "and to double-check numbers you computed by other means. "
        "Not a Python interpreter: no variables, strings, or statements."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string",
                           "description": "Arithmetic expression, e.g. '(12.5 + 7) / 3' or 'sqrt(2)**2'"},
        },
    },
    read_only=True,
    concurrency_safe=True,
    validate_input=validate_calculator_input,
)
def calculator(ctx: ToolContext, expression: str) -> str:
    try:
        node = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"syntax error in expression: {e.msg} (offset {e.offset})")
    result = _eval_node(node)
    if isinstance(result, list):
        raise ToolError("expression evaluates to a list, not a number; wrap it in sum()/min()/max()")
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        return f"{expression} = {result:g}"
    return f"{expression} = {result!r}"

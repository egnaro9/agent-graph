"""The agent's tools — all deterministic, so runs are reproducible.

The calculator is a *safe* evaluator: it walks a parsed AST and only permits
arithmetic nodes. There is no ``eval()`` and no name/attribute/call access, so
``__import__('os')`` or ``os.system(...)`` is rejected rather than executed —
a small but real example of guarding an agent's tool surface.
"""
from __future__ import annotations

import ast
import operator as op
from typing import Callable, Dict

_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}
_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}


class ToolError(RuntimeError):
    """Raised for bad tool input — caught by the graph and recorded as an observation."""


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise ToolError(f"unsupported or unsafe expression element: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression safely (+ - * / // % ** and parens)."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {expression!r}") from exc
    value = _eval(tree)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


# A tiny knowledge base standing in for a real retrieval/search tool.
_KB = {
    "hamlet": "Hamlet was written by William Shakespeare.",
    "capital of france": "The capital of France is Paris.",
    "speed of light": "The speed of light is about 299,792 km/s.",
    "largest planet": "Jupiter is the largest planet in the Solar System.",
    "python creator": "Python was created by Guido van Rossum.",
}


def search(query: str) -> str:
    """Return the first knowledge-base fact whose keywords are all present."""
    q = query.lower()
    for key, fact in _KB.items():
        if all(word in q for word in key.split()):
            return fact
    return f"No results found for {query!r}."


def wordcount(text: str) -> str:
    return str(len(text.split()))


TOOLS: Dict[str, Callable[[str], str]] = {
    "calculator": calculator,
    "search": search,
    "wordcount": wordcount,
}


def run_tool(name: str, arg: str, tools: Dict[str, Callable[[str], str]] | None = None) -> str:
    """Execute a tool from ``tools`` (default: the built-in registry).

    The registry is injectable so a caller can swap an implementation without
    touching the graph — e.g. backing ``search`` with a real retriever instead
    of the toy knowledge base. See :mod:`agentgraph.rag`.
    """
    registry = TOOLS if tools is None else tools
    if name not in registry:
        raise ToolError(f"unknown tool: {name!r}")
    return registry[name](arg)

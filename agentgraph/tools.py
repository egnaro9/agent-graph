"""The agent's tools — all deterministic, so runs are reproducible.

The calculator is a *safe* evaluator: it walks a parsed AST and only permits
arithmetic nodes. There is no ``eval()`` and no name/attribute/call access, so
``__import__('os')`` or ``os.system(...)`` is rejected rather than executed —
a small but real example of guarding an agent's tool surface.
"""
from __future__ import annotations

import ast
import operator as op
import re
from typing import Callable, Dict, List, Protocol

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


# ── tools as objects: each one owns its own trigger ───────────────────────────
#
# A tool knows two things: how to run, and *when it applies*. Keeping the
# second half here — rather than in the planner — is what stops the policy from
# having to know that `search` happens to be backed by a five-entry dict. Swap
# in a real retriever and the trigger swaps with it, because it belongs to the
# tool. (It didn't always: the planner used to import this module's private
# _KB to decide whether to search, so a real corpus never triggered one.)


class Tool(Protocol):
    name: str

    def __call__(self, arg: str) -> str:
        """Run the tool."""

    def plan(self, query: str) -> List[Dict[str, str]]:
        """The calls this query implies for *this* tool. Empty = not applicable."""


# "X% of Y"
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)")
# an explicit arithmetic expression: numbers joined by + - * /
_EXPR_RE = re.compile(r"\d+(?:\.\d+)?(?:\s*[-+*/]\s*\d+(?:\.\d+)?)+")

_QUESTION_WORDS = ("what", "which", "who", "where", "when", "why", "how",
                   "is ", "are ", "does ", "did ")


def looks_like_a_question(query: str) -> bool:
    q = query.lower().strip()
    return q.endswith("?") or q.startswith(_QUESTION_WORDS)


class CalculatorTool:
    """Arithmetic. Applies when the query contains something to compute."""

    name = "calculator"

    def __call__(self, arg: str) -> str:
        return calculator(arg)

    def plan(self, query: str) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for m in _PCT_RE.finditer(query.lower()):
            pct, whole = m.group(1), m.group(2)
            out.append({"tool": self.name, "args": f"{pct}/100*{whole}",
                        "reason": f"compute {pct}% of {whole}"})
        for m in _EXPR_RE.finditer(query):
            expr = m.group(0).strip()
            out.append({"tool": self.name, "args": expr, "reason": f"evaluate {expr}"})
        return out


class KbSearchTool:
    """Lookup over a small fixed knowledge base.

    Applies only when the query mentions a topic it actually knows — which is
    the honest trigger for a lookup table, and exactly the wrong one for a real
    retriever. See :class:`agentgraph.rag.RagSearch`, which applies to any
    question because it has a corpus to search rather than keys to match.
    """

    name = "search"

    def __call__(self, arg: str) -> str:
        return search(arg)

    def plan(self, query: str) -> List[Dict[str, str]]:
        ql = query.lower()
        for key in _KB:
            if all(word in ql for word in key.split()):
                return [{"tool": self.name, "args": query, "reason": f"look up '{key}'"}]
        return []


class WordCountTool:
    name = "wordcount"

    def __call__(self, arg: str) -> str:
        return wordcount(arg)

    def plan(self, query: str) -> List[Dict[str, str]]:
        return []          # only ever called explicitly


# Registry order is plan order: compute before looking things up.
TOOLS: Dict[str, Tool] = {
    "calculator": CalculatorTool(),
    "search": KbSearchTool(),
    "wordcount": WordCountTool(),
}


def plan_tools(query: str, tools: Dict[str, Tool] | None = None) -> List[Dict[str, str]]:
    """Ask every tool in the registry what this query implies for it."""
    registry = TOOLS if tools is None else tools
    out: List[Dict[str, str]] = []
    for tool in registry.values():
        planner = getattr(tool, "plan", None)
        if planner is not None:
            out.extend(planner(query))
    return out


def run_tool(name: str, arg: str, tools: Dict[str, Tool] | None = None) -> str:
    """Execute a tool from ``tools`` (default: the built-in registry)."""
    registry = TOOLS if tools is None else tools
    if name not in registry:
        raise ToolError(f"unknown tool: {name!r}")
    return registry[name](arg)

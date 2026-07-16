"""The "brain" that decides the next action — a swappable policy.

``MockPolicy`` is a deterministic, rule-based planner: it derives an ordered
plan of tool calls from the query, then on each turn returns the next
un-executed action (given the observations gathered so far) or a final answer.
That determinism is what lets the whole agent be unit-tested.

``LLMPolicy`` (optional) delegates the same decision to a real model via
function-calling; the graph is identical either way.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Protocol

from .tools import _KB

# "X% of Y" -> compute a percentage
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)")
# an explicit arithmetic expression: numbers joined by + - * /
_EXPR_RE = re.compile(r"\d+(?:\.\d+)?(?:\s*[-+*/]\s*\d+(?:\.\d+)?)+")


class Policy(Protocol):
    def decide(self, query: str, observations: List[Dict[str, Any]]) -> Dict[str, Any]: ...
    def compose(self, query: str, observations: List[Dict[str, Any]]) -> str: ...


class MockPolicy:
    def plan(self, query: str) -> List[Dict[str, str]]:
        """Ordered tool calls implied by the query (deterministic)."""
        actions: List[Dict[str, str]] = []
        ql = query.lower()
        for m in _PCT_RE.finditer(ql):
            pct, whole = m.group(1), m.group(2)
            actions.append(
                {"tool": "calculator", "args": f"{pct}/100*{whole}",
                 "reason": f"compute {pct}% of {whole}"}
            )
        for m in _EXPR_RE.finditer(query):
            expr = m.group(0).strip()
            actions.append(
                {"tool": "calculator", "args": expr, "reason": f"evaluate {expr}"}
            )
        for key in _KB:
            if all(word in ql for word in key.split()):
                actions.append(
                    {"tool": "search", "args": query, "reason": f"look up '{key}'"}
                )
                break
        return actions

    def decide(self, query: str, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        plan = self.plan(query)
        if len(observations) < len(plan):
            return {"action": plan[len(observations)]}
        return {"final": self.compose(query, observations)}

    def compose(self, query: str, observations: List[Dict[str, Any]]) -> str:
        if not observations:
            return "I don't have a tool that helps with that. (mock policy)"
        return " ".join(str(o["result"]) for o in observations)


class LLMPolicy:  # pragma: no cover - optional real path
    """Function-calling policy backed by OpenAI. Same interface as MockPolicy."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI  # type: ignore

        self._client = OpenAI()
        self.model = model
        self._schema = [
            {
                "type": "function",
                "function": {
                    "name": t,
                    "description": f"call the {t} tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"arg": {"type": "string"}},
                        "required": ["arg"],
                    },
                },
            }
            for t in ("calculator", "search", "wordcount")
        ]

    def decide(self, query, observations):
        import json

        context = "\n".join(f"{o['tool']}({o['args']}) -> {o['result']}" for o in observations)
        messages = [
            {"role": "system", "content": "Use tools to answer. Call a tool or answer directly."},
            {"role": "user", "content": f"Question: {query}\nObservations so far:\n{context}"},
        ]
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, tools=self._schema, temperature=0
        )
        choice = resp.choices[0].message
        if choice.tool_calls:
            call = choice.tool_calls[0]
            arg = json.loads(call.function.arguments).get("arg", "")
            return {"action": {"tool": call.function.name, "args": arg, "reason": "llm"}}
        return {"final": choice.content or ""}

    def compose(self, query, observations):
        return " ".join(str(o["result"]) for o in observations)

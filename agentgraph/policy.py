"""The "brain" that decides the next action — a swappable policy.

``MockPolicy`` is a deterministic, rule-based planner, and it deliberately
knows nothing about what any individual tool does: it asks the tool registry
what a query implies and sequences the answers.

That separation is load-bearing. The planner used to reach into the search
tool's private knowledge base to decide whether searching was worth it — so
swapping in a real retriever produced an agent that silently never searched.
A tool's trigger belongs to the tool; the planner only orders the result.

``LLMPolicy`` (optional) delegates the same decision to a real model via
function-calling; the graph is identical either way.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from .tools import Tool, plan_tools


class Policy(Protocol):
    def decide(self, query: str, observations: List[Dict[str, Any]]) -> Dict[str, Any]: ...
    def compose(self, query: str, observations: List[Dict[str, Any]]) -> str: ...


class MockPolicy:
    """Deterministic planner: ask the tools, sequence their answers.

    :param tools: the registry to plan against. Defaults to the built-ins.
        Hand it :func:`agentgraph.rag.rag_tools` and the plan changes on its
        own — because the retriever's trigger differs from the lookup table's.
        No flag, no branch here.
    """

    def __init__(self, tools: Optional[Dict[str, Tool]] = None) -> None:
        self.tools = tools

    def plan(self, query: str) -> List[Dict[str, str]]:
        """Ordered tool calls implied by the query (deterministic).

        The planner contributes sequencing, not knowledge — every rule about
        *when* a tool applies lives with that tool.
        """
        return plan_tools(query, self.tools)

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

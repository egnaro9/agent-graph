"""The LangGraph agent: a ReAct-style loop with a max-step guard.

    START -> agent --(action)--> tools --> agent --(final|guard)--> END

`agent` (the policy) decides to call a tool or finish; `tools` executes the
chosen tool and records an observation; control loops back. A ``max_steps``
guard forces a finish so a mis-behaving policy can never loop forever.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from .policy import MockPolicy, Policy
from .state import AgentState
from .tools import ToolError, run_tool

DEFAULT_MAX_STEPS = 6


def build_graph(policy: Optional[Policy] = None, max_steps: int = DEFAULT_MAX_STEPS,
                tools_registry: Optional[Dict[str, Any]] = None):
    # Plan against the registry we'll execute with — otherwise a swapped-in
    # tool (a real retriever, say) would never be planned for, only run.
    policy = policy or MockPolicy(tools=tools_registry)

    def agent(state: AgentState) -> Dict[str, Any]:
        step = state.get("step_count", 0)
        obs = state.get("observations", [])
        limit = state.get("max_steps", max_steps)
        # Guardrail: never exceed the step budget.
        if step >= limit:
            return {
                "next_action": None,
                "answer": policy.compose(state["query"], obs),
                "steps": [{"type": "finish", "reason": "max_steps guard reached"}],
                "step_count": step + 1,
            }
        decision = policy.decide(state["query"], obs)
        if "final" in decision:
            return {
                "next_action": None,
                "answer": decision["final"],
                "steps": [{"type": "final", "answer": decision["final"]}],
                "step_count": step + 1,
            }
        action = decision["action"]
        return {
            "next_action": action,
            "steps": [{"type": "action", **action}],
            "step_count": step + 1,
        }

    def tools(state: AgentState) -> Dict[str, Any]:
        action = state["next_action"]
        assert action is not None
        try:
            result = run_tool(action["tool"], action["args"], tools_registry)
            obs = {"tool": action["tool"], "args": action["args"], "result": result}
        except ToolError as exc:
            obs = {
                "tool": action["tool"],
                "args": action["args"],
                "result": f"ERROR: {exc}",
                "error": True,
            }
        return {
            "observations": [obs],
            "steps": [{"type": "observation", **obs}],
            "next_action": None,
        }

    def route(state: AgentState) -> str:
        return "tools" if state.get("next_action") else END

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def run(query: str, max_steps: int = DEFAULT_MAX_STEPS, policy: Optional[Policy] = None,
        tools_registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the agent to completion and return the final state (answer + trace)."""
    app = build_graph(policy=policy, max_steps=max_steps, tools_registry=tools_registry)
    return app.invoke(
        {
            "query": query,
            "steps": [],
            "observations": [],
            "step_count": 0,
            "max_steps": max_steps,
        }
    )

from agentgraph.graph import build_graph, run


def _actions(state):
    return [s for s in state["steps"] if s["type"] == "action"]


def test_single_math_query():
    state = run("What is 12 * 8?")
    assert "96" in state["answer"]
    acts = _actions(state)
    assert len(acts) == 1 and acts[0]["tool"] == "calculator"


def test_single_search_query():
    state = run("Who wrote Hamlet?")
    assert "Shakespeare" in state["answer"]
    assert _actions(state)[0]["tool"] == "search"


def test_multi_step_tool_use():
    state = run("What is 15% of 240 and who wrote Hamlet?")
    assert "36" in state["answer"]
    assert "Shakespeare" in state["answer"]
    # Two tool calls: a calculator then a search.
    tools_used = [a["tool"] for a in _actions(state)]
    assert tools_used == ["calculator", "search"]
    assert len(state["observations"]) == 2


def test_determinism():
    q = "What is 15% of 240 and who wrote Hamlet?"
    assert run(q)["answer"] == run(q)["answer"]


def test_no_tool_query_still_finishes():
    state = run("hello there")
    assert state["answer"]  # non-empty
    assert _actions(state) == []


class _AlwaysActPolicy:
    """A deliberately broken policy that never finishes — the guard must stop it."""

    def decide(self, query, observations):
        return {"action": {"tool": "wordcount", "args": query, "reason": "loop"}}

    def compose(self, query, observations):
        return "stopped by guard"


def test_max_steps_guard_prevents_infinite_loop():
    app = build_graph(policy=_AlwaysActPolicy(), max_steps=3)
    state = app.invoke(
        {"query": "x", "steps": [], "observations": [], "step_count": 0, "max_steps": 3}
    )
    assert state["answer"] == "stopped by guard"
    # agent runs at most max_steps+1 times (the +1 is the guard turn).
    assert state["step_count"] <= 4
    assert any(s["type"] == "finish" for s in state["steps"])


def test_trace_is_ordered_action_then_observation():
    state = run("What is 12 * 8?")
    types = [s["type"] for s in state["steps"]]
    assert types[0] == "action"
    assert types[1] == "observation"
    assert types[-1] == "final"

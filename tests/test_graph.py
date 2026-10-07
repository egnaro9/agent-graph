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


# ---------------------------------------------------------------------------
# Two fields nothing read.
#
# A mutation audit flipped the tool-error flag to False and cut DEFAULT_MAX_STEPS from 6 to 2,
# and all 42 tests stayed green.

def test_a_tool_error_is_flagged_as_one():
    """`obs["error"] = True` could be flipped to False with the suite green. The existing
    test asserts only that 'division by zero' appears in str(out), which the result STRING
    satisfies regardless of the flag.

    There is a real consumer: demo/app.js:192 renders a "(tool error)" badge from this field,
    so under the mutation the demo silently presents a failed tool call as a successful one.
    """
    state = run("what is 4/0")
    errs = [s for s in state["steps"] if s["type"] == "observation" and "ERROR" in str(s.get("result"))]
    assert errs, "the divide-by-zero observation is missing"
    assert errs[0].get("error") is True, (
        f"a failed tool call is not flagged: {errs[0]}")


def test_a_successful_tool_call_is_not_flagged_as_an_error():
    """The other side, so the flag cannot be satisfied by always setting it."""
    state = run("What is 12 * 8?")
    obs = [s for s in state["steps"] if s["type"] == "observation"]
    assert obs and not any(s.get("error") for s in obs)


def test_the_step_budget_fits_the_queries_this_agent_answers():
    """DEFAULT_MAX_STEPS could be cut from 6 to 2 invisibly, because no test distinguishes a
    completed plan from an exhausted one.

    A three-tool query is the case that separates them: under a budget of 2 the third tool is
    dropped and the user gets a silently truncated answer, not merely a different trace label.
    """
    state = run("What is 15% of 240 and 20% of 50 and who wrote Hamlet?")
    tools = [a["tool"] for a in _actions(state)]
    assert tools == ["calculator", "calculator", "search"], tools
    assert "36" in state["answer"] and "10" in state["answer"]
    assert "Shakespeare" in state["answer"], (
        f"the search was dropped, so the answer is truncated: {state['answer']!r}")


def test_a_plan_that_finishes_is_not_reported_as_exhausted():
    """The terminating step type distinguishes 'the agent is done' from 'the agent ran out',
    and nothing asserted it."""
    state = run("What is 12 * 8?")
    assert state["steps"][-1]["type"] == "final", state["steps"][-1]

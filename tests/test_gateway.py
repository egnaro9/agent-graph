"""GatewayPolicy: the agent's LLM traffic goes through llm-gateway.

The transport is faked here so these stay offline and deterministic — the point
under test is the agent-side contract (what it sends, what it does with the
answer, how it behaves when the gateway misbehaves), not the gateway itself,
which has its own suite.
"""
import pytest

from agentgraph.gateway import GatewayPolicy
from agentgraph.graph import build_graph, run


def fake_gateway(reply="Venus is the hottest planet.", cached=False, cost=0.0002, record=None):
    def _send(path, body, api_key):
        if record is not None:
            record.append({"path": path, "body": body, "api_key": api_key})
        return {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            "cached": cached,
            "cost_usd": cost,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    return _send


def test_compose_goes_through_the_gateway():
    sent = []
    p = GatewayPolicy(transport=fake_gateway(record=sent))
    out = p.compose("Which planet is hottest?", [{"tool": "search", "args": "q", "result": "Venus is hottest."}])
    assert out == "Venus is the hottest planet."
    assert len(sent) == 1
    assert sent[0]["path"] == "/v1/chat/completions"
    assert sent[0]["api_key"] == "dev-key"


def test_sends_temperature_zero_so_the_gateway_may_cache():
    sent = []
    p = GatewayPolicy(transport=fake_gateway(record=sent))
    p.compose("q", [{"tool": "search", "args": "a", "result": "r"}])
    # Caching a sampled completion would be wrong; determinism is the precondition.
    assert sent[0]["body"]["temperature"] == 0.0


def test_tool_results_are_passed_to_the_model():
    sent = []
    p = GatewayPolicy(transport=fake_gateway(record=sent))
    p.compose("Which planet?", [{"tool": "calculator", "args": "2+2", "result": "4"}])
    content = sent[0]["body"]["messages"][0]["content"]
    assert "calculator(2+2) = 4" in content


def test_surfaces_what_the_gateway_did():
    p = GatewayPolicy(transport=fake_gateway(cached=True, cost=0.0))
    p.compose("q", [{"tool": "search", "args": "a", "result": "r"}])
    assert p.cached is True
    assert p.cost_usd == 0.0


def test_tool_selection_stays_deterministic():
    # The gateway is never consulted for WHICH tool to call.
    sent = []
    p = GatewayPolicy(transport=fake_gateway(record=sent))
    d = p.decide("What is 12 * 8?", [])
    assert d["action"]["tool"] == "calculator"
    assert sent == []          # no LLM call to pick a tool


def test_unreachable_gateway_degrades_instead_of_losing_the_run():
    def boom(path, body, api_key):
        raise ConnectionError("gateway down")

    p = GatewayPolicy(transport=boom)
    obs = [{"tool": "search", "args": "a", "result": "Hamlet was written by Shakespeare."}]
    out = p.compose("Who wrote Hamlet?", obs)
    # Tool work is already done — an unreachable gateway must not throw it away.
    assert "Shakespeare" in out
    assert "error" in p.last_response


def test_malformed_gateway_response_degrades():
    p = GatewayPolicy(transport=lambda path, body, key: {"detail": "rate limit exceeded"})
    out = p.compose("Who wrote Hamlet?", [{"tool": "search", "args": "a", "result": "Shakespeare wrote it."}])
    assert "Shakespeare" in out


def test_no_observations_needs_no_gateway_call():
    sent = []
    p = GatewayPolicy(transport=fake_gateway(record=sent))
    assert "don't have a tool" in p.compose("hello", [])
    assert sent == []


def test_end_to_end_through_the_graph():
    # The whole LangGraph run, with the compose step served by the gateway.
    policy = GatewayPolicy(transport=fake_gateway(reply="36, and Shakespeare wrote Hamlet."))
    app = build_graph(policy=policy, max_steps=6)
    state = app.invoke({"query": "What is 15% of 240 and who wrote Hamlet?",
                        "steps": [], "observations": [], "step_count": 0, "max_steps": 6})
    assert state["answer"] == "36, and Shakespeare wrote Hamlet."
    tools_used = [s["tool"] for s in state["steps"] if s["type"] == "action"]
    assert tools_used == ["calculator", "search"]   # real tools still ran

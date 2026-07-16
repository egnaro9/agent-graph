"""The agent's search tool backed by rag-eval-lab's retriever (arrow 1 of the stack)."""
import pytest

ragevallab = pytest.importorskip("ragevallab", reason="optional rag extra")

from agentgraph.graph import run
from agentgraph.rag import RagSearch, rag_policy, rag_tools
from agentgraph.tools import run_tool


def test_retrieves_instead_of_looking_up():
    s = RagSearch()
    out = s("Which planet is the hottest?")
    assert "Venus" in out
    # The contexts are kept — an eval harness needs them to judge grounding.
    assert s.last_contexts and s.last_retrieved
    assert s.last_retrieved[0] == "venus#0"


def test_registry_swaps_only_search():
    tools = rag_tools()
    assert isinstance(tools["search"], RagSearch)
    assert tools["calculator"]("2+2") == "4"      # untouched


def test_graph_runs_with_the_real_retriever():
    state = run("Who wrote Hamlet?", policy=rag_policy(), tools_registry=rag_tools())
    # The planet corpus has nothing about Hamlet — a REAL retriever returns its
    # closest chunk anyway rather than admitting defeat. That's the honest
    # failure mode the toy tool hid, and exactly what faithfulness catches.
    assert state["answer"]
    assert [s["tool"] for s in state["steps"] if s["type"] == "action"] == ["search"]


def test_real_retrieval_answers_an_in_corpus_question():
    state = run("Which planet is the hottest?", policy=rag_policy(), tools_registry=rag_tools())
    assert "Venus" in state["answer"]


def test_custom_corpus():
    s = RagSearch(docs={"cats": "Cats sleep about sixteen hours a day.",
                        "dogs": "Dogs are descended from wolves."})
    assert "wolves" in s("Where do dogs come from?")


def test_run_tool_accepts_an_injected_registry():
    assert "Venus" in run_tool("search", "hottest planet", rag_tools())


def test_default_planner_still_ignores_non_questions():
    from agentgraph.policy import MockPolicy
    assert MockPolicy(always_search=True).plan("hello there") == []


def test_always_search_does_not_change_the_default_planner():
    from agentgraph.policy import MockPolicy
    assert MockPolicy().plan("Which planet is the hottest?") == []   # no KB key matches
    assert MockPolicy(always_search=True).plan("Which planet is the hottest?")[0]["tool"] == "search"

# agent-graph

[![ci](https://github.com/egnaro9/agent-graph/actions/workflows/ci.yml/badge.svg)](https://github.com/egnaro9/agent-graph/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/built%20with-LangGraph-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![live demo](https://img.shields.io/badge/demo-run%20the%20agent%20in%20your%20browser-f2a53c)](https://egnaro9.github.io/agent-graph/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A LangGraph ReAct agent with deterministic, safety-guarded tools — a multi-step tool-using agent you can actually unit-test.**

Agents are hard to test because the model is nondeterministic. This repo separates the two concerns: the **graph** (the orchestration — nodes, conditional edges, state, the tool loop, the step-budget guard) is real LangGraph and fully deterministic; the **policy** (the "which tool next" brain) is a swappable interface. A rule-based `MockPolicy` makes the whole agent reproducible and CI-testable with no API key; an `LLMPolicy` drops a real function-calling model into the exact same graph.

```
START ─► agent ──(tool call)──► tools ──► agent ──(final answer │ step-guard)──► END
```

- **Real LangGraph.** `StateGraph` with append-only `steps`/`observations` reducers, a conditional `agent → tools → agent` loop, and a compiled app you `invoke`.
- **Multi-step tool use.** *"What is 15% of 240 and who wrote Hamlet?"* → the agent calls `calculator`, then `search`, then composes the answer — and the full trace is returned.
- **Guardrails that matter — mapped to [OWASP LLM06 (Excessive Agency)](https://genai.owasp.org/llmrisk/llm06-2025-excessive-agency/).** A **safe calculator** (AST allow-list, so `__import__('os')` is rejected, not executed) and a **max-step budget** so a mis-behaving policy can never loop forever — limited tool functionality and limited autonomy, the two mitigations OWASP names. Both are unit-tested.
- **Deterministic & offline.** **17 tests, green CI, no secrets.**

### ▶ [Run the agent in your browser](https://egnaro9.github.io/agent-graph/)

Real LangGraph, compiled to WebAssembly via [Pyodide](https://pyodide.org) — watch the `agent ⇄ tools` loop pick its tools live, then **try to break the guardrail**: throw `__import__('os').system('ls')` at the calculator and watch it get rejected.

---

## How it works

```mermaid
flowchart LR
    S((START)) --> AG[agent<br/>policy decides]
    AG -->|action| T[tools<br/>calculator · search · wordcount]
    T --> AG
    AG -->|final answer| E((END))
    AG -->|step budget hit| E
```

Each turn appends to a trace, so you can see exactly what the agent did:

```bash
pip install -e ".[dev]"
python -m agentgraph.cli run "What is 15% of 240 and who wrote Hamlet?"
```
```
Q: What is 15% of 240 and who wrote Hamlet?
  → call calculator('15/100*240')  [compute 15% of 240]
    = 36
  → call search('What is 15% of 240 and who wrote Hamlet?')  [look up 'hamlet']
    = Hamlet was written by William Shakespeare.
A: 36 Hamlet was written by William Shakespeare.
```

`run()` returns the final state — `answer`, the full `steps` trace, and every tool `observation` — so it's easy to assert on in a test:

```python
from agentgraph import run
state = run("What is 12 * 8?")
assert "96" in state["answer"]
assert [s for s in state["steps"] if s["type"] == "action"][0]["tool"] == "calculator"
```

## Guardrails, mapped to the OWASP LLM Top 10

Both guardrails are concrete mitigations for **[LLM06: Excessive Agency](https://genai.owasp.org/llmrisk/llm06-2025-excessive-agency/)** — the risk that an LLM, given a tool, does more with it than intended. The mitigation OWASP names is *limit tool functionality and limit autonomy*; that's exactly these two.

**Safe calculator — minimal tool functionality (LLM06), and no prompt-injection → RCE (LLM01).**
```python
from agentgraph import calculator, ToolError
calculator("2 + 3 * 4")           # "14"
calculator("__import__('os')")     # raises ToolError — names/calls are not allowed
```
The naive version of this tool is `eval(expression)` — a remote-code-execution hole one crafted model output away. Instead it parses to an `ast` and walks an **allow-list of arithmetic nodes only** ([`tools.py`](agentgraph/tools.py)); a function call, attribute access, or name is rejected before anything executes. The tool can do arithmetic and *nothing else*, so a manipulated or injected instruction can't escalate it.

**Step budget — bounded autonomy (LLM06).**
```python
build_graph(policy=AlwaysActsPolicy(), max_steps=3)   # the agent node forces a finish at the budget
```
An agent that can loop forever is an agent that can exhaust cost and resources on a single request. A hard step budget caps the autonomy of *any* policy, well-behaved or not. See [`test_graph.py::test_max_steps_guard_prevents_infinite_loop`](tests/test_graph.py).

Both are unit-tested, so the mitigations can't silently regress — the point of a guardrail is that it stays a guardrail.

## Composing with rag-eval-lab

The built-in `search` tool is a five-entry dict — fine for testing the graph, useless as retrieval. [`agentgraph.rag`](agentgraph/rag.py) swaps it for [rag-eval-lab](https://github.com/egnaro9/rag-eval-lab)'s pipeline, which is what a RAG-backed agent *is*:

```python
from agentgraph.graph import run
from agentgraph.rag import rag_tools

state = run("Which planet is the hottest?", tools_registry=rag_tools())
```

That's the whole swap — no policy change, no flag. The planner adapts because **each tool owns its own trigger**: a lookup table only applies to keys it knows, a retriever applies to any question.

Two things this buys beyond a nicer demo. **The agent's answers become gradeable** — retrieval now yields contexts, and grounding can only be measured against the context that was actually retrieved. And **real failure modes appear**: the toy tool either finds the fact or says "no results", while a real retriever confidently returns the *wrong* chunk — which is precisely what faithfulness scoring exists to catch.

It also exposed a real design flaw worth describing, because the first fix was the wrong one. `policy.py` imported the search tool's private `_KB` to decide whether searching was worthwhile — so a real corpus produced an agent that **silently never searched**. The quick fix was an `always_search` flag; the actual fix was to notice the coupling: a tool knows two things, *how to run* and *when it applies*, and the planner had been holding the second half for it. Now tools implement `plan(query)`, `MockPolicy` just sequences what they return, and the flag is gone. A regression test asserts `policy.py` never mentions `_KB` again.

## Composing with llm-gateway

An agent is a *chatty* LLM client — one call per turn, and multi-step runs repeat near-identical prompts constantly. [`GatewayPolicy`](agentgraph/gateway.py) points those calls at [llm-gateway](https://github.com/egnaro9/llm-gateway), which then authenticates, caches, retries and cost-accounts every one of them without the agent knowing:

```python
from agentgraph.gateway import GatewayPolicy
from agentgraph.graph import build_graph

app = build_graph(policy=GatewayPolicy(base_url="http://localhost:8000", model="mock-1"))
```

Ask the same question twice and the second run's LLM call is a **cache hit** — the provider is never touched. [Try it in the browser](https://egnaro9.github.io/agent-graph/), where both projects run in one tab.

Two things stay honest here: **tool selection remains deterministic** (the rule-based planner, so the graph stays reproducible) — the gateway governs the *compose* call, where a model turns tool results into prose. And if the gateway is unreachable, the policy **degrades to composing locally** rather than throwing away work the tools already did. Both are tested.

## Swapping in a real model

The graph doesn't change — only the policy does:
```python
from agentgraph.graph import build_graph
from agentgraph.policy import LLMPolicy      # needs: pip install -e ".[openai]" + OPENAI_API_KEY

app = build_graph(policy=LLMPolicy(model="gpt-4o-mini"))
app.invoke({"query": "...", "steps": [], "observations": [], "step_count": 0, "max_steps": 6})
```
`LLMPolicy` uses OpenAI function-calling to pick tools; `MockPolicy` uses deterministic rules. Same `decide()` / `compose()` interface, same loop, same guardrails.

## Layout

```
agentgraph/
  graph.py    build_graph() -> compiled LangGraph; run(query) -> final state
  state.py    AgentState (TypedDict) with append-only trace reducers
  policy.py   MockPolicy (deterministic) · LLMPolicy (optional, function-calling)
  tools.py    calculator (safe AST eval) · search (KB) · wordcount · run_tool
  cli.py      `run "<query>"` · `demo`
tests/        17 tests — tools, safety, multi-step traces, the step-budget guard
```

## Design notes

- **Why split graph from policy?** So the deterministic part (orchestration, guardrails, tracing) is testable without a model, and the model is a drop-in. This is the pattern that makes agent code maintainable: the risky, nondeterministic piece is isolated behind one interface.
- **Why a rule-based mock instead of a stub?** The mock actually *plans* (parses the query into an ordered tool sequence), so it exercises the real multi-step loop — not just a single hop.

## Run it in Docker

```bash
docker build -t agent-graph . && docker run --rm agent-graph   # runs the demo
```

---

Built by [Erik Hill](https://egnaro9.github.io) · MIT licensed.

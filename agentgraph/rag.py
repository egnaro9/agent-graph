"""Back the agent's ``search`` tool with a real retriever.

The built-in ``search`` tool is a five-entry dict — fine for testing the graph,
useless as retrieval. This module swaps it for
[rag-eval-lab](https://github.com/egnaro9/rag-eval-lab)'s pipeline, which is
what a RAG-backed agent actually is: the tool retrieves from a corpus instead
of looking up a hard-coded answer.

Two things this buys beyond a better demo:

- **The agent's answers become gradeable.** rag-eval-lab scores an answer's
  *grounding* against the context that was retrieved for it. Once retrieval
  runs through that pipeline, you have the contexts, so you can measure whether
  the agent's answer is actually supported — which is the only honest way to
  evaluate a RAG agent.
- **Real retrieval failure modes appear.** The toy tool always finds the fact
  or says "no results". A real retriever returns the *wrong* chunk, or a
  lexically-close-but-irrelevant one, and the agent has to live with it.

Optional dependency::

    pip install "agentgraph[rag]"     # or: pip install ragevallab
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .tools import TOOLS, calculator, wordcount


class RagSearch:
    """A ``search`` tool that retrieves from a corpus via rag-eval-lab.

    Keeps the last retrieval's contexts and chunk ids, because that's what an
    eval harness needs afterwards to judge whether the answer was grounded.
    """

    def __init__(self, docs: Optional[Dict[str, str]] = None, k: int = 3) -> None:
        try:
            from ragevallab.data import SAMPLE_DOCS
            from ragevallab.pipeline import RagPipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "RagSearch needs rag-eval-lab: pip install ragevallab"
            ) from exc

        self.k = k
        self.pipeline = RagPipeline().ingest(docs if docs is not None else SAMPLE_DOCS)
        self.last_contexts: List[str] = []
        self.last_retrieved: List[str] = []

    def __call__(self, query: str) -> str:
        records = self.pipeline.retrieve(query, k=self.k)
        self.last_contexts = [r.text for r in records]
        self.last_retrieved = [r.id for r in records]
        if not records:
            return f"No results found for {query!r}."
        # The tool's job is to retrieve, not to decide — hand back the best
        # chunk verbatim and let the policy compose from it.
        return records[0].text


def rag_policy():
    """The planner to pair with :func:`rag_tools`.

    The default planner only searches when the query hits the toy knowledge
    base's keys — a trigger that makes no sense once ``search`` is a real
    retriever. This one retrieves for any question.
    """
    from .policy import MockPolicy

    return MockPolicy(always_search=True)


def rag_tools(docs: Optional[Dict[str, str]] = None, k: int = 3) -> Dict[str, Callable[[str], str]]:
    """The standard tool registry with ``search`` backed by real retrieval.

    >>> from agentgraph.graph import run
    >>> from agentgraph.rag import rag_tools
    >>> state = run("Which planet is the hottest?", tools_registry=rag_tools())
    """
    return {**TOOLS, "search": RagSearch(docs=docs, k=k)}

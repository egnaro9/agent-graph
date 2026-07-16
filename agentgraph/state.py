"""The agent's graph state.

``steps`` and ``observations`` use ``operator.add`` reducers so every node can
*append* to them and LangGraph merges the partial updates — that append-only
trace is what makes each run fully inspectable.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    query: str
    steps: Annotated[List[Dict[str, Any]], operator.add]          # full trace
    observations: Annotated[List[Dict[str, Any]], operator.add]   # tool results
    next_action: Optional[Dict[str, Any]]                         # set by agent, consumed by tools
    answer: str
    step_count: int
    max_steps: int

"""CLI: ``python -m agentgraph.cli run "your question"`` or ``... demo``."""
from __future__ import annotations

import argparse
import sys
from typing import List

from .graph import run


def _print_run(query: str) -> None:
    state = run(query)
    print(f"Q: {query}")
    for step in state["steps"]:
        t = step["type"]
        if t == "action":
            print(f"  → call {step['tool']}({step['args']!r})  [{step.get('reason','')}]")
        elif t == "observation":
            print(f"    = {step['result']}")
        elif t == "finish":
            print(f"  (finish: {step['reason']})")
    print(f"A: {state['answer']}\n")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agentgraph")
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="answer a single query")
    r.add_argument("query")
    sub.add_parser("demo", help="run a few example queries")
    args = p.parse_args(argv)

    if args.cmd == "run":
        _print_run(args.query)
        return 0
    if args.cmd == "demo":
        for q in (
            "What is 12 * 8?",
            "Who wrote Hamlet?",
            "What is 15% of 240 and who wrote Hamlet?",
        ):
            _print_run(q)
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

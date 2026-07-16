"""Route the agent's LLM calls through an llm-gateway instead of a provider SDK.

Why a gateway sits here at all: an agent is a *chatty* LLM client. It calls a
model once per turn, and multi-step runs repeat near-identical prompts
constantly — the same question asked twice costs twice, and nothing tracks
either call. Pointing the agent at
`[llm-gateway](https://github.com/egnaro9/llm-gateway)` puts auth, per-key rate
limiting, response caching, retries and per-model cost accounting in front of
every call the agent makes, without the agent knowing any of it happened.

What this policy does and doesn't do, plainly:

- **Tool selection stays deterministic.** It reuses :class:`MockPolicy`'s
  rule-based planner, so the graph's behavior is still reproducible and
  testable. Swap in :class:`LLMPolicy` if you want the model choosing tools.
- **The LLM call is the compose step** — turning the tool observations into a
  natural-language answer, which is genuinely what a model is good for here.
  That call is the traffic the gateway governs.

The transport is injectable, which is what lets the same code run against a
real gateway over HTTP *and* against an in-process ASGI app in a browser demo.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from .policy import MockPolicy

# A transport takes (path, json_body, api_key) and returns the parsed response.
Transport = Callable[[str, dict, str], dict]


def http_transport(base_url: str) -> Transport:  # pragma: no cover - needs a live gateway
    """Talk to a real llm-gateway over HTTP."""
    import urllib.request

    def _send(path: str, body: dict, api_key: str) -> dict:
        req = urllib.request.Request(
            base_url.rstrip("/") + path,
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json", "authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    return _send


class GatewayPolicy:
    """A policy whose LLM traffic goes through an llm-gateway.

    :param transport: how to reach the gateway. Defaults to HTTP against
        ``base_url``; a browser demo passes an in-process ASGI caller instead.
    :param model: the gateway model id. ``mock-1`` is the gateway's offline
        deterministic provider, so this whole path runs with no key and no spend.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "dev-key",
        model: str = "mock-1",
        transport: Optional[Transport] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self._transport = transport or http_transport(base_url)
        self._planner = MockPolicy()
        self.last_response: Dict[str, Any] = {}

    # Tool selection: unchanged, deterministic, testable.
    def decide(self, query: str, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        plan = self._planner.plan(query)
        if len(observations) < len(plan):
            return {"action": plan[len(observations)]}
        return {"final": self.compose(query, observations)}

    def build_request(self, query: str, observations: List[Dict[str, Any]]) -> dict:
        """The chat-completions body this policy sends for the compose step.

        Public and separate from :meth:`compose` so an async caller can reuse
        the exact same request without duplicating how it's built — see the
        browser demo, where a WebAssembly runtime has no threads and therefore
        can't drive the gateway's async app from a sync graph node.
        """
        facts = "; ".join(f"{o['tool']}({o['args']}) = {o['result']}" for o in observations)
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Question: {query}\nTool results: {facts}\nAnswer using only those results.",
                }
            ],
            "temperature": 0.0,  # deterministic -> the gateway is allowed to cache it
        }

    def compose(self, query: str, observations: List[Dict[str, Any]]) -> str:
        """Ask the model — through the gateway — to phrase the answer."""
        if not observations:
            return "I don't have a tool that helps with that."

        body = self.build_request(query, observations)
        try:
            resp = self._transport("/v1/chat/completions", body, self.api_key)
        except Exception as exc:  # gateway down / rate limited / no key
            # Degrade to the local composer rather than failing the run: the
            # tool results are already in hand, so an unreachable gateway
            # shouldn't lose the agent's work.
            self.last_response = {"error": str(exc)}
            return self._planner.compose(query, observations)

        self.last_response = resp
        if "choices" not in resp:
            return self._planner.compose(query, observations)
        return resp["choices"][0]["message"]["content"]

    # ── what the gateway did with our traffic ─────────────────────────────────

    @property
    def cached(self) -> bool:
        return bool(self.last_response.get("cached"))

    @property
    def cost_usd(self) -> float:
        return float(self.last_response.get("cost_usd", 0.0))

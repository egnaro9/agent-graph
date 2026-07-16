// Boots Pyodide, installs real LangGraph + the agentgraph wheel, and runs the
// actual StateGraph in the browser. Nothing here reimplements the agent.

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const statusText = $("statusText");
const setStatus = (t, s) => { statusText.textContent = t; statusEl.className = "status" + (s ? " " + s : ""); };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let py = null;

async function boot() {
  try {
    setStatus("Booting Python (WebAssembly)…");
    py = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v314.0.2/full/" });
    await py.loadPackage("micropip");
    const micropip = py.pyimport("micropip");

    setStatus("Installing LangGraph…");
    // ormsgpack (checkpoint serde) and websockets (remote SDK client) ship no
    // WASM wheels. This agent uses neither, so install without dep resolution
    // and shim them — see the note at the bottom of the page.
    for (const p of ["langgraph", "langgraph-checkpoint", "langgraph-sdk", "langgraph-prebuilt"]) {
      await micropip.install.callKwargs(p, { deps: false });
    }
    await micropip.install(["langchain-core", "pydantic", "xxhash"]);

    setStatus("Installing the agentgraph wheel…");
    await micropip.install(window.WHEEL_URL || "./agentgraph-0.1.0-py3-none-any.whl");

    // The companion project, fetched from ITS OWN published Pages build — the
    // same wheel its CI ships. Same origin, so the agent can genuinely call it.
    setStatus("Installing llm-gateway (the companion project)…");
    await micropip.install(["fastapi"]);
    await micropip.install.callKwargs(
      window.GATEWAY_WHEEL_URL || "https://egnaro9.github.io/llm-gateway/llmgateway-0.1.0-py3-none-any.whl",
      { deps: false }
    );

    setStatus("Compiling the graph…");
    await py.runPythonAsync(`
import sys, types, json

class _Shim(types.ModuleType):
    """Stands in for a dependency with no WASM build.

    Only satisfies import-time access. Any real call raises rather than
    silently faking behavior — if an unused path ever runs, it fails loudly.
    """
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name.startswith("OPT_"):
            return 1
        if name.endswith(("Error", "Exception", "OK", "Closed")) or name.startswith("Connection"):
            cls = type(name, (Exception,), {})
            setattr(self, name, cls)
            return cls
        def _unavailable(*a, **k):
            raise NotImplementedError(f"{self.__name__}.{name} is not available in WebAssembly")
        return _unavailable

def _register(name, is_pkg=False):
    m = _Shim(name)
    if is_pkg:
        m.__path__ = []
    sys.modules.setdefault(name, m)

_register("ormsgpack")
for mod, pkg in [("websockets", True), ("websockets.sync", True), ("websockets.sync.client", False),
                 ("websockets.asyncio", True), ("websockets.asyncio.client", False),
                 ("websockets.exceptions", False), ("websockets.client", False), ("websockets.typing", False)]:
    _register(mod, pkg)

import agentgraph
from agentgraph.graph import run, build_graph, DEFAULT_MAX_STEPS
from agentgraph.tools import _KB, calculator, ToolError
import langgraph, importlib.metadata as md

LANGGRAPH_VERSION = md.version("langgraph")

def kb_json():
    return json.dumps(list(_KB.keys()))

def run_json(query):
    state = run(query)
    return json.dumps({
        "answer": state["answer"],
        "steps": state["steps"],
        "observations": state.get("observations", []),
        "step_count": state.get("step_count", 0),
    })

def calc_json(expr):
    try:
        return json.dumps({"ok": True, "result": calculator(expr)})
    except ToolError as e:
        return json.dumps({"ok": False, "error": str(e)})

# ── the composition: this agent's LLM calls, routed through llm-gateway ───────
#
# Both projects are running in this one tab, so the agent can actually call the
# gateway rather than us drawing an arrow between two boxes. A browser can't
# listen on a socket, so the transport hands each request straight to the
# gateway's ASGI app — the same object uvicorn would serve.
import anyio.to_thread
async def _run_sync_inline(func, *args, **kwargs):
    return func(*args)
anyio.to_thread.run_sync = _run_sync_inline   # WASM has no threads

from llmgateway.app import Config as GwConfig, create_app as create_gateway
from agentgraph.gateway import GatewayPolicy

GATEWAY = create_gateway(GwConfig(api_keys=frozenset({"dev-key"}), rate_capacity=40))

async def _asgi(method, path, body):
    """Await the gateway's real ASGI app in-process."""
    payload = json.dumps(body).encode() if body else b""
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"authorization", b"Bearer dev-key")],
        "client": ("127.0.0.1", 0), "server": ("localhost", 80),
    }
    sent = False
    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        return {"type": "http.disconnect"}
    msgs = []
    async def send(m):
        msgs.append(m)
    await GATEWAY(scope, receive, send)
    raw = b"".join(m.get("body", b"") for m in msgs if m["type"] == "http.response.body")
    return json.loads(raw)

# The policy builds the request; here we await it instead of calling it
# synchronously. Why: the graph's nodes are sync, and a sync node can't drive
# an async ASGI app inside an already-running event loop — normally Python
# would hand that to a worker thread, and WebAssembly has none. Against a real
# gateway over HTTP (the path the tests cover) the call happens inside the
# policy, in the node. Same request either way — build_request() is shared.
GW_POLICY = GatewayPolicy(transport=None, model="mock-1")

async def run_via_gateway_json(query):
    state = run(query)                                  # tools: real graph, deterministic
    obs = state.get("observations", [])
    if obs:
        body = GW_POLICY.build_request(query, obs)      # exactly what the policy would send
        resp = await _asgi("POST", "/v1/chat/completions", body)
        answer = (resp["choices"][0]["message"]["content"]
                  if "choices" in resp else state["answer"])
        cached, cost = bool(resp.get("cached")), float(resp.get("cost_usd", 0.0))
    else:
        answer, cached, cost, resp = state["answer"], False, 0.0, {}
    metrics = await _asgi("GET", "/metrics", None)
    return json.dumps({
        "answer": answer, "steps": state["steps"],
        "cached": cached, "cost_usd": cost, "gateway_metrics": metrics,
    })
`);

    const lgv = await py.runPythonAsync("LANGGRAPH_VERSION");
    setStatus(`Ready — LangGraph ${lgv} compiled and running in this tab`, "ready");
    document.querySelectorAll("button").forEach((b) => (b.disabled = false));
    renderKB(JSON.parse(await py.runPythonAsync("kb_json()")));
  } catch (err) {
    setStatus("Failed to boot: " + err, "err");
    console.error(err);
  }
}

function renderKB(keys) {
  $("kb").innerHTML =
    "<div style='color:var(--fg-dim);margin-bottom:8px'>The <code>search</code> tool knows about these topics — anything else returns no results:</div>" +
    keys.map((k) => `<span class="tok">${esc(k)}</span>`).join(" ");
}

async function ask(query) {
  const btn = $("ask");
  btn.disabled = true;
  $("trace").innerHTML = `<div class="thinking">running the graph…</div>`;
  try {
    const s = JSON.parse(await py.runPythonAsync(`run_json(${JSON.stringify(query)})`));
    const rows = s.steps.map((st) => {
      if (st.type === "action")
        return `<div class="step act"><span class="node">agent</span> decides → <code>${esc(st.tool)}(${esc(JSON.stringify(st.args))})</code><div class="why">${esc(st.reason || "")}</div></div>`;
      if (st.type === "observation")
        return `<div class="step obs"><span class="node">tools</span> returns → <b>${esc(st.result)}</b>${st.error ? ' <span class="bad">(tool error)</span>' : ""}</div>`;
      if (st.type === "final")
        return `<div class="step fin"><span class="node">agent</span> finishes</div>`;
      if (st.type === "finish")
        return `<div class="step guard"><span class="node">guard</span> ${esc(st.reason)}</div>`;
      return "";
    }).join("");
    $("trace").innerHTML =
      `<div class="tracebox">
         <div class="tracehead">START → agent ⇄ tools → END &nbsp;·&nbsp; ${s.steps.filter(x=>x.type==="action").length} tool call(s) · ${s.step_count} agent turns</div>
         ${rows}
         <div class="answer"><span class="node ans">answer</span> ${esc(s.answer)}</div>
       </div>`;
  } catch (err) {
    $("trace").innerHTML = `<div class="tracebox"><div class="bad">Error: ${esc(err)}</div></div>`;
  }
  btn.disabled = false;
}

async function tryCalc() {
  const expr = $("expr").value.trim();
  const r = JSON.parse(await py.runPythonAsync(`calc_json(${JSON.stringify(expr)})`));
  $("calcOut").innerHTML = r.ok
    ? `<div class="verdict ok"><b>${esc(expr)}</b> = <b>${esc(r.result)}</b><div class="why">Evaluated by walking a parsed AST — only arithmetic nodes are permitted.</div></div>`
    : `<div class="verdict bad"><b>Rejected:</b> ${esc(r.error)}<div class="why">The tool refused it. There is no <code>eval()</code> here — names, calls and attribute access simply aren't reachable nodes.</div></div>`;
}

async function askViaGateway() {
  const btn = $("askGw");
  btn.disabled = true;
  const q = $("gwq").value.trim();
  $("gwOut").innerHTML = `<div class="thinking">agent thinking · its LLM call routing through the gateway…</div>`;
  try {
    const s = JSON.parse(await py.runPythonAsync(`await run_via_gateway_json(${JSON.stringify(q)})`));
    const m = s.gateway_metrics;
    const tools = s.steps.filter((x) => x.type === "action").map((x) => x.tool);
    $("gwOut").innerHTML = `
      <div class="tracebox">
        <div class="tracehead">agent-graph ──► llm-gateway ──► mock provider</div>
        <div class="step act"><span class="node">agent</span> ran tools: <code>${tools.join(" → ") || "none"}</code></div>
        <div class="step obs"><span class="node">gateway</span> handled the compose call ·
          <b class="${s.cached ? "teal" : ""}">cached=${s.cached}</b> · cost $${s.cost_usd}</div>
        <div class="step fin"><span class="node">provider</span> replied: <span style="color:var(--fg-dim)">${esc(String(s.answer).slice(0, 150))}${String(s.answer).length > 150 ? "…" : ""}</span>
          <div class="why">The offline mock echoes its prompt rather than synthesizing — that's what a deterministic test double does, and it's why this runs with no key. With a real key the same call returns a real answer; the gateway's behavior around it is identical.</div></div>
      </div>
      <div class="cards" style="margin-top:12px">
        <div class="card"><div class="k">Gateway requests</div><div class="v">${m.requests}</div></div>
        <div class="card"><div class="k">Cache hits</div><div class="v teal">${m.cache_hits}</div></div>
        <div class="card"><div class="k">Hit rate</div><div class="v teal">${(m.cache_hit_rate * 100).toFixed(0)}%</div></div>
        <div class="card"><div class="k">Tokens</div><div class="v">${m.total_tokens}</div></div>
      </div>
      ${m.cache_hits > 0 ? `<div class="note" style="margin-top:12px">Ask the same question again and the gateway serves the agent's LLM call from cache — the provider is never touched. That's the whole point of putting one in front of an agent.</div>` : ""}`;
  } catch (err) {
    $("gwOut").innerHTML = `<div class="verdict bad">Error: ${esc(err)}</div>`;
  }
  btn.disabled = false;
}

$("askGw").addEventListener("click", askViaGateway);
document.querySelectorAll("[data-gwq]").forEach((b) =>
  b.addEventListener("click", () => { $("gwq").value = b.dataset.gwq; askViaGateway(); })
);
$("ask").addEventListener("click", () => ask($("q").value.trim()));
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") ask($("q").value.trim()); });
document.querySelectorAll("[data-q]").forEach((b) =>
  b.addEventListener("click", () => { $("q").value = b.dataset.q; ask(b.dataset.q); })
);
$("runCalc").addEventListener("click", tryCalc);
document.querySelectorAll("[data-expr]").forEach((b) =>
  b.addEventListener("click", () => { $("expr").value = b.dataset.expr; tryCalc(); })
);
boot();

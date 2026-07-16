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

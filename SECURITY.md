# Security policy

This is a ReAct agent loop with deterministic tools. The security-relevant part
is not the model, it is the loop: what the tools will do with text the model
produces, and whether the loop can be made not to stop.

## Threat model

Assume the text reaching a tool is attacker-controlled, because the model
writes it from a prompt the attacker influenced. The model is not a filter and
a system prompt is not a boundary. Assume the attacker can also steer the
agent's trajectory: which tool, how many times, with what argument.

## In scope

- **Code execution or import through the calculator.** It parses to an AST and
  allow-lists arithmetic nodes. Reaching an attribute access, a name lookup, a
  call, or a comprehension is a finding.
- **Cost exhaustion through an allow-listed expression.** The exponent is
  bounded by result width; an arithmetic expression that still hangs or
  allocates without limit is a finding.
- **Escaping the step guard.** The loop has a maximum step count. Any
  trajectory that runs unbounded, or that resets the counter, is a finding,
  including one built from tool outputs that re-enter the loop.
- **A tool that is not deterministic when the README says it is.** A tool whose
  output depends on the host, the clock, or the environment breaks the property
  the test suite relies on, and a test that cannot observe that is part of the
  finding.
- **Disclosure through a tool result or an error.** Environment values,
  filesystem paths, or URL query strings reaching the transcript. A masked
  value that reappears because the error string was truncated counts.
- **Path escape** through any tool argument.

## Out of scope

- **The model making a poor decision.** Choosing a silly tool or getting the
  arithmetic wrong is a capability question. The mock policy exists so the loop
  can be tested without one.
- **Prompt injection that only changes what the agent says.** In scope only
  when it reaches a tool and crosses one of the boundaries above. An agent
  persuaded to be rude is not a vulnerability here; an agent persuaded to make
  a tool read a file outside the corpus is.
- **The host application's authorization decisions** about which tools to
  register.

## Reporting

- **GitHub private advisory**, preferred:
  <https://github.com/egnaro9/agent-graph/security/advisories/new>
- **Email**: erik@erikhill.dev

Send the trajectory, or the tool call on its own if that is enough.
Acknowledgement within 3 days, an answer within 14, credit in the fix commit
unless you would rather not. Accepted findings get a test confirmed to fail
against the pre-fix code.

## Known and fixed

- **Unbounded `ast.Pow`**, fixed 2026-09-28. The allow-list rejected injection
  but bounded no cost, so an allow-listed expression could hang the tool and
  with it the graph. The bound now estimates the result's decimal width and
  refuses above 1,000 digits. The same mapping in
  [mcp-tools](https://github.com/egnaro9/mcp-tools) was fixed in the same pass.

## Supported versions

`main`.

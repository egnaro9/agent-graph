"""The planner must not know any tool's internals.

This check lives outside test_rag.py on purpose. That module skips as a whole
when ragevallab is missing, and CI installs only the dev extra, so a check
placed there never ran in CI. This one reads policy.py and nothing else, so it
needs no optional dependency and runs on every install.
"""
import pathlib


def test_planner_does_not_import_tool_internals():
    # The regression that started this: the policy used to reach into the
    # search tool's private _KB, which is why a real corpus never triggered.
    import agentgraph.policy as policy_mod
    src = pathlib.Path(policy_mod.__file__).read_text()
    # Prove the file read is the planner itself. If policy.py became a
    # re-export shim, the two checks below would pass while scanning nothing.
    assert "class MockPolicy" in src
    assert "_KB" not in src
    assert "always_search" not in src

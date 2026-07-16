import pytest

from agentgraph.tools import ToolError, calculator, run_tool, search, wordcount


def test_calculator_basic_and_precedence():
    assert calculator("2 + 3") == "5"
    assert calculator("2 + 3 * 4") == "14"
    assert calculator("(2 + 3) * 4") == "20"
    assert calculator("15/100*240") == "36"      # float that is integral -> int
    assert calculator("2 ** 5") == "32"
    assert calculator("-7 + 2") == "-5"


def test_calculator_float_result():
    assert calculator("1/4") == "0.25"


@pytest.mark.parametrize("bad", [
    "__import__('os')",
    "os.system('ls')",
    "1 + foo",
    "open('x')",
    "1 +",           # syntax error
])
def test_calculator_rejects_unsafe_or_invalid(bad):
    with pytest.raises(ToolError):
        calculator(bad)


def test_search_hits_and_misses():
    assert "Shakespeare" in search("who wrote hamlet")
    assert "Paris" in search("what is the capital of france")
    assert "No results" in search("what is the airspeed of a swallow")


def test_wordcount():
    assert wordcount("one two three") == "3"


def test_run_tool_unknown():
    with pytest.raises(ToolError):
        run_tool("nope", "x")

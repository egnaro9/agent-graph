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


@pytest.mark.parametrize("expr", ["4/0", "0%0", "1//0"])
def test_calculator_reports_undefined_arithmetic_as_a_tool_error(expr):
    """Well-formed but undefined arithmetic must not escape as ArithmeticError.

    The graph's tools node only catches ToolError, so a bare ZeroDivisionError
    propagated out and killed the whole run on a query a user can type by
    accident. This asserts it arrives as a tool error like any other bad input.
    """
    with pytest.raises(ToolError):
        calculator(expr)


def test_divide_by_zero_is_recorded_not_raised():
    """The end-to-end path: the run completes and reports the error."""
    from agentgraph.graph import run

    out = run("what is 4/0")
    assert "division by zero" in str(out).lower()


def test_search_hits_and_misses():
    assert "Shakespeare" in search("who wrote hamlet")
    assert "Paris" in search("what is the capital of france")
    assert "No results" in search("what is the airspeed of a swallow")


def test_wordcount():
    assert wordcount("one two three") == "3"


def test_run_tool_unknown():
    with pytest.raises(ToolError):
        run_tool("nope", "x")


def test_calculator_refuses_an_expression_too_wide_to_compute():
    """`9 ** 9 ** 9` is pure arithmetic, so the allow-list passes it happily.

    Its result is about 369 million digits, so the tool used to hang rather than
    answer, and a hung tool stalls the entire graph: no observation, no error,
    no completion. The timing assertion is the point. Raising is easy; refusing
    before starting the multiplication is the property under test.
    """
    import time

    start = time.monotonic()
    with pytest.raises(ToolError, match="digits"):
        calculator("9**9**9")
    assert time.monotonic() - start < 1.0


def test_calculator_still_does_ordinary_exponentiation():
    assert calculator("2**10") == "1024"
    assert calculator("(-2)**3") == "-8"
    assert calculator("2**0.5").startswith("1.414")


# ---------------------------------------------------------------------------
# The guard's PREDICATE, not just one input it happens to refuse.
#
# A mutation audit reduced `exponent * math.log10(abs(base)) > _MAX_POW_DIGITS` to a bare
# `exponent > _MAX_POW_DIGITS` and all 42 tests stayed green. 9**9**9 is refused under both,
# because its outer exponent is 387,420,489 on its own, so the one input the suite tested
# could not tell the predicates apart.

@pytest.mark.parametrize("expr,digits", [
    ("(10**400)**500", 200_000),
    ("(10**1000)**1000", 1_000_000),
])
def test_a_wide_result_is_refused_even_when_the_exponent_is_small(expr, digits):
    """These have exponents of 500 and 1000, inside any exponent-only limit, and results of
    2e5 and 1e6 digits, far outside the 1000-digit one. The guard is about the WIDTH of the
    result, so testing only a large exponent leaves the class it exists to close wide open.

    Under the reduced predicate this does not merely compute: it dies with an uncaught
    ValueError from `str(value)`, which sits outside the ArithmeticError try-block.
    """
    with pytest.raises(ToolError, match="digits"):
        calculator(expr)


def test_a_narrow_result_with_a_large_base_is_still_allowed():
    """The other side, so the guard cannot be satisfied by refusing everything."""
    assert calculator("(10**4)**2") == "100000000"


def test_the_historic_hang_is_still_refused():
    """9**9**9 is ~369 million digits and CPython computes it eagerly. This is the input the
    guard was written for; it is kept, but it is no longer the only one."""
    with pytest.raises(ToolError):
        calculator("9 ** 9 ** 9")


# ---------------------------------------------------------------------------
# looks_like_a_question's question-word branch was unreachable by the suite: every positive
# query in the RAG tests ends in '?', so `q.startswith(_QUESTION_WORDS)` could be deleted and
# nothing noticed. The retriever then stops firing on unpunctuated questions entirely and the
# agent answers "I don't have a tool that helps with that."

@pytest.mark.parametrize("q", [
    "what is the capital of france",
    "which planet is hottest",
    "who wrote hamlet",
    "is water wet",
    "are these covered",
    "does it apply",
    "did it run",
    "how many moons",
])
def test_an_unpunctuated_question_is_still_a_question(q):
    from agentgraph.tools import looks_like_a_question
    assert looks_like_a_question(q) is True


@pytest.mark.parametrize("q", ["the capital of france", "calculate 2 + 2", "hamlet"])
def test_a_statement_is_not_a_question(q):
    from agentgraph.tools import looks_like_a_question
    assert looks_like_a_question(q) is False


# ---------------------------------------------------------------------------
# agentgraph.cli had no test module at all, and the CI smoke step that exercised it masked
# its exit code: `python -m agentgraph.cli run "..." | tee run.out` under GitHub's default
# `bash -e {0}` takes the pipeline's status from tee, so a CLI exiting 1 left the step GREEN.
# That is the same defect class this author's own deadgate reports as D3.

def test_the_cli_reports_success_as_zero_and_is_importable():
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "agentgraph.cli", "run", "What is 12 * 8?"],
                       capture_output=True, text=True, cwd=str(__import__("pathlib")
                       .Path(__file__).resolve().parents[1]))
    assert r.returncode == 0, f"exit {r.returncode}: {r.stderr[:300]}"
    assert "96" in r.stdout


def test_the_cli_exit_code_survives_a_pipe():
    """The CI step pipes through tee. Without pipefail the pipeline reports tee's status, so
    this asserts the shape CI actually runs rather than the bare command."""
    import subprocess
    import sys
    root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    cmd = (f'set -o pipefail; {sys.executable} -m agentgraph.cli run "What is 12 * 8?" '
           '| tee /dev/null > /dev/null')
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, f"exit {r.returncode}: {r.stderr[:300]}"

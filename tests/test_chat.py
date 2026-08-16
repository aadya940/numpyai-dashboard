"""Tests for chat, with the model stubbed out. No LLM, no API key."""

from __future__ import annotations

import numpy as np
import pytest

from numpyai_dashboard import ChatResult, array
from numpyai_dashboard._ai import CodeResponse, Judgment


class FakeCodeGen:
    """Stands in for NumpyCodeGen, replaying a scripted sequence of attempts.

    Each entry is ``(code, explanation, accepted, reason)``. The last entry is
    reused if the caller retries more often than the script anticipated.
    """

    def __init__(self, script):
        self.script = script
        self.calls = 0
        self.last_prompt = ""

    @staticmethod
    def prompt_single(**kwargs):
        return "prompt"

    def generate_code(self, prompt):
        self.last_prompt = prompt
        code, explanation, _, _ = self.script[min(self.calls, len(self.script) - 1)]
        return CodeResponse(code=code, explanation=explanation)

    def judge(self, query, code, metadata, context=""):
        self.last_context = context
        _, _, accepted, reason = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return Judgment(interprets_query_correctly=accepted, reason=reason)


def make(script, **kwargs):
    arr = array(np.array([[1.0, 2.0], [3.0, 4.0]]), **kwargs)
    arr._code_generator = FakeCodeGen(script)
    return arr


PASSES = [("output = arr.sum()\nmetadata = 'sum of arr'", "sums", True, "")]
REJECTED = [("output = arr.min()\nmetadata = 'min'", "min", False, "asked for max")]


# --------------------------------------------------------------------------
# chat returns the whole record
# --------------------------------------------------------------------------


def test_chat_returns_a_chatresult():
    assert isinstance(make(PASSES).chat("total"), ChatResult)


def test_chat_carries_value_code_and_description():
    result = make(PASSES).chat("total")
    assert result.value == 10.0
    assert result.code == "output = arr.sum()\nmetadata = 'sum of arr'"
    assert result.description == "sum of arr"
    assert result.attempts == 1
    assert result.ok is True


def test_chat_carries_the_judgment():
    result = make(PASSES).chat("total")
    assert result.judgment.interprets_query_correctly is True


def test_chat_reports_failure_without_warning(recwarn):
    result = make(REJECTED, max_tries=2).chat("largest")
    assert result.ok is False
    assert result.value is None
    assert result.attempts == 2
    assert len(result.errors) == 2
    assert "asked for max" in result.errors[0]
    assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []


def test_chat_counts_attempts_until_accepted():
    script = REJECTED + PASSES
    result = make(script, max_tries=3).chat("total")
    assert result.ok is True
    assert result.attempts == 2
    assert len(result.errors) == 1


# --------------------------------------------------------------------------
# display and misuse
# --------------------------------------------------------------------------


def test_repr_leads_with_the_answer():
    assert repr(make(PASSES).chat("total")) == "ChatResult(np.float64(10.0))"


def test_repr_shows_attempts_when_more_than_one():
    assert "attempts=2" in repr(make(REJECTED + PASSES, max_tries=3).chat("total"))


def test_repr_explains_failure():
    text = repr(make(REJECTED, max_tries=2).chat("largest"))
    assert "failed after 2 attempts" in text
    assert "asked for max" in text


def test_chat_rejects_a_non_string_query():
    with pytest.raises(TypeError, match="query must be a string"):
        make(PASSES).chat(42)


def test_execution_errors_reach_the_retry_feedback():
    """The model must see the real exception, not 'returned None'."""
    broken = [("output = df.resample('M')\nmetadata = 'x'", "resamples", True, "")]
    result = make(broken, max_tries=1).chat("monthly")
    assert result.ok is False
    assert any("df" in e and "not defined" in e for e in result.errors)


def test_stray_figure_is_adopted_when_output_is_missing():
    plt = pytest.importorskip("matplotlib.pyplot")
    from matplotlib.figure import Figure

    from numpyai_dashboard._engine import execute

    code = "plt.figure()\nplt.plot([1, 2, 3])\nmetadata = 'a plot'"
    value, _ = execute(code, {}, verbose=False)
    assert isinstance(value, Figure)
    assert plt.get_fignums() == []


def test_stray_figure_is_closed_when_output_is_set():
    """Side-effect plots must not leak in pyplot's global registry."""
    plt = pytest.importorskip("matplotlib.pyplot")

    from numpyai_dashboard._engine import execute

    code = "plt.figure()\nplt.plot([1, 2])\noutput = 42\nmetadata = 'x'"
    value, _ = execute(code, {}, verbose=False)
    assert value == 42
    assert plt.get_fignums() == []


def test_pandas_is_in_the_execution_namespace():
    """Multi-file chat promised pd and the namespace lacked it; never again."""
    pytest.importorskip("pandas")
    from numpyai_dashboard._engine import execute

    value, _ = execute(
        "output = pd.DataFrame({'a': [1, 2]})['a'].sum()\nmetadata = 'x'",
        {},
        verbose=False,
    )
    assert value == 3

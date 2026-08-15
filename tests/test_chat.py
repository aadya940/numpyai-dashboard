"""Tests for chat/ask, with the model stubbed out. No LLM, no API key."""

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

    @staticmethod
    def prompt_single(**kwargs):
        return "prompt"

    def generate_code(self, prompt):
        code, explanation, _, _ = self.script[min(self.calls, len(self.script) - 1)]
        return CodeResponse(code=code, explanation=explanation)

    def judge(self, query, code, metadata):
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
# ask returns the whole record
# --------------------------------------------------------------------------


def test_ask_returns_a_chatresult():
    assert isinstance(make(PASSES).ask("total"), ChatResult)


def test_ask_carries_value_code_and_description():
    result = make(PASSES).ask("total")
    assert result.value == 10.0
    assert result.code == "output = arr.sum()\nmetadata = 'sum of arr'"
    assert result.description == "sum of arr"
    assert result.attempts == 1
    assert result.ok is True


def test_ask_carries_the_judgment():
    result = make(PASSES).ask("total")
    assert result.judgment.interprets_query_correctly is True


def test_ask_reports_failure_without_warning(recwarn):
    result = make(REJECTED, max_tries=2).ask("largest")
    assert result.ok is False
    assert result.value is None
    assert result.attempts == 2
    assert len(result.errors) == 2
    assert "asked for max" in result.errors[0]
    assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []


def test_ask_counts_attempts_until_accepted():
    script = REJECTED + PASSES
    result = make(script, max_tries=3).ask("total")
    assert result.ok is True
    assert result.attempts == 2
    assert len(result.errors) == 1


# --------------------------------------------------------------------------
# chat keeps its old contract exactly
# --------------------------------------------------------------------------


def test_chat_returns_the_bare_value():
    assert make(PASSES).chat("total") == 10.0


def test_chat_returns_none_and_warns_on_failure():
    arr = make(REJECTED, max_tries=2)
    with pytest.warns(UserWarning, match="Validation failed after 2 attempts"):
        assert arr.chat("largest") is None


def test_chat_rejects_a_non_string_query():
    with pytest.raises(TypeError, match="query must be a string"):
        make(PASSES).chat(42)


def test_ask_rejects_a_non_string_query():
    with pytest.raises(TypeError, match="query must be a string"):
        make(PASSES).ask(42)

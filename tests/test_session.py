"""Tests for NumpyAISession, with the model stubbed out. No LLM, no API key."""

from __future__ import annotations

import numpy as np
import pytest

from numpyai_dashboard import ChatResult, NumpyAISession, array, frame
from numpyai_dashboard._ai import prompt_multiple

pd = pytest.importorskip("pandas")

from test_chat import FakeCodeGen  # noqa: E402

A = np.array([[1.0, 2.0], [3.0, np.nan]])
B = np.array([[10.0, 20.0], [30.0, 40.0]])


@pytest.fixture
def sales():
    return pd.DataFrame({"region": ["EMEA", "APAC"], "units": [3.0, 5.0]})


def make(items, script, **kwargs):
    sess = NumpyAISession(items, **kwargs)
    sess._code_generator = FakeCodeGen(script)
    return sess


# --------------------------------------------------------------------------
# what the session accepts
# --------------------------------------------------------------------------


def test_accepts_bare_arrays():
    assert list(NumpyAISession([A, B]).context) == ["arr1", "arr2"]


def test_accepts_wrapped_arrays():
    assert list(NumpyAISession([array(A), array(B)]).context) == ["arr1", "arr2"]


def test_accepts_dataframes(sales):
    assert list(NumpyAISession([sales]).context) == ["df1"]


def test_accepts_wrapped_frames(sales):
    assert list(NumpyAISession([frame(sales)]).context) == ["df1"]


def test_mixed_inputs_keep_positional_numbering(sales):
    """`df2` is the second input, so 'the second one' stays unambiguous."""
    assert list(NumpyAISession([A, sales, B]).context) == ["arr1", "df2", "arr3"]


def test_rejects_unsupported_input():
    with pytest.raises(TypeError, match="at index 1"):
        NumpyAISession([A, "not data"])


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


def test_chat_sees_every_array():
    script = [
        (
            "output = float(np.nanmean(arr1) + np.nanmean(arr2))\nmetadata = 'sum'",
            "adds",
            True,
            "",
        )
    ]
    result = make([A, B], script).chat("combine them")
    assert isinstance(result, ChatResult)
    assert result.value == pytest.approx(2.0 + 25.0)


def test_chat_sees_a_frame_and_an_array(sales):
    script = [
        (
            "output = float(df2['units'].sum() + np.nansum(arr1))\nmetadata = 'mix'",
            "mixes",
            True,
            "",
        )
    ]
    assert make([A, sales], script).chat("combine").value == pytest.approx(8.0 + 6.0)


def test_pandas_is_bound_only_when_a_frame_is_present(sales):
    script = [("output = pd.Series([1, 2]).sum()\nmetadata = 'pd'", "pd", True, "")]
    assert make([sales], script).chat("check").value == 3

    # No frame in the session, so `pd` is not in scope and the attempt fails.
    result = make([A], script, max_tries=1).chat("check")
    assert result.ok is False


def test_failure_is_reported_not_raised():
    script = [("output = np.nanmean(arr1)\nmetadata = 'mean'", "mean", False, "nope")]
    result = make([A, B], script, max_tries=2).chat("something else")
    assert result.ok is False
    assert result.attempts == 2


# --------------------------------------------------------------------------
# metadata and prompt
# --------------------------------------------------------------------------


def test_context_follows_a_mutated_frame(sales):
    sess = NumpyAISession([sales])
    assert sess.context["df1"]["metadata"]["rows"] == 2
    sess._context["df1"]["data"] = sales[sales["region"] == "EMEA"]
    assert sess.context["df1"]["metadata"]["rows"] == 1


def test_prompt_describes_arrays_and_frames(sales):
    prompt = prompt_multiple("compare", NumpyAISession([A, sales]).context)
    assert "`arr1` is a NumPy array" in prompt
    assert "`df2` is a pandas DataFrame" in prompt
    assert "df2['region']" in prompt
    assert "'EMEA'" in prompt


def test_prompt_only_mentions_pandas_when_relevant(sales):
    assert "`pd` (pandas)" in prompt_multiple("x", NumpyAISession([sales]).context)
    assert "`pd` (pandas)" not in prompt_multiple("x", NumpyAISession([A]).context)


def test_prompt_carries_the_rejection_reason():
    prompt = prompt_multiple("x", NumpyAISession([A]).context, prior_feedback="bad")
    assert "bad" in prompt

"""Tests for the array wrapper's column-name support. No LLM, no API key."""

from __future__ import annotations

import numpy as np
import pytest

from numpyai_dashboard import array
from numpyai_dashboard._ai import NumpyCodeGen


@pytest.fixture
def data():
    return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_columns_default_to_none(data):
    assert array(data).columns is None


def test_columns_are_exposed(data):
    assert array(data, columns=["a", "b", "c"]).columns == ["a", "b", "c"]


def test_columns_reach_the_metadata(data):
    arr = array(data, columns=["a", "b", "c"])
    assert arr.metadata["columns"] == ["a", "b", "c"]


def test_columns_render_in_the_prompt(data):
    arr = array(data, columns=["units", "price", "discount"])
    prompt = NumpyCodeGen.prompt_single("total revenue", arr.metadata)
    assert "arr[:, 0] -> units" in prompt
    assert "arr[:, 2] -> discount" in prompt


def test_no_column_block_without_names(data):
    prompt = NumpyCodeGen.prompt_single("mean", array(data).metadata)
    assert "whose columns are, in order" not in prompt


def test_wrong_number_of_names_raises(data):
    with pytest.raises(ValueError, match="2 column names for 3 columns"):
        array(data, columns=["a", "b"])


def test_columns_require_2d():
    with pytest.raises(ValueError, match="2-D"):
        array(np.array([1.0, 2.0, 3.0]), columns=["a"])


def test_columns_dropped_when_width_changes(data):
    arr = array(data, columns=["a", "b", "c"])
    arr.data = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert arr.columns is None
    assert "columns" not in arr.metadata


def test_columns_survive_a_same_width_replacement(data):
    arr = array(data, columns=["a", "b", "c"])
    arr.data = np.zeros((5, 3))
    assert arr.columns == ["a", "b", "c"]


def test_bridging_from_a_dataframe():
    """The documented DataFrame -> numpy bridge."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"units": [1.0, 2.0], "price": [10.0, 20.0]})
    cols = frame.columns.tolist()
    arr = array(frame[cols].to_numpy(), columns=cols)
    assert arr.columns == ["units", "price"]
    assert arr.data.shape == (2, 2)

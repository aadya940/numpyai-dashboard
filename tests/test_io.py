"""Tests for the spreadsheet loaders. No LLM and no API key required."""

from __future__ import annotations

import numpy as np
import pytest

from numpyai_dashboard import read_excel
from numpyai_dashboard._exceptions import NumpyAIError

from _xlsx import write_xlsx

pytest.importorskip("python_calamine", reason="needs numpyai-dashboard[excel]")


@pytest.fixture
def book(tmp_path):
    """Write rows to a temp .xlsx and return the path."""

    def _write(rows, name="book.xlsx"):
        path = tmp_path / name
        write_xlsx(path, rows)
        return str(path)

    return _write


def test_reads_numeric_columns_with_header(book):
    path = book([["a", "b"], [1, 2.5], [3, 4.5]])
    arr = read_excel(path)
    assert arr.columns == ["a", "b"]
    assert arr.data.dtype == np.float64
    np.testing.assert_array_equal(arr.data, [[1.0, 2.5], [3.0, 4.5]])


def test_blank_cells_become_nan(book):
    path = book([["a", "b"], [1, None], [None, 4.5]])
    arr = read_excel(path)
    assert np.isnan(arr.data[0, 1])
    assert np.isnan(arr.data[1, 0])


def test_booleans_become_floats(book):
    path = book([["flag"], [True], [False]])
    arr = read_excel(path)
    np.testing.assert_array_equal(arr.data, [[1.0], [0.0]])


def test_text_and_date_columns_are_dropped_with_warning(book):
    path = book(
        [
            ["id", "name", "when"],
            [1, "alpha", ("date", 45000)],
            [2, "beta", ("date", 45001)],
        ]
    )
    with pytest.warns(UserWarning, match="non-numeric column"):
        arr = read_excel(path)
    assert arr.columns == ["id"]
    assert arr.data.shape == (2, 1)


def test_header_false_generates_positional_names(book):
    path = book([[1, 2], [3, 4]])
    arr = read_excel(path, header=False)
    assert arr.columns == ["col0", "col1"]
    assert arr.data.shape == (2, 2)


def test_blank_header_cell_gets_positional_name(book):
    path = book([["a", None], [1, 2]])
    arr = read_excel(path)
    assert arr.columns == ["a", "col1"]


def test_numeric_strings_are_accepted(book):
    path = book([["a"], ["1.5"], ["2.5"]])
    arr = read_excel(path)
    np.testing.assert_array_equal(arr.data, [[1.5], [2.5]])


def test_columns_reach_the_metadata(book):
    path = book([["a", "b"], [1, 2]])
    arr = read_excel(path)
    assert arr.metadata["columns"] == ["a", "b"]


def test_no_numeric_columns_raises(book):
    path = book([["name"], ["alpha"], ["beta"]])
    with pytest.warns(UserWarning), pytest.raises(NumpyAIError, match="no numeric columns"):
        read_excel(path)


def test_header_only_sheet_raises(book):
    path = book([["a", "b"]])
    with pytest.raises(NumpyAIError, match="no data rows"):
        read_excel(path)


def test_sheet_selected_by_name(book):
    path = book([["a"], [1]])
    arr = read_excel(path, sheet="Sheet1")
    np.testing.assert_array_equal(arr.data, [[1.0]])


def test_csv_gets_a_pointed_error(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(NumpyAIError, match="CSV is not supported"):
        read_excel(str(path))


def test_unsupported_extension_lists_supported_ones(tmp_path):
    path = tmp_path / "data.parquet"
    path.write_bytes(b"")
    with pytest.raises(NumpyAIError, match=r"\.xlsx"):
        read_excel(str(path))


def test_unknown_sheet_name_raises(book):
    path = book([["a"], [1]])
    with pytest.raises(Exception):
        read_excel(path, sheet="NoSuchSheet")

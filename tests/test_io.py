"""Tests for the spreadsheet loaders. No LLM and no API key required."""

from __future__ import annotations

import numpy as np
import pytest
from _xlsx import write_xlsx

from numpyai_dashboard import array, read_excel
from numpyai_dashboard._exceptions import NumpyAIError

pytest.importorskip("python_calamine", reason="needs numpyai-dashboard[excel]")


@pytest.fixture
def book(tmp_path):
    """Write rows to a temp .xlsx and return the path."""

    def _write(rows, name="book.xlsx"):
        path = tmp_path / name
        write_xlsx(path, rows)
        return str(path)

    return _write


# --------------------------------------------------------------------------
# all-numeric sheets stay a plain 2-D float64 array
# --------------------------------------------------------------------------


def test_all_numeric_sheet_is_a_plain_2d_array(book):
    path = book([["a", "b"], [1, 2.5], [3, 4.5]])
    arr = read_excel(path)
    assert arr.is_table is False
    assert arr.columns == ["a", "b"]
    assert arr.data.dtype == np.float64
    np.testing.assert_array_equal(arr.data, [[1.0, 2.5], [3.0, 4.5]])


def test_blank_numeric_cells_become_nan(book):
    path = book([["a", "b"], [1, None], [None, 4.5]])
    arr = read_excel(path)
    assert np.isnan(arr.data[0, 1])
    assert np.isnan(arr.data[1, 0])


def test_booleans_become_floats(book):
    path = book([["flag"], [True], [False]])
    arr = read_excel(path)
    np.testing.assert_array_equal(arr.data, [[1.0], [0.0]])


def test_numeric_strings_are_accepted(book):
    path = book([["a"], ["1.5"], ["2.5"]])
    arr = read_excel(path)
    np.testing.assert_array_equal(arr.data, [[1.5], [2.5]])


# --------------------------------------------------------------------------
# mixed sheets keep every column, as a structured array
# --------------------------------------------------------------------------


def test_mixed_sheet_keeps_every_column(book):
    path = book(
        [
            ["id", "name", "when"],
            [1, "alpha", ("date", 45000)],
            [2, "beta", ("date", 45001)],
        ]
    )
    arr = read_excel(path)
    assert arr.is_table is True
    assert arr.columns == ["id", "name", "when"]
    np.testing.assert_array_equal(arr.data["id"], [1.0, 2.0])
    np.testing.assert_array_equal(arr.data["name"], ["alpha", "beta"])
    assert arr.data["when"][0] == np.datetime64("2023-03-15T00:00:00")


def test_mixed_sheet_emits_no_warning(book, recwarn):
    path = book([["id", "name"], [1, "alpha"]])
    read_excel(path)
    assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []


def test_text_column_dtype_and_grouping(book):
    path = book(
        [
            ["region", "score"],
            ["EMEA", 3.0],
            ["APAC", 5.0],
            ["EMEA", 7.0],
        ]
    )
    arr = read_excel(path)
    assert arr.data["region"].dtype.kind == "U"
    emea = arr.data["score"][arr.data["region"] == "EMEA"]
    np.testing.assert_array_equal(emea, [3.0, 7.0])


def test_dates_become_datetime64(book):
    path = book([["when", "label"], [("date", 45000), "x"]])
    arr = read_excel(path)
    assert arr.data["when"].dtype == np.dtype("datetime64[s]")


def test_blank_date_becomes_nat(book):
    path = book([["when", "label"], [("date", 45000), "x"], [None, "y"]])
    arr = read_excel(path)
    assert np.isnat(arr.data["when"][1])


def test_blank_text_becomes_empty_string(book):
    path = book([["name", "n"], ["alpha", 1], [None, 2]])
    arr = read_excel(path)
    assert arr.data["name"][1] == ""


def test_text_only_sheet_loads(book):
    path = book([["name"], ["alpha"], ["beta"]])
    arr = read_excel(path)
    assert arr.is_table is True
    np.testing.assert_array_equal(arr.data["name"], ["alpha", "beta"])


def test_duplicate_headers_are_disambiguated(book):
    path = book([["score", "score"], [1, "a"]])
    arr = read_excel(path)
    assert arr.columns == ["score", "score_1"]


# --------------------------------------------------------------------------
# headers, metadata, errors
# --------------------------------------------------------------------------


def test_header_false_generates_positional_names(book):
    path = book([[1, 2], [3, 4]])
    arr = read_excel(path, header=False)
    assert arr.columns == ["col0", "col1"]
    assert arr.data.shape == (2, 2)


def test_blank_header_cell_gets_positional_name(book):
    path = book([["a", None], [1, 2]])
    arr = read_excel(path)
    assert arr.columns == ["a", "col1"]


def test_metadata_summarises_each_column(book):
    path = book([["region", "score"], ["EMEA", 3.0], ["APAC", 5.0]])
    arr = read_excel(path)
    summary = arr.metadata["column_summary"]
    assert arr.metadata["is_table"] is True
    assert summary["region"]["kind"] == "text"
    assert summary["region"]["categories"] == ["APAC", "EMEA"]
    assert summary["score"]["kind"] == "numeric"
    assert summary["score"]["min"] == 3.0


def test_columns_rejected_for_structured_array():
    data = np.array([(1.0, "a")], dtype=[("x", "f8"), ("y", "U1")])
    with pytest.raises(ValueError, match="structured array"):
        array(data, columns=["x", "y"])


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
    from python_calamine import WorksheetNotFound

    path = book([["a"], [1]])
    with pytest.raises(WorksheetNotFound):
        read_excel(path, sheet="NoSuchSheet")

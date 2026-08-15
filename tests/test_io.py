"""Tests for the file loaders. No LLM and no API key required."""

from __future__ import annotations

import numpy as np
import pytest
from _xlsx import write_xlsx

from numpyai_dashboard import read_csv, read_excel
from numpyai_dashboard._exceptions import NumpyAIError

pytest.importorskip("fastexcel", reason="needs numpyai-dashboard[excel]")
pd = pytest.importorskip("pandas", reason="needs numpyai-dashboard[excel]")


@pytest.fixture
def book(tmp_path):
    """Write rows to a temp .xlsx and return the path."""

    def _write(rows, name="book.xlsx"):
        path = tmp_path / name
        write_xlsx(path, rows)
        return str(path)

    return _write


# --------------------------------------------------------------------------
# every column survives, with a sensible dtype
# --------------------------------------------------------------------------


def test_returns_a_dataframe(book):
    path = book([["a", "b"], [1, 2.5]])
    frame = read_excel(path)
    assert isinstance(frame, pd.DataFrame)
    assert frame.columns.tolist() == ["a", "b"]


def test_mixed_sheet_keeps_every_column(book):
    path = book(
        [
            ["id", "name", "when", "flag"],
            [1, "alpha", ("date", 45000), True],
            [2, "beta", ("date", 45001), False],
        ]
    )
    frame = read_excel(path)
    assert frame.columns.tolist() == ["id", "name", "when", "flag"]
    assert frame.shape == (2, 4)


def test_column_dtypes(book):
    path = book(
        [
            ["num", "text", "when", "flag"],
            [1.5, "alpha", ("date", 45000), True],
            [2.5, "beta", ("date", 45001), False],
        ]
    )
    frame = read_excel(path)
    assert frame["num"].dtype == np.float64
    assert frame["when"].dtype.kind == "M"
    assert frame["flag"].dtype == bool
    assert frame["text"].dtype != np.float64


def test_blank_numeric_cell_is_null(book):
    path = book([["a", "b"], [1, None], [None, 4.5]])
    frame = read_excel(path)
    assert pd.isna(frame["b"][0])
    assert pd.isna(frame["a"][1])


def test_blank_date_is_null(book):
    path = book([["when", "label"], [("date", 45000), "x"], [None, "y"]])
    frame = read_excel(path)
    assert pd.isna(frame["when"][1])


def test_text_only_sheet_loads(book):
    path = book([["name"], ["alpha"], ["beta"]])
    frame = read_excel(path)
    assert frame["name"].tolist() == ["alpha", "beta"]


def test_numeric_columns_hand_off_to_numpy(book):
    path = book([["region", "units"], ["EMEA", 3.0], ["APAC", 5.0], ["EMEA", 7.0]])
    frame = read_excel(path)
    units = frame["units"].to_numpy()
    region = frame["region"].to_numpy()
    assert units.dtype == np.float64
    np.testing.assert_array_equal(units[region == "EMEA"], [3.0, 7.0])


# --------------------------------------------------------------------------
# headers and options
# --------------------------------------------------------------------------


def test_header_false_generates_positional_names(book):
    path = book([[1, 2], [3, 4]])
    frame = read_excel(path, header=False)
    assert frame.columns.tolist() == ["col0", "col1"]
    assert frame.shape == (2, 2)


def test_blank_header_cell_gets_positional_name(book):
    path = book([["a", None], [1, 2]])
    frame = read_excel(path)
    assert frame.columns.tolist() == ["a", "col1"]


def test_duplicate_headers_are_disambiguated(book):
    path = book([["score", "score"], [1, 2]])
    frame = read_excel(path)
    assert frame.columns.tolist() == ["score", "score_1"]


def test_n_rows_limits_the_read(book):
    path = book([["a"], [1], [2], [3], [4]])
    assert read_excel(path, n_rows=2).shape == (2, 1)


def test_sheet_selected_by_name(book):
    path = book([["a"], [1]])
    frame = read_excel(path, sheet="Sheet1")
    assert frame["a"].tolist() == [1.0]


def test_header_only_sheet_is_empty_not_an_error(book):
    path = book([["a", "b"]])
    frame = read_excel(path)
    assert frame.empty
    assert frame.columns.tolist() == ["a", "b"]


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


def test_unsupported_extension_lists_supported_ones(tmp_path):
    path = tmp_path / "data.parquet"
    path.write_bytes(b"")
    with pytest.raises(NumpyAIError, match=r"\.xlsx"):
        read_excel(str(path))


def test_unknown_sheet_name_raises(book):
    from fastexcel import SheetNotFoundError

    path = book([["a"], [1]])
    with pytest.raises(SheetNotFoundError):
        read_excel(path, sheet="NoSuchSheet")


# --------------------------------------------------------------------------
# delimited text
# --------------------------------------------------------------------------


@pytest.fixture
def csv_file(tmp_path):
    def _write(text, name="data.csv"):
        path = tmp_path / name
        path.write_text(text)
        return str(path)

    return _write


def test_read_csv_returns_a_dataframe(csv_file):
    frame = read_csv(csv_file("a,b\n1,2.5\n"))
    assert isinstance(frame, pd.DataFrame)
    assert frame.columns.tolist() == ["a", "b"]


def test_read_csv_blank_becomes_nan(csv_file):
    frame = read_csv(csv_file("a,b\n1,\n"))
    assert pd.isna(frame["b"][0])


def test_read_csv_header_false_uses_positional_names(csv_file):
    frame = read_csv(csv_file("1,2\n3,4\n"), header=False)
    assert frame.columns.tolist() == ["col0", "col1"]


def test_read_csv_n_rows_limits_the_read(csv_file):
    assert read_csv(csv_file("a\n1\n2\n3\n"), n_rows=2).shape == (2, 1)


def test_read_csv_infers_tab_separator(csv_file):
    frame = read_csv(csv_file("a\tb\n1\t2\n", name="data.tsv"))
    assert frame.columns.tolist() == ["a", "b"]


def test_read_csv_handles_compression(tmp_path):
    import gzip

    path = tmp_path / "data.csv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("a,b\n1,2\n")
    assert read_csv(str(path)).shape == (1, 2)


def test_read_csv_forwards_kwargs_to_pandas(csv_file):
    frame = read_csv(csv_file("a,b\n1,2\n"), usecols=["a"])
    assert frame.columns.tolist() == ["a"]


def test_read_csv_rejects_spreadsheets(tmp_path):
    path = tmp_path / "book.xlsx"
    path.write_bytes(b"")
    with pytest.raises(NumpyAIError, match="Use read_excel"):
        read_csv(str(path))


def test_read_excel_points_at_read_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(NumpyAIError, match="Use read_csv"):
        read_excel(str(path))

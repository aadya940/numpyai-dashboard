"""File loaders that turn spreadsheets into NumPy arrays.

Every column is preserved in the NumPy dtype that fits it: numbers become
``float64``, dates become ``datetime64``, and anything else becomes a unicode
string. Nothing is discarded.

A sheet whose columns are all numeric loads as a plain 2-D ``float64`` array. A
sheet with mixed types loads as a `structured array
<https://numpy.org/doc/stable/user/basics.rec.html>`_, whose fields are accessed
by name (``arr["score"]``) and which supports the filtering and grouping that
mixed data is usually loaded for.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._ai import DEFAULT_MODEL
from ._array import array
from ._exceptions import NumpyAIError

#: Extensions the calamine backend understands.
EXCEL_SUFFIXES = (".xlsx", ".xls", ".xlsb", ".ods")


def _load_calamine():
    """Import python-calamine, with an actionable message if it is missing."""
    try:
        import python_calamine
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "read_excel requires the 'python-calamine' package.\n"
            "Install it with:  pip install 'numpyai-dashboard[excel]'"
        ) from exc
    return python_calamine


def _is_blank(value: Any) -> bool:
    """True for cells calamine reports as empty."""
    return value is None or (isinstance(value, str) and not value.strip())


def _header_names(row: Sequence[Any], width: int) -> list[str]:
    """Build unique column names from a header row, filling blanks positionally.

    Names must be unique because they become structured-array field names, which
    NumPy does not allow to collide.
    """
    names: list[str] = []
    seen: dict[str, int] = {}
    for j in range(width):
        cell = row[j] if j < len(row) else None
        name = f"col{j}" if _is_blank(cell) else str(cell).strip()
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    return names


def _convert_column(values: Sequence[Any]) -> np.ndarray:
    """Convert one spreadsheet column to its natural NumPy dtype.

    Tried in order: float64, datetime64, then unicode text. The text fallback
    always succeeds, so no column is ever dropped.
    """
    cleaned = [None if _is_blank(v) else v for v in values]

    # Numbers, booleans and numeric text. Blanks become NaN.
    try:
        return np.asarray(
            [np.nan if v is None else v for v in cleaned], dtype=np.float64
        )
    except (TypeError, ValueError):
        pass

    # Dates and datetimes. Blanks become NaT.
    if all(v is None or isinstance(v, (_dt.date, _dt.datetime)) for v in cleaned):
        return np.array(
            [np.datetime64("NaT") if v is None else np.datetime64(v) for v in cleaned],
            dtype="datetime64[s]",
        )

    # Everything else, including times and mixed columns. Blanks become "".
    return np.array(["" if v is None else str(v) for v in cleaned], dtype=np.str_)


def read_excel(
    path: str,
    *,
    sheet: int | str = 0,
    header: bool = True,
    verbose: bool = False,
    model: Any = DEFAULT_MODEL,
    max_tries: int = 3,
) -> array:
    """Read a spreadsheet into a :class:`numpyai_dashboard.array`.

    Supports ``.xlsx``, ``.xls``, ``.xlsb`` and ``.ods`` via `python-calamine
    <https://github.com/dimastbk/python-calamine>`_.

    Every column is kept. Numbers and booleans become ``float64`` (blanks become
    ``NaN``), dates become ``datetime64[s]`` (blanks become ``NaT``), and
    anything else becomes text (blanks become ``""``).

    If every column is numeric the result is a plain 2-D ``float64`` array. If
    the types are mixed the result is a structured array whose fields carry the
    column names, so ``arr.data["region"] == "EMEA"`` and
    ``arr.data["date"] > np.datetime64("2023-04-01")`` both work.

    Parameters
    ----------
    path:
        Path to the workbook.
    sheet:
        Sheet index (default ``0``) or sheet name.
    header:
        Treat the first row as column names (default ``True``). When ``False``,
        columns are named ``col0``, ``col1``, ...
    verbose, model, max_tries:
        Forwarded to :class:`numpyai_dashboard.array`.

    Returns
    -------
    numpyai_dashboard.array
        Whose ``.columns`` holds the column names in order.

    Raises
    ------
    NumpyAIError
        If the file extension is unsupported, or the sheet has no data.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in EXCEL_SUFFIXES:
        hint = (
            " CSV is not supported yet."
            if suffix == ".csv"
            else f" Supported: {', '.join(EXCEL_SUFFIXES)}."
        )
        raise NumpyAIError(
            f"read_excel cannot read {suffix or 'extensionless'} files.{hint}"
        )

    calamine = _load_calamine()

    workbook = calamine.load_workbook(path)
    try:
        if isinstance(sheet, str):
            worksheet = workbook.get_sheet_by_name(sheet)
        else:
            worksheet = workbook.get_sheet_by_index(sheet)
        rows = worksheet.to_python()
    finally:
        workbook.close()

    if not rows:
        raise NumpyAIError(f"sheet {sheet!r} in {path!r} is empty")

    width = max(len(r) for r in rows)
    if width == 0:
        raise NumpyAIError(f"sheet {sheet!r} in {path!r} has no columns")

    if header:
        names = _header_names(rows[0], width)
        body = rows[1:]
    else:
        names = [f"col{j}" for j in range(width)]
        body = rows

    if not body:
        raise NumpyAIError(f"sheet {sheet!r} in {path!r} has a header but no data rows")

    # Pad ragged rows so the transpose below is well-formed.
    if any(len(r) != width for r in body):
        body = [list(r) + [None] * (width - len(r)) for r in body]

    # Rows are padded to `width` above, so the transpose is strictly aligned.
    columns = [_convert_column(column) for column in zip(*body, strict=True)]

    if all(col.dtype == np.float64 for col in columns):
        return array(
            np.column_stack(columns),
            columns=names,
            verbose=verbose,
            model=model,
            max_tries=max_tries,
        )

    record_dtype = np.dtype(
        [(name, col.dtype) for name, col in zip(names, columns, strict=True)]
    )
    table = np.empty(len(body), dtype=record_dtype)
    for name, col in zip(names, columns, strict=True):
        table[name] = col

    return array(table, verbose=verbose, model=model, max_tries=max_tries)

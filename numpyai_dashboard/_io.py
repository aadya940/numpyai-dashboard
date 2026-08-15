"""File loaders that turn spreadsheets into NumPy arrays.

NumpyAI operates on a homogeneous 2-D ``float64`` array, so loading a spreadsheet
means selecting the columns that can be represented that way. Columns that cannot
(text, dates) are reported rather than silently dropped - see :func:`read_excel`.
"""

from __future__ import annotations

import warnings
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
    """Build column names from a header row, filling blanks positionally."""
    names = []
    for j in range(width):
        cell = row[j] if j < len(row) else None
        names.append(f"col{j}" if _is_blank(cell) else str(cell).strip())
    return names


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

    Only columns that convert cleanly to ``float64`` are kept, because the array
    handed to the LLM must be homogeneous and numeric. Empty cells become ``NaN``
    and booleans become ``1.0``/``0.0``. Text and date columns are excluded and
    listed in a :class:`UserWarning`, so nothing disappears without notice.

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
        A 2-D float64 array whose ``.columns`` holds the retained column names.

    Raises
    ------
    NumpyAIError
        If the sheet is empty, or if no column can be represented numerically.
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

    kept_names: list[str] = []
    kept_columns: list[np.ndarray] = []
    rejected: list[str] = []

    # Rows are padded to `width` above, so both zips are strictly aligned.
    for name, column in zip(names, zip(*body, strict=True), strict=True):
        cleaned = [np.nan if _is_blank(v) else v for v in column]
        try:
            kept_columns.append(np.asarray(cleaned, dtype=np.float64))
        except (TypeError, ValueError):
            rejected.append(name)
            continue
        kept_names.append(name)

    if rejected:
        warnings.warn(
            f"read_excel dropped {len(rejected)} non-numeric column(s): "
            f"{', '.join(rejected)}. NumpyAI arrays are numeric-only.",
            stacklevel=2,
        )

    if not kept_columns:
        raise NumpyAIError(
            f"no numeric columns in sheet {sheet!r} of {path!r} "
            f"(found: {', '.join(names)})"
        )

    data = np.column_stack(kept_columns)
    return array(
        data,
        columns=kept_names,
        verbose=verbose,
        model=model,
        max_tries=max_tries,
    )

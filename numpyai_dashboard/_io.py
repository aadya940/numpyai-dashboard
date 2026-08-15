"""Spreadsheet loading.

Reading is delegated to `fastexcel <https://github.com/ToucanToco/fastexcel>`_,
which wraps the Rust `calamine <https://github.com/tafia/calamine>`_ parser and
emits Arrow data directly. Type inference happens in Rust and the result crosses
into Python as columnar buffers rather than an object per cell, which is roughly
3x faster and uses about half the memory of driving calamine from Python.

The result is a :class:`pandas.DataFrame`. Numeric columns can be handed to NumPy
with ``df["units"].to_numpy()`` at essentially no cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ._exceptions import NumpyAIError

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

#: Extensions the calamine backend understands.
EXCEL_SUFFIXES = (".xlsx", ".xls", ".xlsb", ".ods")

#: fastexcel names headerless columns __UNNAMED__0, __UNNAMED__1, ...
_UNNAMED_PREFIX = "__UNNAMED__"


def _load_fastexcel():
    """Import fastexcel, with an actionable message if it is missing."""
    try:
        import fastexcel
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "read_excel requires the 'fastexcel' package.\n"
            "Install it with:  pip install 'numpyai-dashboard[excel]'"
        ) from exc
    return fastexcel


def read_excel(
    path: str,
    *,
    sheet: int | str = 0,
    header: bool = True,
    n_rows: int | None = None,
) -> pd.DataFrame:
    """Read a spreadsheet into a :class:`pandas.DataFrame`.

    Supports ``.xlsx``, ``.xls``, ``.xlsb`` and ``.ods``.

    Every column is kept, with its type inferred by the Rust reader: numbers
    become ``float64``, whole numbers ``int64``, dates ``datetime64``, ``TRUE``/
    ``FALSE`` ``bool``, and anything else string. Blank cells become the null of
    whichever type the column is.

    Parameters
    ----------
    path:
        Path to the workbook.
    sheet:
        Sheet index (default ``0``) or sheet name.
    header:
        Treat the first row as column names (default ``True``). When ``False``,
        columns are named ``col0``, ``col1``, ...
    n_rows:
        Read at most this many data rows. Useful for previewing a large sheet.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    NumpyAIError
        If the file extension is not a supported spreadsheet format.
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

    fastexcel = _load_fastexcel()

    reader = fastexcel.read_excel(path)
    worksheet = reader.load_sheet(
        sheet,
        header_row=0 if header else None,
        n_rows=n_rows,
    )
    frame = worksheet.to_pandas()

    # fastexcel names unnamed columns __UNNAMED__<i>, both for headerless sheets
    # and for individual blank header cells. Present them as col<i> either way.
    frame.columns = [
        (
            f"col{str(name).removeprefix(_UNNAMED_PREFIX)}"
            if str(name).startswith(_UNNAMED_PREFIX)
            else name
        )
        for name in frame.columns
    ]

    return frame

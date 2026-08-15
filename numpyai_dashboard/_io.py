"""Tabular file loading.

Spreadsheet reading is delegated to `fastexcel
<https://github.com/ToucanToco/fastexcel>`_, which wraps the Rust `calamine
<https://github.com/tafia/calamine>`_ parser and emits Arrow data directly. Type
inference happens in Rust and the result crosses into Python as columnar buffers
rather than an object per cell, which is roughly 3x faster and uses about half
the memory of driving calamine from Python.

Delimited text goes through :func:`pandas.read_csv`, whose C parser is already
the fast path for that format.

Both return a :class:`pandas.DataFrame`. Numeric columns can be handed to NumPy
with ``df["units"].to_numpy()`` at essentially no cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._exceptions import NumpyAIError

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

#: Extensions the calamine backend understands.
EXCEL_SUFFIXES = (".xlsx", ".xls", ".xlsb", ".ods")

#: Extensions read as delimited text, before any compression suffix.
TEXT_SUFFIXES = (".csv", ".tsv", ".txt")

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


def _load_pandas():
    """Import pandas, with an actionable message if it is missing."""
    try:
        import pandas
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "read_csv requires the 'pandas' package.\n"
            "Install it with:  pip install 'numpyai-dashboard[csv]'"
        ) from exc
    return pandas


def _positional_names(columns) -> list[str]:
    """Rename unnamed columns to col0, col1, ... for a stable contract."""
    return [
        (
            f"col{str(name).removeprefix(_UNNAMED_PREFIX)}"
            if str(name).startswith(_UNNAMED_PREFIX)
            else f"col{name}" if isinstance(name, int) else name
        )
        for name in columns
    ]


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
            " Use read_csv for delimited text."
            if suffix in TEXT_SUFFIXES
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
    frame.columns = _positional_names(frame.columns)

    return frame


def read_csv(
    path: str,
    *,
    header: bool = True,
    n_rows: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read delimited text into a :class:`pandas.DataFrame`.

    A thin wrapper over :func:`pandas.read_csv` that matches
    :func:`read_excel`'s contract: the same ``header`` and ``n_rows`` arguments,
    and ``col0``, ``col1``, ... for unnamed columns. The separator is inferred
    from the extension (tab for ``.tsv``), and compressed files such as
    ``.csv.gz`` are handled by pandas.

    Any other keyword is forwarded to :func:`pandas.read_csv` unchanged, so
    ``sep``, ``encoding``, ``usecols``, ``dtype`` and the rest remain available.

    Parameters
    ----------
    path:
        Path to the file.
    header:
        Treat the first row as column names (default ``True``). When ``False``,
        columns are named ``col0``, ``col1``, ...
    n_rows:
        Read at most this many data rows.
    **kwargs:
        Passed through to :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    NumpyAIError
        If the path looks like a spreadsheet, which :func:`read_excel` handles.
    """
    suffixes = [s.lower() for s in Path(path).suffixes]
    if any(s in EXCEL_SUFFIXES for s in suffixes):
        raise NumpyAIError(
            f"read_csv cannot read {suffixes[-1]} files. Use read_excel instead."
        )

    pandas = _load_pandas()

    kwargs.setdefault("sep", "\t" if ".tsv" in suffixes else ",")
    frame = pandas.read_csv(
        path,
        header=0 if header else None,
        nrows=n_rows,
        **kwargs,
    )

    # With header=False pandas numbers the columns 0, 1, 2; match read_excel.
    frame.columns = _positional_names(frame.columns)

    return frame

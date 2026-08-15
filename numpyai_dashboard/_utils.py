"""Utility helpers: metadata collection and code cleanup."""

from __future__ import annotations

import contextlib
import importlib
import re
from collections.abc import Sequence

import numpy as np

#: A text column with at most this many distinct values is listed in full.
_MAX_CATEGORIES = 20

#: Bounds on the array preview embedded in the LLM prompt.
_PREVIEW_ROWS = 10
_PREVIEW_COLS = 20


#: Packages whose submodules are not imported eagerly, so they need the proxy.
_LAZY_PACKAGES = ("scipy", "sklearn")


class _SubmodulePackage:
    """Wraps a package so ``pkg.submodule`` imports on first access.

    SciPy and scikit-learn do not import their submodules eagerly, so generated
    code calling ``scipy.stats.ttest_ind`` or ``sklearn.linear_model.Ridge``
    would raise ``AttributeError`` even with the package itself available.
    Importing every submodule up front would cost seconds on the first query
    and guess at which ones matter, so the cost is deferred to first use.
    """

    def __init__(self, module) -> None:
        self._module = module

    def __getattr__(self, name: str):
        # Guards against recursing through _module before __init__ assigns it.
        if name == "_module":
            raise AttributeError(name)
        try:
            return getattr(self._module, name)
        except AttributeError:
            pass
        try:
            return importlib.import_module(f"{self._module.__name__}.{name}")
        except ImportError as exc:
            raise AttributeError(
                f"{self._module.__name__} has no attribute or submodule {name!r}"
            ) from exc

    def __dir__(self) -> list[str]:
        return dir(self._module)

    def __repr__(self) -> str:
        return f"<{self._module.__name__} (submodules imported on demand)>"


def optional_globals() -> dict:
    """Optional libraries exposed to generated code, when they are installed.

    Generated code is told not to import anything, so whatever it may reach for
    has to be in the namespace already. Each library is optional; a missing one
    is simply absent rather than an error.
    """
    namespace: dict = {}

    with contextlib.suppress(ImportError):
        import matplotlib.pyplot as plt

        namespace["plt"] = plt

    for name in _LAZY_PACKAGES:
        with contextlib.suppress(ImportError):
            namespace[name] = _SubmodulePackage(importlib.import_module(name))

    return namespace


class NumpyMetadataCollector:
    """Collect metadata from NumPy arrays and NumPy-operation outputs."""

    def metadata(self, data: np.ndarray, columns: Sequence[str] | None = None) -> dict:
        """Collect metadata about the given NumPy array.

        ``columns`` optionally names the columns of a 2-D array (populated by the
        file loaders) so the LLM knows what each column represents.
        """
        md: dict = {
            "is_numpy": isinstance(data, np.ndarray),
            "dims": data.ndim,
            "shape": data.shape,
            "size": data.size,
            "element_type": data.dtype,
            "byte_size": data.nbytes,
        }

        if columns is not None:
            md["columns"] = list(columns)

        if np.issubdtype(data.dtype, np.number):
            try:
                # Compute the NaN mask once - it is needed both for `has_nan` and
                # for the all-NaN guard below, and a second pass over a large
                # array is expensive.
                nan_mask = np.isnan(data)
                md["has_nan"] = bool(nan_mask.any())
                md["has_inf"] = bool(np.isinf(data).any())
                if data.size > 0 and not nan_mask.all():
                    md["min"] = float(np.nanmin(data))
                    md["max"] = float(np.nanmax(data))
            except (TypeError, ValueError):
                pass

        if (
            data.size > 0
            and data.size <= 10_000
            and np.issubdtype(data.dtype, np.number)
        ):
            try:
                md["zeros_count"] = int(np.count_nonzero(data == 0))
                md["non_zeros_count"] = int(np.count_nonzero(data))
            except (TypeError, ValueError):
                pass

        if data.ndim >= 1:
            try:
                # The preview is interpolated straight into the LLM prompt, so it
                # must stay small regardless of how large the array is.
                preview = data[: min(len(data), _PREVIEW_ROWS)]
                if preview.ndim > 1 and preview.shape[1] > _PREVIEW_COLS:
                    preview = preview[:, :_PREVIEW_COLS]
                md["array-preview"] = preview
                if preview.shape != data.shape:
                    md["array-preview-truncated"] = True
            except TypeError:
                pass

        if data.size > 1_000_000:
            md["large_array"] = True

        return md

    @staticmethod
    def collect_output_metadata(output) -> dict:
        """Collect metadata about a NumPy operation output."""
        metadata: dict = {"type": type(output).__name__}

        if isinstance(output, np.ndarray):
            metadata.update(
                {
                    "shape": output.shape,
                    "ndim": output.ndim,
                    "size": output.size,
                    "dtype": str(output.dtype),
                    "memory_size": output.nbytes,
                    "is_contiguous": output.flags.contiguous,
                    "is_fortran": output.flags.f_contiguous,
                    "has_nan": (
                        bool(np.isnan(output).any())
                        if np.issubdtype(output.dtype, np.number)
                        else False
                    ),
                    "has_inf": (
                        bool(np.isinf(output).any())
                        if np.issubdtype(output.dtype, np.number)
                        else False
                    ),
                    "is_structured": np.issubdtype(output.dtype, np.void),
                }
            )

            # Non-numeric dtypes raise here; the summary stats are best-effort.
            with contextlib.suppress(TypeError, ValueError):
                metadata.update(
                    {
                        "min": float(output.min()),
                        "max": float(output.max()),
                        "mean": float(output.mean()),
                        "std": float(output.std()),
                    }
                )

            if output.size > 0:
                sample_size = min(5, output.size)
                # `.flat` is a view-based iterator; `.flatten()` would copy the
                # whole array just to read a handful of elements.
                metadata["first_elements"] = output.flat[:sample_size].tolist()
                if output.size > sample_size * 2:
                    metadata["last_elements"] = output.flat[-sample_size:].tolist()

            if output.size > 1_000_000:
                metadata["large_array"] = True
            if not output.flags.contiguous and not output.flags.f_contiguous:
                metadata["non_contiguous"] = True

        elif np.isscalar(output):
            metadata["value"] = output
            if hasattr(output, "dtype"):
                metadata["dtype"] = str(output.dtype)

        elif isinstance(output, (list, tuple)):
            metadata.update(
                {
                    "length": len(output),
                    "sample": output[:5] if len(output) > 5 else output,
                }
            )

        elif output is None:
            metadata["is_none"] = True

        elif isinstance(output, str):
            metadata.update(
                {
                    "length": len(output),
                    "preview": output[:100] + "..." if len(output) > 100 else output,
                }
            )

        return metadata


_FENCE_RE = re.compile(r"^\s*```(?:\w+)?\s*|\s*```\s*$", re.MULTILINE)


def clean_code(code: str) -> str:
    """Strip markdown code fences from an LLM response."""
    return _FENCE_RE.sub("", code).strip()


def describe_column(series) -> dict:
    """Summarise one DataFrame column for the LLM prompt."""
    import pandas as pd

    info: dict = {"dtype": str(series.dtype)}
    n_null = int(series.isna().sum())
    if n_null:
        info["nulls"] = n_null

    if pd.api.types.is_bool_dtype(series):
        info["kind"] = "boolean"
        info["true_count"] = int(series.sum())

    elif pd.api.types.is_numeric_dtype(series):
        info["kind"] = "numeric"
        present = series.dropna()
        if len(present):
            info["min"] = float(present.min())
            info["max"] = float(present.max())

    elif pd.api.types.is_datetime64_any_dtype(series):
        info["kind"] = "datetime"
        present = series.dropna()
        if len(present):
            info["min"] = str(present.min())
            info["max"] = str(present.max())

    else:
        info["kind"] = "text"
        distinct = series.dropna().unique()
        info["n_unique"] = int(len(distinct))
        if len(distinct) <= _MAX_CATEGORIES:
            info["categories"] = [str(v) for v in distinct]
        else:
            info["examples"] = [str(v) for v in distinct[:5]]

    return info


def frame_metadata(frame) -> dict:
    """Collect metadata about a DataFrame, column by column.

    The per-column detail is the point: a model that knows `region` holds four
    named values can answer questions about them, where one told only that the
    frame is 150x9 cannot.
    """
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "column_summary": {
            str(name): describe_column(frame[name]) for name in frame.columns
        },
    }

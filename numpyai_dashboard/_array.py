"""Single-array natural-language interface."""

from __future__ import annotations

from collections.abc import Sequence
from operator import add, floordiv, matmul, mod, mul, pow, sub, truediv
from typing import Any

import numpy as np
from rich.console import Console

from ._ai import DEFAULT_MODEL, ChatResult, NumpyCodeGen
from ._engine import run_chat
from ._utils import NumpyMetadataCollector
from ._validator import NumpyValidator

console = Console()


class array:
    """A wrapper around ``numpy.ndarray`` with an AI-powered ``.chat()`` method.

    Parameters
    ----------
    data:
        The underlying NumPy array.
    verbose:
        If True, print every LLM step. If False, only print on the final attempt.
    model:
        Any model spec accepted by ``pydantic-ai`` (default:
        ``"google:gemini-2.5-flash"``).
    max_tries:
        Number of code-generation attempts before giving up (default: 3).
    columns:
        Optional names for the columns of a 2-D array, passed to the LLM so it can
        refer to columns by name instead of by index. Handy when bridging from a
        DataFrame: ``array(df[cols].to_numpy(), columns=cols)``.
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        verbose: bool = False,
        model: Any = DEFAULT_MODEL,
        max_tries: int = 3,
        columns: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be a numpy.ndarray, got {type(data).__name__}")

        if columns is not None:
            columns = list(columns)
            if data.ndim != 2:
                raise ValueError(f"columns requires a 2-D array, got {data.ndim}-D")
            if len(columns) != data.shape[1]:
                raise ValueError(
                    f"got {len(columns)} column names for {data.shape[1]} columns"
                )

        self._columns = columns
        self._data = data
        self._metadata_collector = NumpyMetadataCollector()
        self._validator = NumpyValidator()
        self._code_generator = NumpyCodeGen(model=model)
        self.MAX_TRIES = max_tries
        self.verbose = verbose

        self.current_prompt: str | None = None
        self.metadata = self._metadata_collector.metadata(self._data, self.columns)
        self._model = model

    @property
    def columns(self) -> list[str] | None:
        """Names for the columns of a 2-D array, if any were given."""
        return self._columns

    # ------------------------------------------------------------------
    # numpy interop
    # ------------------------------------------------------------------
    def _apply_operator(self, other, op):
        other_data = other._data if isinstance(other, array) else other
        return array(op(self._data, other_data))

    def _apply_r_operator(self, other, op):
        return array(op(other, self._data))

    def __add__(self, other):
        return self._apply_operator(other, add)

    def __sub__(self, other):
        return self._apply_operator(other, sub)

    def __mul__(self, other):
        return self._apply_operator(other, mul)

    def __truediv__(self, other):
        return self._apply_operator(other, truediv)

    def __floordiv__(self, other):
        return self._apply_operator(other, floordiv)

    def __mod__(self, other):
        return self._apply_operator(other, mod)

    def __pow__(self, other):
        return self._apply_operator(other, pow)

    def __matmul__(self, other):
        return self._apply_operator(other, matmul)

    def __radd__(self, other):
        return self._apply_r_operator(other, add)

    def __rsub__(self, other):
        return self._apply_r_operator(other, sub)

    def __rmul__(self, other):
        return self._apply_r_operator(other, mul)

    def __rtruediv__(self, other):
        return self._apply_r_operator(other, truediv)

    def __rfloordiv__(self, other):
        return self._apply_r_operator(other, floordiv)

    def __rmod__(self, other):
        return self._apply_r_operator(other, mod)

    def __rpow__(self, other):
        return self._apply_r_operator(other, pow)

    def __rmatmul__(self, other):
        return self._apply_r_operator(other, matmul)

    def __getitem__(self, index):
        return array(self._data[index])

    def __setitem__(self, index, value) -> None:
        self._data[index] = value

    def __repr__(self) -> str:
        return (
            f"numpyai_dashboard.array(shape={self._data.shape}, "
            f"dtype={self._data.dtype})"
        )

    def __getattr__(self, name):
        attr = getattr(self._data, name)
        if callable(attr):

            def method_proxy(*args, **kwargs):
                result = attr(*args, **kwargs)
                return array(result) if isinstance(result, np.ndarray) else result

            return method_proxy
        return attr

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_array: np.ndarray) -> None:
        # Column names only survive if the new array is still the same width.
        if self._columns is not None and (
            new_array.ndim != 2 or new_array.shape[1] != len(self._columns)
        ):
            self._columns = None
        self._data = new_array
        self.metadata = self._metadata_collector.metadata(self._data, self.columns)

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------
    def chat(self, query: str) -> ChatResult:
        """Answer a natural-language query by generating and executing NumPy code.

        Returns a :class:`ChatResult` carrying the answer in ``.value`` along
        with the code that produced it, the judgment, and any errors. Failure is
        reported through ``.ok`` rather than by raising, so a bad query never
        ends a session.
        """
        self.current_prompt = query
        return run_chat(
            query,
            data_vars={"arr": self._data},
            build_prompt=lambda prior: self._code_generator.prompt_single(
                query=query, metadata=self.metadata, prior_feedback=prior
            ),
            generator=self._code_generator,
            validator=self._validator,
            max_tries=self.MAX_TRIES,
            verbose=self.verbose,
        )

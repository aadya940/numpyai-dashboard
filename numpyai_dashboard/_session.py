"""Multi-object session for chatting over several arrays or tables at once."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._ai import DEFAULT_MODEL, ChatResult, NumpyCodeGen, prompt_multiple
from ._array import array
from ._engine import run_chat
from ._frame import frame
from ._utils import NumpyMetadataCollector, frame_metadata
from ._validator import NumpyValidator


def _unwrap(item: Any, index: int) -> tuple[str, Any]:
    """Return ``(kind, data)`` for one session input.

    Accepts a NumPy array or a pandas DataFrame, wrapped or bare.
    """
    if isinstance(item, array):
        return "array", item.data
    if isinstance(item, frame):
        return "frame", item.data
    if isinstance(item, np.ndarray):
        return "array", item

    # Checked last so pandas is only imported when something might need it.
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - depends on install extras
        pd = None
    if pd is not None and isinstance(item, pd.DataFrame):
        return "frame", item

    raise TypeError(
        "session data must be a numpy.ndarray, a pandas.DataFrame, or the "
        f"numpyai_dashboard wrapper of either, got {type(item).__name__} "
        f"at index {index}"
    )


class NumpyAISession:
    """Chat across several arrays or tables at once.

    Each input is exposed to the model under a positional name: arrays as
    ``arr1``, ``arr2``, ... and DataFrames as ``df1``, ``df2``, ... The number
    always matches the position in ``data``, so a mixed session reads
    ``arr1``, ``df2``, ``arr3`` and "the second one" is unambiguous.

    Parameters
    ----------
    data:
        List of ``numpy.ndarray`` or ``pandas.DataFrame`` objects, or the
        :class:`~numpyai_dashboard.array` and :class:`~numpyai_dashboard.frame`
        wrappers of either.
    verbose:
        Show all intermediate LLM steps.
    model:
        Any model spec accepted by ``pydantic-ai`` (default:
        ``"google:gemini-2.5-flash"``).
    max_tries:
        Number of code-generation attempts before giving up (default: 3).
    """

    def __init__(
        self,
        data: list[Any],
        *,
        verbose: bool = False,
        model: Any = DEFAULT_MODEL,
        max_tries: int = 3,
    ) -> None:
        self._context: dict[str, dict[str, Any]] = {}
        self._metadata_collector = NumpyMetadataCollector()
        self._code_generator = NumpyCodeGen(model=model)
        self._validator = NumpyValidator()

        self._initialize(data)
        self.current_prompt: str | None = None
        self.verbose = verbose
        self.MAX_TRIES = max_tries
        self._model = model

    def _initialize(self, data: list[Any]) -> None:
        for i, item in enumerate(data, start=1):
            kind, unwrapped = _unwrap(item, i - 1)
            name = f"arr{i}" if kind == "array" else f"df{i}"
            self._context[name] = {"kind": kind, "data": unwrapped}

    @property
    def context(self) -> dict[str, dict[str, Any]]:
        """Each input with its current metadata, keyed by the name the model sees.

        Metadata is recomputed on access so a session holding a frame that has
        since been filtered describes the rows it holds now.
        """
        described = {}
        for name, info in self._context.items():
            metadata = (
                frame_metadata(info["data"])
                if info["kind"] == "frame"
                else self._metadata_collector.metadata(info["data"])
            )
            described[name] = {**info, "metadata": metadata}
        return described

    def chat(self, query: str) -> ChatResult:
        """Answer a natural-language query across the session's inputs.

        Returns a :class:`ChatResult` carrying the answer in ``.value`` along
        with the code that produced it, the judgment, and any errors. Failure is
        reported through ``.ok`` rather than by raising.
        """
        self.current_prompt = query
        context = self.context

        data_vars: dict[str, Any] = {
            name: info["data"] for name, info in context.items()
        }
        if any(info["kind"] == "frame" for info in context.values()):
            import pandas as pd

            data_vars["pd"] = pd

        return run_chat(
            query,
            data_vars=data_vars,
            build_prompt=lambda prior: prompt_multiple(
                query=query, context=context, prior_feedback=prior
            ),
            generator=self._code_generator,
            validator=self._validator,
            max_tries=self.MAX_TRIES,
            verbose=self.verbose,
        )

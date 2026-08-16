"""Natural-language interface over a pandas DataFrame."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._ai import DEFAULT_MODEL, ChatResult, NumpyCodeGen, prompt_frame
from ._engine import run_chat
from ._utils import frame_metadata
from ._validator import NumpyValidator

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


class frame:
    """A pandas DataFrame with an AI-powered ``.chat()`` method.

    Where :class:`numpyai_dashboard.array` shows the model one homogeneous
    NumPy array, this shows it the whole table: every column with its own dtype,
    the range of the numeric ones, and the distinct values of the categorical
    ones. Questions like "revenue by region since March" need columns that a
    single array cannot hold together.

    Attribute and item access forward to the underlying DataFrame, so ``f.head()``,
    ``f["units"]`` and ``f["revenue"] = ...`` all work. Use :attr:`data` when
    something needs the real DataFrame, such as a Panel ``Tabulator``.

    Parameters
    ----------
    data:
        The DataFrame to wrap.
    verbose:
        If True, print every LLM step. If False, only print on the final attempt.
    model:
        Any model spec accepted by ``pydantic-ai`` (default:
        ``"google:gemini-2.5-flash"``).
    max_tries:
        Number of code-generation attempts before giving up (default: 3).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        verbose: bool = False,
        model: Any = DEFAULT_MODEL,
        max_tries: int = 3,
    ) -> None:
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"data must be a pandas.DataFrame, got {type(data).__name__}"
            )

        self._data = data
        self._validator = NumpyValidator()
        self._code_generator = NumpyCodeGen(model=model)
        self.MAX_TRIES = max_tries
        self.verbose = verbose
        self.current_prompt: str | None = None
        self._model = model
        #: (question, code, description) per successful turn, oldest first.
        #: Rendered into the prompt so follow-ups like "explain it" resolve.
        self.history: list[tuple[str, str, str]] = []

    # ------------------------------------------------------------------
    # pandas interop
    # ------------------------------------------------------------------
    @property
    def data(self) -> pd.DataFrame:
        """The underlying DataFrame."""
        return self._data

    @data.setter
    def data(self, new_frame: pd.DataFrame) -> None:
        self._data = new_frame

    def __getattr__(self, name: str):
        # Guards against recursing through _data before __init__ assigns it.
        if name == "_data":
            raise AttributeError(name)
        return getattr(self._data, name)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value) -> None:
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        rows, cols = self._data.shape
        return f"numpyai_dashboard.frame({rows} rows x {cols} columns)"

    def _repr_html_(self):  # pragma: no cover - notebook display only
        return self._data._repr_html_()

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------
    @property
    def metadata(self) -> dict:
        """Per-column description of the current frame.

        Computed on access rather than cached, so a frame that has been filtered
        or had a column added describes what it holds now. A stale description
        would let the model answer confidently about rows that are gone.
        """
        return frame_metadata(self._data)

    def chat(self, query: str) -> ChatResult:
        """Answer a natural-language query about the table.

        Returns a :class:`ChatResult` carrying the answer in ``.value`` along
        with the code that produced it, the judgment, and any errors. Failure is
        reported through ``.ok`` rather than by raising.

        The generated code sees ``df`` along with ``pd`` and ``np``, so it can
        answer with pandas or NumPy idioms depending on which suits the question.
        """
        self.current_prompt = query
        metadata = self.metadata
        result = run_chat(
            query,
            data_vars={"df": self._data, "pd": _pandas()},
            build_prompt=lambda prior: prompt_frame(
                query=query,
                metadata=metadata,
                prior_feedback=prior,
                history=self.history,
            ),
            generator=self._code_generator,
            validator=self._validator,
            max_tries=self.MAX_TRIES,
            verbose=self.verbose,
        )
        if result.ok:
            self.history.append((query, result.code, result.description))
            del self.history[:-8]
        return result


def _pandas():
    import pandas

    return pandas

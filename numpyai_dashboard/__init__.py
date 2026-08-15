"""NumpyAI - natural-language interface for NumPy, powered by LLMs."""

from ._ai import ChatResult, Judgment
from ._array import array
from ._diagnosis import Diagnosis
from ._io import read_csv, read_excel
from ._session import NumpyAISession

__all__ = [
    "array",
    "NumpyAISession",
    "Diagnosis",
    "ChatResult",
    "Judgment",
    "read_csv",
    "read_excel",
]

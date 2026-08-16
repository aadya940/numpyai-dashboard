"""NumpyAI - natural-language interface for NumPy, powered by LLMs."""

from ._ai import ChatResult, Judgment
from ._array import array
from ._diagnosis import Diagnosis
from ._frame import frame
from ._io import read_csv, read_excel
from ._memory import AgentMemory
from ._session import NumpyAISession

__all__ = [
    "array",
    "NumpyAISession",
    "Diagnosis",
    "frame",
    "AgentMemory",
    "ChatResult",
    "Judgment",
    "read_csv",
    "read_excel",
]

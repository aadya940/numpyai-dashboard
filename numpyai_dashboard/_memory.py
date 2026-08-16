"""Long-term memory across sessions, backed by `mem0 <https://mem0.ai>`_.

The frame's built-in history covers one conversation. This remembers across
restarts: past findings, phrasing the user prefers, quirks of a dataset. What
comes back is *context*, never answers - retrieved memories are rendered into
the prompt with an instruction to recompute any number stated, because the
data may have changed since the memory was written.

Writes go through a background thread: mem0 runs an LLM extraction on every
add, which takes seconds, and a chat turn should not wait on bookkeeping.
"""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from typing import Any

_DEFAULT_DIR = Path.home() / ".numpyai_dashboard" / "memory"


class AgentMemory:
    """Remember (question, answer) pairs and recall the relevant ones.

    Parameters
    ----------
    user_id:
        Scope for memories - pass the dataset name so findings about one
        spreadsheet do not surface while analysing another.
    config:
        Optional full mem0 config dict, replacing the default (Gemini LLM and
        embedder, local on-disk Qdrant under ``~/.numpyai_dashboard/memory``).
    """

    def __init__(self, *, user_id: str = "default", config: dict | None = None):
        # mem0 does not honour the app's choice here, so make it explicit:
        # no telemetry, and accept GEMINI_API_KEY where it wants GOOGLE_API_KEY.
        os.environ.setdefault("MEM0_TELEMETRY", "False")
        if "GOOGLE_API_KEY" not in os.environ and os.environ.get("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        try:
            from mem0 import Memory
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "AgentMemory requires the 'mem0ai' package.\n"
                "Install it with:  pip install 'numpyai-dashboard[memory]'"
            ) from exc

        if config is None:
            _DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
            config = {
                "llm": {
                    "provider": "gemini",
                    "config": {"model": "gemini-2.5-flash"},
                },
                "embedder": {
                    "provider": "gemini",
                    "config": {
                        "model": "models/gemini-embedding-001",
                        "embedding_dims": 768,
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "path": str(_DEFAULT_DIR / "qdrant"),
                        "collection_name": "numpyai_dashboard",
                        "embedding_model_dims": 768,
                        "on_disk": True,
                    },
                },
            }

        self._memory = Memory.from_config(config)
        self.user_id = user_id
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._worker = threading.Thread(
            target=self._drain, daemon=True, name="numpyai-memory"
        )
        self._worker.start()

    # -- reads ---------------------------------------------------------------

    def recall(self, query: str, k: int = 5) -> list[str]:
        """The ``k`` stored memories most relevant to ``query``.

        Failure returns an empty list: memory is an enhancement, and a chat
        turn must never die because the vector store hiccuped.
        """
        try:
            found: Any = self._memory.search(
                query=query, filters={"user_id": self.user_id}, limit=k
            )
        except Exception:
            return []
        if isinstance(found, dict):
            found = found.get("results", [])
        return [m["memory"] for m in found if isinstance(m, dict) and m.get("memory")]

    # -- writes --------------------------------------------------------------

    def remember(self, question: str, answer: str) -> None:
        """Queue one exchange for storage. Returns immediately."""
        self._queue.put((question, answer))

    def flush(self, timeout: float | None = None) -> None:
        """Block until queued writes are stored. Mainly for tests and scripts."""
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                if not self._queue.all_tasks_done.wait(timeout):
                    return

    def _drain(self) -> None:
        while True:
            question, answer = self._queue.get()
            try:
                self._memory.add(
                    messages=[
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    user_id=self.user_id,
                )
            except Exception:
                pass
            finally:
                self._queue.task_done()

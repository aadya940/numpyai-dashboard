"""LLM-driven NumPy code generation via Pydantic AI."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable
from functools import cached_property
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent

T = TypeVar("T")

_worker_loop: asyncio.AbstractEventLoop | None = None
_worker_lock = threading.Lock()


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    """Return a module-wide, long-lived event loop running in a daemon thread.

    Reused across all calls so that clients (httpx, etc.) held by Pydantic AI
    agents stay bound to a live loop between successive ``.chat()`` calls.
    """
    global _worker_loop
    with _worker_lock:
        if _worker_loop is None or _worker_loop.is_closed():
            _worker_loop = asyncio.new_event_loop()
            threading.Thread(
                target=_worker_loop.run_forever,
                daemon=True,
                name="numpyai-dashboard-asyncio",
            ).start()
        return _worker_loop


def _run_coro(coro: Awaitable[T]) -> T:
    """Run a coroutine to completion from sync code.

    Works in scripts, in Jupyter (where an event loop is already running), and
    across multiple sequential calls without invalidating async HTTP clients.
    """
    fut = asyncio.run_coroutine_threadsafe(coro, _get_worker_loop())  # type: ignore[arg-type]
    return fut.result()


DEFAULT_MODEL = "google:gemini-2.5-flash"

SYSTEM_PROMPT = (
    "You are a coding assistant who generates only Python code, using NumPy and "
    "pandas. You respond with executable Python that operates on pre-defined "
    "arrays and DataFrames."
)


class CodeResponse(BaseModel):
    """Structured response returned by the code-generation agent.

    Exactly one of ``code`` and ``advice`` is filled. Computations produce
    code; questions asking for guidance produce advice, because forcing
    methodology questions through exec() yields a dict dump, not an answer.
    """

    code: str = Field(
        default="",
        description=(
            "Executable Python/NumPy code. Must define `output` (the result) and "
            "`metadata` (a short string describing `output`). No markdown fences. "
            "Leave empty when answering with `advice` instead."
        ),
    )
    advice: str = Field(
        default="",
        description=(
            "Only for questions asking HOW to analyse, what to explore, or for "
            "methodology - not for computations. A concise markdown answer that "
            "names the actual columns of this dataset; inline LaTeX ($...$) "
            "renders. Leave empty when answering with `code`."
        ),
    )
    chat_only: bool = Field(
        default=False,
        description=(
            "True when the answer is a quick fact that belongs in the "
            "conversation, not on a dashboard - a row count, a yes/no, a "
            "single lookup. False for any analysis worth keeping as a card."
        ),
    )
    explanation: str = Field(
        description="One-sentence natural-language explanation of the response."
    )


class Judgment(BaseModel):
    """Independent judgment of whether generated code answers the user's query.

    This is a *classification* task, not a code-generation task - so it cannot
    fail with syntax errors, missing names, or misapplied math (all failure
    modes of the previous LLM-rewrites-verification-code design).
    """

    interprets_query_correctly: bool = Field(
        description=(
            "True iff the generated code computes what the user's query asks for. "
            "Judge intent only; do not re-derive the answer."
        )
    )
    reason: str = Field(
        description=(
            "Short reason. If False, name the specific misinterpretation "
            "(e.g. 'query asked for mean, code computes median'). If True, empty."
        ),
        default="",
    )


class ChatResult(BaseModel):
    """Everything one natural-language query produced.

    The console shows all of this as it happens, which is enough for a notebook.
    A user interface cannot read a console, so it comes back as data too.

    ``value`` is whatever the generated code assigned to ``output``: a scalar, an
    array, a DataFrame, a figure. It is produced by executing that code locally,
    not by the model, so pydantic-ai cannot validate it and pydantic needs
    ``arbitrary_types_allowed`` to hold it. That also means ``model_dump_json``
    will not serialise ``value`` for types it does not know.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: Any = Field(
        default=None,
        description="The `output` variable, or None if every attempt failed.",
    )
    code: str = Field(default="", description="The code that produced `value`.")
    description: str = Field(
        default="", description="The model's own one-line summary of `output`."
    )
    judgment: Judgment | None = Field(
        default=None, description="Verdict on the final attempt, if one was reached."
    )
    chat_only: bool = Field(
        default=False,
        description="The model judged this a conversational fact, not a card.",
    )
    attempts: int = Field(default=0, description="Attempts made, successful or not.")
    errors: list[str] = Field(
        default_factory=list, description="One entry per failed attempt."
    )

    @property
    def ok(self) -> bool:
        """True when an attempt produced a value the judge accepted."""
        return self.judgment is not None and self.judgment.interprets_query_correctly

    def __repr__(self) -> str:
        # This is what a notebook cell displays, so lead with the answer on
        # success and with the reason on failure.
        if not self.ok:
            reason = self.judgment.reason if self.judgment else "no result produced"
            return f"ChatResult(failed after {self.attempts} attempts: {reason})"
        suffix = "" if self.attempts == 1 else f", attempts={self.attempts}"
        return f"ChatResult({self.value!r}{suffix})"


JUDGE_SYSTEM_PROMPT = (
    "You are an impartial reviewer. You classify whether a short NumPy snippet "
    "correctly interprets a natural-language query. Do NOT re-derive the "
    "numerical answer. Judge only whether the code addresses what was asked."
)


class NumpyCodeGen:
    """Generates NumPy code from natural-language queries using Pydantic AI.

    Parameters
    ----------
    model:
        Any model spec accepted by ``pydantic-ai`` - for example
        ``"google:gemini-2.5-flash"``, ``"anthropic:claude-sonnet-4-5"``,
        ``"openai:gpt-4o"``, or a pre-configured ``pydantic_ai.models.Model``
        instance. Defaults to Google Gemini 2.5 Flash.
    system_prompt:
        Optional override for the agent's system prompt.
    """

    def __init__(
        self,
        model: Any = DEFAULT_MODEL,
        *,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self._system_prompt = system_prompt or SYSTEM_PROMPT

    # Agents are built on first use rather than in __init__: constructing one
    # resolves the provider and requires credentials, and plenty of usage
    # (loading a file, inspecting metadata) never reaches the LLM at all.
    @cached_property
    def _code_agent(self) -> Agent[None, CodeResponse]:
        return Agent(
            model=self.model,
            output_type=CodeResponse,
            system_prompt=self._system_prompt,
        )

    @cached_property
    def _text_agent(self) -> Agent[None, str]:
        return Agent(model=self.model, system_prompt=self._system_prompt)

    @cached_property
    def _judge_agent(self) -> Agent[None, Judgment]:
        return Agent(
            model=self.model,
            output_type=Judgment,
            system_prompt=JUDGE_SYSTEM_PROMPT,
        )

    def generate_code(self, prompt: str) -> CodeResponse:
        """Run the code-generation agent and return structured output."""
        result = _run_coro(self._code_agent.run(prompt))
        return result.output

    def generate_text(self, prompt: str) -> str:
        """Run the free-form text agent (used by :class:`Diagnosis`)."""
        result = _run_coro(self._text_agent.run(prompt))
        return result.output

    def judge(
        self, query: str, code: str, metadata: str, context: str = ""
    ) -> Judgment:
        """Classify whether ``code`` correctly interprets ``query``.

        ``context`` carries recent conversation turns. Without it, a follow-up
        like "do the first one" is judged in a vacuum: the generator resolves
        the reference through history and the judge then rejects the correct
        code as unrelated to a "vague" query.
        """
        context_block = (
            f"CONVERSATION SO FAR (the query may refer back to this):\n"
            f"{context}\n\n"
            if context
            else ""
        )
        prompt = (
            f"{context_block}"
            f"USER QUERY:\n{query}\n\n"
            f"GENERATED CODE (defines `output`):\n{code}\n\n"
            f"AUTHOR'S DESCRIPTION OF `output`:\n{metadata}\n\n"
            "Question: does this code compute what the query asks for, read in "
            "the context of the conversation? Answer via the structured "
            "output. Do NOT recompute the answer."
        )
        return _run_coro(self._judge_agent.run(prompt)).output


def _column_block(summary: dict, var: str = "df") -> str:
    """Render a DataFrame's column summary for the prompt."""
    lines = []
    for name, info in summary.items():
        kind = info.get("kind", "")
        bits = []
        if kind == "numeric" and "min" in info:
            bits.append(f"{info['min']:g} to {info['max']:g}")
        elif kind == "datetime" and "min" in info:
            bits.append(f"{info['min']} to {info['max']}")
        elif kind == "boolean":
            bits.append(f"{info.get('true_count', '?')} true")
        elif kind == "text":
            if "categories" in info:
                bits.append("one of: " + ", ".join(repr(c) for c in info["categories"]))
            else:
                bits.append(f"{info.get('n_unique', '?')} distinct values")
        if info.get("nulls"):
            bits.append(f"{info['nulls']} null")
        detail = f"  ({', '.join(bits)})" if bits else ""
        lines.append(f"    {var}[{name!r}]  {info.get('dtype', '')}{detail}")
    return "\n".join(lines)


def prompt_single(
    query: str,
    metadata: dict,
    prior_feedback: str | None = None,
) -> str:
    feedback_block = (
        f"\nPREVIOUS ATTEMPT WAS REJECTED. Reason: {prior_feedback}\n"
        "Correct that specific issue this time.\n"
        if prior_feedback
        else ""
    )
    columns = metadata.get("columns")
    column_block = ""
    if columns:
        listing = "\n".join(f"    arr[:, {i}] -> {c}" for i, c in enumerate(columns))
        column_block = (
            "\n`arr` is a 2-D table whose columns are, in order:\n"
            f"{listing}\n"
            "Refer to columns by these positional indices. There are no named\n"
            "fields on `arr` - `arr['score']` will NOT work, use `arr[:, i]`.\n"
        )
    return f"""Generate NumPy code to perform the following operation:

{query}
{feedback_block}

CRITICAL INSTRUCTIONS:
1. The array is ALREADY defined as `arr`. DO NOT create a new array with `arr = ...`.
2. `np` (numpy) is pre-imported. You may import from numpy, scipy, sklearn
   and matplotlib when a bare name is cleaner (e.g.
   `from sklearn.linear_model import LinearRegression`). Import nothing else.
3. Prefer NumPy for everything. `scipy`, `sklearn` and `matplotlib.pyplot` (as
   `plt`) are available as a last resort - do not use them unless necessary.
   Reach for `scipy` only where NumPy genuinely lacks the routine, such as
   `scipy.stats` tests or `scipy.optimize` fitting.
4. Return ONLY code that operates on the existing `arr` variable.
5. There MUST be exactly one variable named `output` containing what the user asked for.
6. There MUST be exactly one variable named `metadata` - a short string describing `output`.
7. Ensure data is properly cleaned before executing any computation.

The array has these properties:
{metadata}
{column_block}
CORRECT EXAMPLES:
    # Replace NaN values with zero
    output = np.where(np.isnan(arr), 0, arr)
    metadata = "arr with NaNs replaced by 0"

    # Calculate mean of array
    output = np.mean(arr)
    metadata = "scalar: mean of arr"
"""


def prompt_multiple(
    query: str,
    context: dict,
    prior_feedback: str | None = None,
) -> str:
    """Build the code-generation prompt for a multi-input session."""
    blocks = []
    for name, info in context.items():
        md = info["metadata"]
        if info.get("kind") == "frame":
            blocks.append(
                f"`{name}` is a pandas DataFrame with {md.get('rows', '?')} rows:\n"
                + _column_block(md.get("column_summary", {})).replace("df[", f"{name}[")
            )
        else:
            bits = [
                f"shape {md['shape']}, dtype {md['element_type']}",
            ]
            if md.get("has_nan"):
                bits.append("contains NaN")
            if "min" in md and "max" in md:
                bits.append(f"range {md['min']:g} to {md['max']:g}")
            if md.get("columns"):
                bits.append("columns " + ", ".join(repr(c) for c in md["columns"]))
            blocks.append(f"`{name}` is a NumPy array: {'; '.join(bits)}")

    names = ", ".join(context.keys())
    has_frame = any(i.get("kind") == "frame" for i in context.values())
    feedback_block = (
        f"\nPREVIOUS ATTEMPT WAS REJECTED. Reason: {prior_feedback}\n"
        "Correct that specific issue this time.\n"
        if prior_feedback
        else ""
    )
    return f"""Generate Python code to perform the following operation:

{query}
{feedback_block}

{chr(10).join(blocks)}

CRITICAL INSTRUCTIONS:
1. These are ALREADY defined: {names}. DO NOT redefine or reload them, and do
   not invent names or columns that are not listed above.
2. `np` (numpy){", `pd` (pandas)" if has_frame else ""} and, where installed,
   `scipy`, `sklearn` and `plt` are pre-bound. You may also import from numpy,
   pandas, scipy, sklearn and matplotlib when a bare name is cleaner (e.g.
   `from sklearn.cluster import KMeans`). Import nothing else.
3. DO NOT mutate the inputs. Derive new values into `output` instead.
4. There MUST be exactly one variable named `output` containing the result.
5. There MUST be exactly one variable named `metadata` - a short string describing
   `output`.

CORRECT EXAMPLES:
    output = np.nanmean(arr1) - np.nanmean(arr2)
    metadata = "scalar: difference of the two means"

    output = np.where(np.isnan(arr2), np.nanmean(arr1), arr2)
    metadata = "arr2 with NaNs replaced by mean of arr1"
"""


def _history_block(history: list[tuple[str, str, str]] | None) -> str:
    """Render earlier turns so follow-ups like "explain it" have a referent."""
    if not history:
        return ""
    lines = []
    for question, code, description in history[-6:]:
        snippet = code if len(code) <= 400 else code[:400] + " ..."
        desc = description if len(description) <= 600 else description[:600] + " ..."
        lines.append(
            f"- asked: {question}\n  ran: {snippet or '(advice, no code)'}\n"
            f"  result: {desc or '(no description)'}"
        )
    return (
        "\nEARLIER IN THIS CONVERSATION (oldest first):\n"
        + "\n".join(lines)
        + "\n\nDo not simply reproduce an analysis listed above. Asked for\n"
        "something similar, extend it or take a genuinely different angle -\n"
        "a different column, statistic, or method.\n"
        "The user may refer back to these ('it', 'that chart', 'the tree').\n"
        "To explain or interpret an earlier result, re-run what you need from the\n"
        "code above and ground the explanation in the actual numbers - for example\n"
        "`sklearn.tree.export_text(model)` for a fitted tree, or the real group\n"
        "totals - then put the explanation in `output` as a plain string. Never\n"
        "explain from memory alone.\n"
    )


def _memories_block(memories: list[str] | None) -> str:
    """Render long-term memories as context that must not replace computation."""
    if not memories:
        return ""
    lines = "\n".join(f"- {m}" for m in memories[:6])
    return (
        "\nREMEMBERED FROM EARLIER SESSIONS (context - possibly stale):\n"
        f"{lines}\n"
        "Use these for preferences and pointers only. The data may have changed\n"
        "since they were written: recompute from `df` any number you state.\n"
    )


def prompt_frame(
    query: str,
    metadata: dict,
    prior_feedback: str | None = None,
    history: list[tuple[str, str, str]] | None = None,
    memories: list[str] | None = None,
) -> str:
    """Build the code-generation prompt for a DataFrame-backed query."""
    feedback_block = (
        f"\nPREVIOUS ATTEMPT WAS REJECTED. Reason: {prior_feedback}\n"
        "Correct that specific issue this time.\n"
        if prior_feedback
        else ""
    )
    history_block = _history_block(history)
    memories_block = _memories_block(memories)
    return f"""Generate Python code to perform the following operation:

{query}
{feedback_block}

`df` is a pandas DataFrame with {metadata.get("rows", "?")} rows. Its columns:

{_column_block(metadata.get("column_summary", {}))}
{history_block}{memories_block}

CRITICAL INSTRUCTIONS:
1. The DataFrame is ALREADY defined as `df`. DO NOT create or reload it, and do
   not invent columns that are not listed above.
2. `pd` (pandas), `np` (numpy) and, where installed, `scipy`, `sklearn` and
   `plt` are pre-bound. You may also import from numpy, pandas, scipy, sklearn
   and matplotlib when a bare name is cleaner (e.g.
   `from sklearn.linear_model import LinearRegression`). Import nothing else.
3. Use whichever fits the question. pandas for grouping, joining and resampling;
   NumPy for array maths. Both are fine, and mixing them is fine:
       output = df.groupby('region')['units'].sum()
       output = np.nansum(df['units'].to_numpy()[df['region'].to_numpy() == 'EMEA'])
4. `df[col].to_numpy()` on a text column gives dtype=object. `np.char` and
   `np.strings` need `.astype(str)` first; comparisons and `np.isin` do not.
5. DO NOT mutate `df`. Derive new values into `output` instead.
6. Prefer vectorised operations (groupby, boolean masks, column arithmetic) over
   Python loops over rows, groups or unique values.
   If the question asks HOW to analyse, what to explore, or for methodology
   rather than a computation, answer in the `advice` field instead of `code`:
   concrete suggestions naming the actual columns above, no code at all,
   and only angles not already covered in the conversation above.
   But when the user says to proceed - "do it", "do the first one", "all of
   that" - that refers to the conversation above: compute it now with `code`.
   For several related results, return a dict of labelled outputs. Never ask
   the user to pick one at a time.
   For a quick conversational fact - a row count, a yes/no, one looked-up
   value nobody would pin to a dashboard - set `chat_only` to true.
7. There MUST be exactly one variable named `output` containing what was asked for.
   If the user asked for a plot, `output` must be the Figure itself
   (`output = plt.gcf()`) - a plot drawn as a side effect is discarded.
8. There MUST be exactly one variable named `metadata` - a short string describing
   `output`.

CORRECT EXAMPLES:
    # Revenue by region
    output = (df['units'] * df['unit_price']).groupby(df['region']).sum()
    metadata = "Series: total revenue per region"

    # Rows after a date
    output = df[df['order_date'] > '2023-04-01']
    metadata = "DataFrame: orders placed after 2023-04-01"

    # A single number
    output = float(df['units'].mean())
    metadata = "scalar: mean units"
"""


def prompt_frames(
    query: str,
    tables: dict[str, dict],
    prior_feedback: str | None = None,
    history: list[tuple[str, str, str]] | None = None,
) -> str:
    """Build the prompt for a question spanning several DataFrames.

    ``tables`` maps variable name -> frame metadata. Each table is described
    the way :func:`prompt_frame` describes ``df``, under the name the generated
    code must use, so "@q1 vs @q2" questions reference `q1` and `q2` directly.
    """
    feedback_block = (
        f"\nPREVIOUS ATTEMPT WAS REJECTED. Reason: {prior_feedback}\n"
        "Correct that specific issue this time.\n"
        if prior_feedback
        else ""
    )
    names = list(tables)
    described = "\n\n".join(
        f"`{name}` is a pandas DataFrame with {md.get('rows', '?')} rows. "
        f"Its columns:\n\n{_column_block(md.get('column_summary', {}), var=name)}"
        for name, md in tables.items()
    )
    a, b = names[0], names[-1]
    return f"""Generate Python code to perform the following operation:

{query}
{feedback_block}

{described}
{_history_block(history)}

CRITICAL INSTRUCTIONS:
1. The DataFrames are ALREADY defined under the names above ({", ".join(names)}).
   DO NOT create or reload them, and do not invent columns not listed.
2. `pd` (pandas) and `np` (numpy) are pre-bound. You may also import from
   numpy, pandas, scipy, sklearn and matplotlib when a bare name is cleaner.
   Import nothing else.
3. Prefer vectorised operations over Python loops. To combine tables use
   merge/concat/set operations on the listed columns.
   If the question asks HOW to analyse or for methodology rather than a
   computation, answer in the `advice` field instead of `code`.
4. DO NOT mutate the DataFrames. Derive new values into `output`.
5. There MUST be exactly one variable named `output` containing what was asked
   for, and one named `metadata` - a short string describing `output`.

CORRECT EXAMPLES:
    # Difference in a total between two tables
    output = float({a}['units'].sum() - {b}['units'].sum())
    metadata = "scalar: difference in total units"

    # Rows in one table but not the other, by a key column
    output = {a}[~{a}['id'].isin({b}['id'])]
    metadata = "DataFrame: rows of {a} whose id is absent from {b}"
"""

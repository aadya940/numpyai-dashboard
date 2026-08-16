"""The generate, execute, judge, retry loop.

Every entry point shares this: a NumPy array, a DataFrame, or a multi-array
session. They differ only in what the prompt says and which variables the
generated code can see, so both are passed in and the loop itself is written
once.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ._ai import ChatResult, CodeResponse, Judgment, NumpyCodeGen
from ._exceptions import NumpyAIError
from ._utils import clean_code, optional_globals
from ._validator import NumpyValidator

console = Console()


def execute(
    code: str,
    data_vars: dict[str, Any],
    *,
    verbose: bool,
    raise_errors: bool = False,
) -> tuple[Any, Any]:
    """Run generated code and return ``(output, metadata)``.

    ``data_vars`` is what the code is allowed to see beyond NumPy and the
    optional libraries: ``{"arr": ...}``, ``{"df": ...}``, or one entry per
    array in a session.

    With ``raise_errors`` the exception propagates, which is what the retry
    loop wants: the message becomes feedback for the next attempt, and
    "'M' is not a valid frequency" is fixable where "returned None" is not.
    Without it, errors collapse to ``(None, None)``.
    """
    try:
        local_vars: dict[str, Any] = {"np": np, **optional_globals(), **data_vars}
        exec(code, {"__builtins__": __builtins__}, local_vars)
        result = local_vars.get("output")
        explainer = local_vars.get("metadata")
        result = _collect_stray_figures(local_vars.get("plt"), result)

        if verbose:
            if result is not None:
                console.print("\n".join(str(result).split("\n")[:10]))
            if explainer is not None:
                console.print(str(explainer))

        return result, explainer

    except Exception as e:
        if raise_errors:
            raise
        if verbose:
            console.print(f"[bold red]✗[/bold red] Error executing code: {e}")
        return None, None


def _collect_stray_figures(plt, result):
    """Deal with figures the generated code drew but never assigned.

    Models often plot as a side effect (``plt.plot(...)``) while assigning
    something else to ``output``. pyplot keeps a global reference to every such
    figure, so they leak in a long-running server; and if ``output`` was never
    set, the figure IS the answer. So: adopt the figure when there is nothing
    better, and close pyplot's registry either way.
    """
    if plt is None:
        return result
    try:
        fignums = plt.get_fignums()
        if not fignums:
            return result
        if result is None:
            result = plt.figure(fignums[-1])
        # Drop pyplot's global references. An adopted figure stays alive
        # through `result`; rendering does not need the registry.
        plt.close("all")
        return result
    except Exception:
        return result


def _print_judgment(j: Judgment) -> None:
    mark = "[green]✓[/green]" if j.interprets_query_correctly else "[red]✗[/red]"
    body = j.reason or "correctly interprets the query"
    console.print(Panel(f"{mark} {body}", title="Judgment", border_style="magenta"))


def _print_error_table(errors: list[str]) -> None:
    table = Table(title="Error Details", box=box.DOUBLE_EDGE)
    table.add_column("Attempt", style="cyan")
    table.add_column("Error", style="red")
    for i, msg in enumerate(errors, 1):
        table.add_row(str(i), msg)
    console.print(table)


def _generate(
    build_prompt: Callable[[str | None], str],
    prior_feedback: str | None,
    generator: NumpyCodeGen,
    validator: NumpyValidator,
    *,
    show: bool,
) -> CodeResponse:
    response = generator.generate_code(build_prompt(prior_feedback))
    response = CodeResponse(
        code=clean_code(response.code),
        advice=getattr(response, "advice", ""),
        explanation=response.explanation,
    )

    if response.advice and not response.code:
        return response

    if show:
        console.print(
            Panel(
                Syntax(response.code, "python", theme="monokai", line_numbers=True),
                title="[bold]Generated Code[/bold]",
                border_style="blue",
            )
        )

    if not validator.validate_code(response.code):
        raise NumpyAIError("Generated code failed syntax validation")
    return response


def run_chat(
    query: str,
    *,
    data_vars: dict[str, Any],
    build_prompt: Callable[[str | None], str],
    generator: NumpyCodeGen,
    validator: NumpyValidator,
    max_tries: int,
    verbose: bool,
    context: str = "",
) -> ChatResult:
    """Answer ``query``, retrying until the judge accepts or attempts run out.

    ``build_prompt`` takes the previous rejection reason, or None on the first
    attempt, so each retry can tell the model what went wrong last time.
    ``context`` is a compact rendering of recent turns, shown to the judge so
    follow-up queries are reviewed in the conversation they belong to.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    console.print(Panel(f"[bold cyan]Query:[/bold cyan] {query}", border_style="blue"))

    errors: list[str] = []
    prior_feedback: str | None = None
    last_judgment: Judgment | None = None

    for attempt in range(1, max_tries + 1):
        loud = verbose or attempt == max_tries
        if loud:
            console.print(f"[bold green]Attempt {attempt}/{max_tries}...[/bold green]")

        try:
            response = _generate(
                build_prompt, prior_feedback, generator, validator, show=loud
            )

            if getattr(response, "advice", "") and not response.code.strip():
                # An advisory answer: there is nothing to execute and nothing
                # for the judge, whose brief is whether code matches the query.
                console.print(
                    Panel(response.advice, title="Advice", border_style="cyan")
                )
                return ChatResult(
                    value=response.advice,
                    code="",
                    description="advisory answer",
                    judgment=Judgment(
                        interprets_query_correctly=True,
                        reason="advisory answer - nothing to execute",
                    ),
                    attempts=attempt,
                    errors=errors,
                )

            if loud:
                console.print("[bold]Executing generated code...[/bold]")
            result, explainer = execute(
                response.code, data_vars, verbose=verbose, raise_errors=True
            )

            if result is None:
                errors.append(f"Try {attempt}: Code execution returned None")
                prior_feedback = "code execution produced no `output` variable"
                if loud:
                    console.print(
                        f"[bold red]✗[/bold red] Attempt {attempt} failed: execution returned None"
                    )
                continue

            judgment = generator.judge(
                query=query,
                code=response.code,
                metadata=str(explainer or ""),
                context=context,
            )
            last_judgment = judgment
            if loud:
                _print_judgment(judgment)

            if judgment.interprets_query_correctly:
                console.print("[bold green]✓[/bold green] Judgment passed!")
                return ChatResult(
                    value=result,
                    code=response.code,
                    description=str(explainer or ""),
                    judgment=judgment,
                    attempts=attempt,
                    errors=errors,
                )

            prior_feedback = f"judgment rejected: {judgment.reason}"
            errors.append(f"Try {attempt}: {prior_feedback}")

        except Exception as e:
            errors.append(f"Try {attempt}: {e}")
            prior_feedback = f"exception in previous attempt: {e}"
            if loud:
                console.print(f"[bold red]✗[/bold red] Attempt {attempt} failed: {e}")

    # Failure is carried in the return value, but the table is still printed
    # because that is how a notebook user sees what went wrong.
    _print_error_table(errors)
    return ChatResult(judgment=last_judgment, attempts=max_tries, errors=errors)

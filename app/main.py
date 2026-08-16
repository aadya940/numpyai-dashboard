"""numpyai-dashboard Panel app.

Run with:

    panel serve app/main.py --show

Chat on the left; every answered question becomes a draggable block on the
right. Blocks remember the code that produced them, so the global filter bar
re-executes them against the filtered rows without another model call.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
from dotenv import load_dotenv

import numpyai_dashboard as npi
from numpyai_dashboard._ai import ChatResult
from numpyai_dashboard._engine import execute

pn.extension("echarts", "tabulator", notifications=True)

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / "examples" / ".env")

SAMPLE = REPO / "examples" / "sample_sales.xlsx"

PALETTE = ["#5B8FF9", "#5AD8A6", "#F6BD16", "#E8684A", "#6DC8EC", "#9270CA"]
CARD_CSS = {"border-radius": "10px", "box-shadow": "0 1px 4px rgba(0,0,0,.08)"}

_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
HAS_KEY = any(os.getenv(v) for v in _KEY_VARS)


# ---------------------------------------------------------------------------
# rendering a ChatResult value
# ---------------------------------------------------------------------------


def _echarts_spec(series: pd.Series) -> dict:
    """Bar for categories, line for anything time-shaped."""
    time_like = isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex))
    return {
        "color": PALETTE,
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 60, "right": 16, "top": 16, "bottom": 40},
        "xAxis": {
            "type": "category",
            "data": [str(i) for i in series.index],
            "axisLabel": {"rotate": 30 if len(series) > 6 else 0},
        },
        "yAxis": {"type": "value"},
        "series": [
            {
                "type": "line" if time_like else "bar",
                "data": [None if pd.isna(v) else round(float(v), 2) for v in series],
                "smooth": time_like,
                "areaStyle": {"opacity": 0.15} if time_like else None,
            }
        ],
    }


def render_value(value, chart: str | None = None):
    """Pick a pane for whatever the generated code produced."""
    if isinstance(value, pd.Series) and np.issubdtype(
        np.asarray(value.to_numpy()).dtype, np.number
    ):
        if len(value) > 30 or chart == "table":
            return pn.widgets.Tabulator(
                value.to_frame(), height=260, disabled=True, sizing_mode="stretch_width"
            )
        spec = _echarts_spec(value)
        if chart in ("bar", "line"):
            spec["series"][0]["type"] = chart
        return pn.pane.ECharts(
            spec, height=260, sizing_mode="stretch_width", theme="light"
        )

    if isinstance(value, pd.DataFrame):
        # Two numeric columns is an (x, y) relationship - frequency/amplitude,
        # size/price - and reads as a curve, not 400 rows of a table.
        numeric = value.select_dtypes("number")
        if (
            chart != "table"
            and value.shape[1] == 2
            and numeric.shape[1] == 2
            and 1 < len(value) <= 5000
        ):
            x, y = value.columns
            spec = {
                "color": PALETTE,
                "tooltip": {"trigger": "axis"},
                "grid": {"left": 60, "right": 16, "top": 16, "bottom": 40},
                "xAxis": {"type": "value", "name": str(x)},
                "yAxis": {"type": "value", "name": str(y)},
                "series": [
                    {
                        "type": "bar" if chart == "bar" else "line",
                        "showSymbol": len(value) <= 50,
                        "data": [
                            [float(a), float(b)]
                            for a, b in zip(value[x], value[y], strict=True)
                            if pd.notna(a) and pd.notna(b)
                        ],
                    }
                ],
            }
            return pn.pane.ECharts(
                spec, height=260, sizing_mode="stretch_width", theme="light"
            )
        return pn.widgets.Tabulator(
            value,
            pagination="remote" if len(value) > 10_000 else None,
            page_size=8,
            height=260,
            disabled=True,
            sizing_mode="stretch_width",
        )

    if isinstance(value, np.ndarray):
        if value.ndim <= 2 and value.size <= 400:
            return pn.widgets.Tabulator(
                pd.DataFrame(value),
                height=260,
                disabled=True,
                sizing_mode="stretch_width",
            )
        return pn.pane.Markdown(f"```\n{value!r}\n```")

    try:
        from matplotlib.figure import Figure

        if isinstance(value, Figure):
            return pn.pane.Matplotlib(value, tight=True, sizing_mode="stretch_width")
    except ImportError:
        pass

    if isinstance(value, (int, float, np.floating, np.integer)):
        return pn.indicators.Number(
            value=round(float(value), 2),
            format="{value:,}",
            font_size="42pt",
            sizing_mode="stretch_width",
        )

    return pn.pane.Markdown(f"**{value}**")


# ---------------------------------------------------------------------------
# a dashboard block
# ---------------------------------------------------------------------------


class Block:
    """One answered question: value on the front, code on the flip side."""

    def __init__(self, board: Board, result: ChatResult, question: str) -> None:
        self.board = board
        self.result = result
        self.question = question
        self.chart_choice = pn.widgets.Select(
            options=["auto", "bar", "line", "table"],
            value="auto",
            width=76,
            align="end",
        )
        self.chart_choice.param.watch(lambda _: self._redraw(), "value")

        verdict = "✓" if result.ok else "✗"
        color = "#1D9E75" if result.ok else "#E24B4A"
        tries = f" · {result.attempts} tries" if result.attempts > 1 else ""
        dismiss = pn.widgets.Button(name="✕", width=32, height=28, align="end")
        dismiss.on_click(lambda _: board.remove(self))

        self._body = pn.Column(sizing_mode="stretch_width")
        self._redraw()

        code_view = pn.Accordion(
            ("code", pn.pane.Markdown(f"```python\n{result.code}\n```")),
            active=[],
            sizing_mode="stretch_width",
        )
        self.panel = pn.Column(
            pn.Row(
                pn.pane.Markdown(
                    f"**{question}**  "
                    f"<span style='color:{color}'>{verdict}{tries}</span>",
                    margin=(0, 8),
                ),
                self.chart_choice,
                dismiss,
                sizing_mode="stretch_width",
            ),
            self._body,
            code_view,
            styles={"background": "#ffffff", "padding": "10px", **CARD_CSS},
            width=470,
            margin=8,
        )

    def _redraw(self) -> None:
        chart = None if self.chart_choice.value == "auto" else self.chart_choice.value
        self._body.objects = [render_value(self.result.value, chart)]

    def recompute(self, df: pd.DataFrame) -> None:
        """Re-run this block's stored code against (filtered) rows. No LLM."""
        value, _ = execute(self.result.code, {"df": df, "pd": pd}, verbose=False)
        if value is not None:
            self.result = self.result.model_copy(update={"value": value})
            self._redraw()


# ---------------------------------------------------------------------------
# the board
# ---------------------------------------------------------------------------


class Board:
    def __init__(self) -> None:
        self.frame = npi.read_excel(str(SAMPLE))
        self.blocks: list[Block] = []

        self.table = pn.widgets.Tabulator(
            self.frame.data,
            pagination="remote",
            page_size=10,
            height=300,
            disabled=True,
            sizing_mode="stretch_width",
        )
        self.grid_holder = pn.Column(sizing_mode="stretch_width")
        self.filters = pn.Row(sizing_mode="stretch_width")
        self.status = pn.pane.Markdown("", margin=(0, 8))
        self._build_filters()
        self._rebuild_grid()

        self.chat = pn.chat.ChatInterface(
            callback=self._on_ask,
            callback_user="numpyai",
            show_rerun=False,
            show_undo=False,
            sizing_mode="stretch_both",
        )
        self.chat.send(
            "Ask anything about the loaded table - every answer becomes a block "
            "on the right. Try *total revenue by region*.",
            user="numpyai",
            respond=False,
        )

        self.file_input = pn.widgets.FileInput(
            accept=".xlsx,.xls,.xlsb,.ods,.csv,.tsv", multiple=False
        )
        self.file_input.param.watch(self._on_upload, "value")

    # -- data ---------------------------------------------------------------

    def _on_upload(self, event) -> None:
        suffix = Path(self.file_input.filename or "upload.xlsx").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(event.new)
            path = fh.name
        reader = npi.read_csv if suffix in (".csv", ".tsv") else npi.read_excel
        self.frame = reader(path)
        self.blocks.clear()
        self.table.value = self.frame.data
        self._build_filters()
        self._rebuild_grid()
        pn.state.notifications.success(f"Loaded {self.file_input.filename}")

    def _build_filters(self) -> None:
        """One widget per low-cardinality text column, plus a date range."""
        self._filter_widgets = []
        widgets = []
        summary = self.frame.metadata["column_summary"]
        for name, info in summary.items():
            if info.get("kind") == "text" and "categories" in info:
                w = pn.widgets.MultiChoice(
                    name=name, options=info["categories"], width=190
                )
                self._filter_widgets.append((name, "cat", w))
                widgets.append(w)
            elif info.get("kind") == "datetime" and "min" in info:
                col = self.frame.data[name]
                w = pn.widgets.DateRangeSlider(
                    name=name,
                    start=col.min(),
                    end=col.max(),
                    value=(col.min(), col.max()),
                    width=260,
                )
                self._filter_widgets.append((name, "date", w))
                widgets.append(w)
            if len(widgets) >= 4:
                break
        for _, _, w in self._filter_widgets:
            w.param.watch(lambda _: self._on_filter(), "value")
        self.filters.objects = widgets

    def _filtered(self) -> pd.DataFrame:
        df = self.frame.data
        mask = pd.Series(True, index=df.index)
        for name, kind, w in self._filter_widgets:
            if kind == "cat" and w.value:
                mask &= df[name].isin(w.value)
            elif kind == "date":
                lo, hi = w.value
                mask &= (df[name] >= pd.Timestamp(lo)) & (df[name] <= pd.Timestamp(hi))
        return df[mask]

    def _on_filter(self) -> None:
        df = self._filtered()
        self.table.value = df
        for block in self.blocks:
            block.recompute(df)
        self.status.object = f"*{len(df):,} of {len(self.frame.data):,} rows*"

    # -- blocks ---------------------------------------------------------------

    def add_block(self, result: ChatResult, question: str) -> None:
        self.blocks.insert(0, Block(self, result, question))
        self._rebuild_grid()

    def remove(self, block: Block) -> None:
        self.blocks.remove(block)
        self._rebuild_grid()

    def _rebuild_grid(self) -> None:
        if not self.blocks:
            self.grid_holder.objects = [
                pn.pane.Markdown("*No blocks yet - answers pin here.*", margin=(20, 8))
            ]
            return
        # GridStack renders blank in testing (panel 1.9.3); FlexBox holds the
        # cards reliably. Drag/resize is a follow-up, not worth a broken board.
        self.grid_holder.height = None
        self.grid_holder.objects = [
            pn.FlexBox(*[b.panel for b in self.blocks], sizing_mode="stretch_width")
        ]

    # -- chat -----------------------------------------------------------------

    async def _ask(self, question: str) -> ChatResult:
        chatting = npi.frame(self._filtered(), max_tries=3)
        return await asyncio.to_thread(chatting.chat, question)

    async def _on_ask(self, contents: str, user: str, instance):
        if not HAS_KEY:
            return "No provider API key found - set one in `examples/.env`."
        result = await self._ask(contents)
        if result.ok:
            self.add_block(result, contents)
            answer = pn.Column(
                pn.pane.Markdown(f"**{result.description or 'Done.'}** - pinned →"),
                pn.Accordion(
                    ("code", pn.pane.Markdown(f"```python\n{result.code}\n```")),
                    active=[],
                ),
            )
            return answer
        reasons = "\n".join(f"- {e}" for e in result.errors[-3:])
        return f"Couldn't answer after {result.attempts} attempts:\n{reasons}"

    # -- starter dashboard ------------------------------------------------------

    def suggestions(self) -> list[str]:
        s = self.frame.metadata["column_summary"]
        nums = [n for n, i in s.items() if i.get("kind") == "numeric"]
        cats = [
            n for n, i in s.items() if i.get("kind") == "text" and "categories" in i
        ]
        dates = [n for n, i in s.items() if i.get("kind") == "datetime"]
        out = []
        if nums and cats:
            out.append(f"Total {nums[0]} by {cats[0]}.")
        if nums and dates:
            out.append(f"Monthly total {nums[0]} over {dates[0]}.")
        if nums:
            out.append(f"Mean {nums[-1]} overall.")
        return out[:3]

    async def autostart(self) -> None:
        if not HAS_KEY:
            self.status.object = "*No API key - autostart skipped.*"
            return
        for q in self.suggestions():
            # Mutations from a load task need pn.io.unlocked() to be pushed
            # over the websocket; without it they sit server-side, invisible.
            with pn.io.unlocked():
                self.status.object = f"*Building starter block: {q}*"
            result = await self._ask(q)
            if result.ok:
                with pn.io.unlocked():
                    self.add_block(result, q)
        with pn.io.unlocked():
            self.status.object = ""


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------

board = Board()

left = pn.Column(
    pn.pane.Markdown("## numpyai dashboard", margin=(4, 8)),
    board.file_input,
    board.chat,
    width=420,
    styles={"background": "#f4f4f6", "padding": "8px", **CARD_CSS},
    sizing_mode="stretch_height",
)

right = pn.Column(
    pn.Row(board.filters, board.status, sizing_mode="stretch_width"),
    board.grid_holder,
    pn.pane.Markdown("#### Data"),
    board.table,
    sizing_mode="stretch_both",
    margin=(0, 12),
    styles={"background": "#eef0f4", "padding": "12px", "border-radius": "12px"},
)

pn.config.raw_css.append("body { background: #e4e7ec; margin: 0; }")
app = pn.Row(left, right, sizing_mode="stretch_both", margin=8)
app.servable(title="numpyai dashboard")

pn.state.onload(board.autostart)

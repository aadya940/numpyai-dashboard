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
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
from dotenv import load_dotenv

import numpyai_dashboard as npi
from numpyai_dashboard._ai import ChatResult, NumpyCodeGen, prompt_frames
from numpyai_dashboard._engine import execute, run_chat
from numpyai_dashboard._validator import NumpyValidator

pn.extension("echarts", "tabulator", "filedropper", notifications=True)

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / "examples" / ".env")

SAMPLE = REPO / "examples" / "sample_sales.xlsx"

ACCENT = "#6366F1"
PALETTE = ["#6366F1", "#22C55E", "#F59E0B", "#EF4444", "#06B6D4", "#A855F7"]
CARD_CSS = {
    "border-radius": "12px",
    "box-shadow": "0 1px 2px rgba(15,23,42,.05)",
    "border": "1px solid #e5e7eb",
}
ACCORDION_CSS = """
.accordion { border: none !important; box-shadow: none !important;
  background: #f8fafc; border-radius: 8px; }
button.accordion-header { background: transparent; border: none;
  color: #6b7280; font-size: 12px; padding: 6px 10px; font-weight: 500; }
button.accordion-header:hover { color: #111827; }
"""

DROPPER_CSS = """
.filepond--root { margin-bottom: 0; }
.filepond--panel-root { background: #f8fafc; border: 1px dashed #d1d5db;
  border-radius: 10px; }
.filepond--drop-label, .filepond--drop-label label {
  color: #9ca3af; font-size: 12.5px; }
"""

TABLE_CSS = """
.tabulator { border: none; background: transparent; }
.tabulator .tabulator-header { border-bottom: 1px solid #e5e7eb;
  background: transparent; }
.tabulator .tabulator-header .tabulator-col { background: transparent; }
.tabulator-col-title { font-size: 12px; color: #6b7280; font-weight: 600; }
.tabulator-row { font-size: 12.5px; color: #1f2937; }
.tabulator-row.tabulator-row-even { background: #fafbfc; }
"""

AXIS = {
    "axisLine": {"show": False},
    "axisTick": {"show": False},
    "axisLabel": {"color": "#64748b", "fontSize": 11},
}
GRIDLINES = {"splitLine": {"lineStyle": {"type": "dashed", "color": "#e2e8f0"}}}

_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
HAS_KEY = any(os.getenv(v) for v in _KEY_VARS)


def slug(stem: str) -> str:
    """A filename stem as a Python identifier the model can reference."""
    out = re.sub(r"\W+", "_", stem).strip("_").lower() or "table"
    return out if not out[0].isdigit() else f"t_{out}"


def parse_mentions(text: str, names: list[str]) -> list[str]:
    """Resolve @tokens in ``text`` against loaded file names, fuzzily.

    "@q1" matches "q1_sales"; unknown tokens are ignored rather than an error,
    since "@ 3pm" in a question must not derail it.
    """
    found: list[str] = []
    for token in re.findall(r"@([\w.\-]+)", text):
        t = token.lower().rstrip(".")
        for name in names:
            if (name == t or name.startswith(t) or t in name) and name not in found:
                found.append(name)
                break
    return found


# ---------------------------------------------------------------------------
# rendering a ChatResult value
# ---------------------------------------------------------------------------


def _echarts_spec(series: pd.Series) -> dict:
    """Bar for categories, line for anything time-shaped."""
    time_like = isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex))
    fade = {
        "type": "linear",
        "x": 0,
        "y": 0,
        "x2": 0,
        "y2": 1,
        "colorStops": [
            {"offset": 0, "color": "rgba(99,102,241,.35)"},
            {"offset": 1, "color": "rgba(99,102,241,0)"},
        ],
    }
    return {
        "color": PALETTE,
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 56, "right": 12, "top": 14, "bottom": 42},
        "xAxis": {
            "type": "category",
            "data": [str(i) for i in series.index],
            **AXIS,
            "axisLabel": {
                **AXIS["axisLabel"],
                "rotate": 30 if len(series) > 6 else 0,
            },
        },
        "yAxis": {"type": "value", **AXIS, **GRIDLINES},
        "series": [
            {
                "type": "line" if time_like else "bar",
                "data": [None if pd.isna(v) else round(float(v), 2) for v in series],
                "smooth": time_like,
                "lineStyle": {"width": 3} if time_like else None,
                "itemStyle": (None if time_like else {"borderRadius": [6, 6, 0, 0]}),
                "areaStyle": {"color": fade} if time_like else None,
                "barMaxWidth": 46,
            }
        ],
    }


def render_value(value, chart: str | None = None, on_click=None):
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
        pane = pn.pane.ECharts(
            spec, height=260, sizing_mode="stretch_width", theme="light"
        )
        if on_click is not None:
            pane.on_event("click", on_click)
        return pane

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
                "grid": {"left": 56, "right": 16, "top": 14, "bottom": 42},
                "xAxis": {"type": "value", "name": str(x), **AXIS, **GRIDLINES},
                "yAxis": {"type": "value", "name": str(y), **AXIS, **GRIDLINES},
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
            show_index=False,
            sizing_mode="stretch_width",
            stylesheets=[TABLE_CSS],
        )

    if isinstance(value, dict):
        # A dict of results renders as labelled sections, never a repr blob.
        parts = []
        for key, item in list(value.items())[:8]:
            parts.append(
                pn.pane.Markdown(
                    f"<span style='font-weight:600;font-size:12px;"
                    f"color:#6b7280'>{key}</span>",
                    margin=(4, 8, 0, 8),
                )
            )
            parts.append(render_value(item, chart))
        return pn.Column(*parts, sizing_mode="stretch_width")

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
            font_size="38pt",
            default_color="#111827",
            sizing_mode="stretch_width",
            styles={"padding": "14px 8px"},
        )

    if isinstance(value, str):
        return pn.pane.Markdown(value if len(value) > 60 else f"**{value}**")
    return pn.pane.Markdown(f"**{value}**")


def phrase_value(value) -> str:
    """Say the answer in a sentence, the way a person would."""
    if isinstance(value, (int, float, np.floating, np.integer)):
        return f"It comes to **{float(value):,.2f}**."
    if (
        isinstance(value, pd.Series)
        and len(value)
        and pd.api.types.is_numeric_dtype(value)
    ):
        top = value.idxmax()
        return (
            f"**{top}** is highest at **{float(value.max()):,.2f}**, "
            f"across {len(value)} groups."
        )
    if isinstance(value, pd.DataFrame):
        return f"That gives a table of {len(value):,} rows x {value.shape[1]} columns."
    try:
        from matplotlib.figure import Figure

        if isinstance(value, Figure):
            return "Here it is as a figure."
    except ImportError:
        pass
    return ""


# ---------------------------------------------------------------------------
# a dashboard block
# ---------------------------------------------------------------------------


class Block:
    """One answered question: value on the front, code on the flip side."""

    def __init__(
        self, board: Board, result: ChatResult, question: str, number: int
    ) -> None:
        self.board = board
        self.result = result
        self.question = question
        self.number = number
        self.fresh = True
        self.source: str | None = None
        self.chart_choice = pn.widgets.Select(
            options=["auto", "bar", "line", "table"],
            value="auto",
            width=76,
            align="end",
        )
        self.chart_choice.param.watch(lambda _: self._redraw(), "value")

        ok = result.ok
        # A clean board does not announce success on every card; only failures
        # and retries earn a badge.
        if not ok:
            verdict_pill = (
                " <span style='background:#fee2e2;color:#b91c1c;padding:1px 8px;"
                "border-radius:999px;font-size:11px;font-weight:600'>failed</span>"
            )
        elif result.attempts > 1:
            verdict_pill = (
                f" <span style='color:#94a3b8;font-size:11px'>"
                f"{result.attempts} tries</span>"
            )
        else:
            verdict_pill = ""
        number_pill = (
            f"<span style='color:#94a3b8;font-size:12px;font-weight:600'>"
            f"#{number}</span>"
        )
        dismiss = pn.widgets.Button(
            name="✕", width=30, height=26, align="end", button_type="light"
        )
        dismiss.on_click(lambda _: board.remove(self))

        self._body = pn.Column(sizing_mode="stretch_width", min_height=260)
        self._redraw()

        code_view = pn.Accordion(
            ("code", pn.pane.Markdown(f"```python\n{result.code}\n```")),
            active=[],
            sizing_mode="stretch_width",
            stylesheets=[ACCORDION_CSS],
        )
        self.panel = pn.Column(
            pn.Row(
                pn.pane.Markdown(
                    f"{number_pill}&nbsp; <span style='font-weight:600;"
                    f"font-size:13.5px;color:#111827'>{question}</span>"
                    f"{verdict_pill}",
                    margin=(2, 10),
                ),
                self.chart_choice,
                dismiss,
                sizing_mode="stretch_width",
            ),
            self._body,
            code_view,
            styles={"background": "#ffffff", "padding": "12px 14px", **CARD_CSS},
            width=470,
            margin=(0, 12, 12, 0),
        )

    def _redraw(self) -> None:
        chart = None if self.chart_choice.value == "auto" else self.chart_choice.value
        self._body.objects = [
            render_value(self.result.value, chart, on_click=self._on_chart_click)
        ]

    def _on_chart_click(self, event) -> None:
        data = getattr(event, "data", None) or {}
        label = data.get("name")
        if label:
            self.board.drill(str(label), self)

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
        self.files: dict[str, npi.frame] = {
            slug(SAMPLE.stem): npi.read_excel(str(SAMPLE))
        }
        self.active: str = slug(SAMPLE.stem)
        self.blocks: list[Block] = []
        self._counter = 0
        #: shared across the per-question frames so "explain it" has a referent
        self._chat_history: list[tuple[str, str, str]] = []
        self.memories: dict[str, object] = {self.active: self._make_memory(self.active)}
        self.file_chips = pn.Row(margin=(0, 6))
        self._render_chips()
        self._texter = NumpyCodeGen(
            system_prompt=(
                "You are a sharp, plain-spoken data analyst reading results "
                "computed from the user's own data. Lead with the finding, "
                "quantify it, and flag what a careful analyst would flag - "
                "tiny leaf samples, a split on one salesperson's name, likely "
                "overfitting, confounders. Never open with a textbook "
                "definition; the user asked about THEIR data, not the method."
            )
        )

        self.table = pn.widgets.Tabulator(
            self.frame.data,
            pagination="remote",
            page_size=10,
            height=300,
            disabled=True,
            show_index=False,
            sizing_mode="stretch_width",
            stylesheets=[TABLE_CSS],
        )
        self.grid_holder = pn.Column(sizing_mode="stretch_width")
        self.board_caption = pn.pane.Markdown(margin=(10, 14, 0, 6))
        clear = pn.widgets.Button(
            name="Clear board",
            button_type="light",
            height=26,
            align="center",
            margin=(6, 6, 0, 0),
        )
        clear.on_click(lambda _e: self.clear_board())
        self.board_header = pn.Row(
            self.board_caption,
            pn.Spacer(sizing_mode="stretch_width"),
            clear,
            sizing_mode="stretch_width",
        )
        self.filters = pn.Row(sizing_mode="stretch_width")
        self.status = pn.pane.Markdown("", margin=(0, 8))
        self._build_filters()
        self._rebuild_grid()

        self.chat = pn.chat.ChatInterface(
            callback=self._on_ask,
            callback_user="numpyai",
            show_rerun=False,
            show_undo=False,
            show_button_name=False,
            sizing_mode="stretch_both",
            message_params={
                "show_reaction_icons": False,
                "show_copy_icon": False,
                "show_avatar": False,
                "show_timestamp": False,
            },
            widgets=[pn.chat.ChatAreaInput(placeholder="Ask about your data...")],
        )
        cols = ", ".join(f"`{c}`" for c in list(self.frame.data.columns)[:6])
        self.chat.send(
            f"Hi! I'm looking at **{SAMPLE.name}** - {len(self.frame.data):,} rows "
            f"with columns like {cols}.\n\n"
            "Ask me anything about it, in plain English. Each answer gets pinned "
            "to the board on the right, and the filters up top re-shape every "
            "block live."
            + (
                " I also remember our past sessions on this dataset."
                if self.memory
                else ""
            )
            + " I'll start with a few views to get you going.",
            user="numpyai",
            respond=False,
        )

        # FileDropper uploads in chunks. FileInput pushes the whole file
        # through one websocket message, and anything past Tornado's 20MB
        # cap kills the connection - which reads as "server connection lost"
        # the moment a real spreadsheet is dropped.
        self.file_input = pn.widgets.FileDropper(
            accepted_filetypes=[".xlsx", ".xls", ".xlsb", ".ods", ".csv", ".tsv"],
            multiple=True,
            max_file_size="500MB",
            height=76,
            sizing_mode="stretch_width",
            stylesheets=[DROPPER_CSS],
        )
        self.file_input.param.watch(self._on_upload, "value")

        self.forget_button = pn.widgets.Button(
            name="Forget memory",
            button_type="light",
            width=118,
            height=30,
            align="center",
            visible=self.memory is not None,
        )
        self.forget_button.on_click(self._on_forget)

    @property
    def frame(self):
        return self.files[self.active]

    @property
    def memory(self):
        return self.memories.get(self.active)

    def _render_chips(self) -> None:
        chips = []
        removable = len(self.files) > 1
        for name in self.files:
            is_active = name == self.active
            b = pn.widgets.Button(
                name=f"@{name}",
                button_type="primary" if is_active else "light",
                height=26,
                margin=(2, 0, 2, 3),
            )
            b.on_click(lambda _e, n=name: self.switch(n))
            chips.append(b)
            if removable:
                x = pn.widgets.Button(
                    name="✕",
                    button_type="light",
                    width=24,
                    height=26,
                    margin=(2, 3, 2, 0),
                )
                x.on_click(lambda _e, n=name: self.remove_file(n))
                chips.append(x)
        self.file_chips.objects = chips

    def remove_file(self, name: str) -> None:
        """Drop a file from the session, and the blocks that came from it.

        Long-term memories are kept: removing a file from view is not the same
        as forgetting what was learned from it. The last file cannot go.
        """
        if name not in self.files or len(self.files) == 1:
            return
        del self.files[name]
        orphaned = [b for b in self.blocks if b.source == name]
        for block in orphaned:
            self.blocks.remove(block)
        if self.active == name:
            self.active = next(iter(self.files))
            self._build_filters()
            self.table.value = self.frame.data
        self._render_chips()
        self._rebuild_grid()
        self.chat.send(
            f"Removed **@{name}**"
            + (f" and its {len(orphaned)} block(s)" if orphaned else "")
            + f". Active file is **@{self.active}**.",
            user="numpyai",
            respond=False,
        )

    def switch(self, name: str) -> None:
        """Make ``name`` the active file: filters, table and memory follow."""
        if name not in self.files or name == self.active:
            return
        self.active = name
        self.memories.setdefault(name, self._make_memory(name))
        self._build_filters()
        self.table.value = self.frame.data
        self._render_chips()
        self.status.object = f"*Active: @{name}*"

    @staticmethod
    def _make_memory(dataset: str):
        """Long-term memory scoped to this dataset; optional, never fatal."""
        if not HAS_KEY:
            return None
        try:
            return npi.AgentMemory(user_id=dataset)
        except Exception:
            return None

    def _on_forget(self, _event) -> None:
        if self.memory is None:
            return
        count = self.memory.forget()
        self._chat_history.clear()
        pn.state.notifications.info(f"Forgot {count} memories for this dataset.")
        self.chat.send(
            f"Done - I've forgotten everything I knew about this dataset "
            f"({count} memories) and this conversation. Clean slate.",
            user="numpyai",
            respond=False,
        )

    # -- data ---------------------------------------------------------------

    async def _on_upload(self, event) -> None:
        for filename, payload in (event.new or {}).items():
            suffix = Path(filename).suffix or ".xlsx"
            # FileDropper hands text files over as str and binary as bytes.
            data = payload.encode() if isinstance(payload, str) else payload
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
                fh.write(data)
                path = fh.name
            reader = npi.read_csv if suffix in (".csv", ".tsv") else npi.read_excel
            name = slug(Path(filename).stem)
            while name in self.files:
                name += "_2"
            self.status.object = f"*Reading {filename}...*"
            try:
                # Parsing a large file on the event loop blocks the websocket
                # heartbeat until Tornado gives up on the session - the reader
                # must run in a thread, however fast it usually is.
                self.files[name] = await asyncio.to_thread(reader, path)
            except Exception as exc:
                self.chat.send(
                    f"Couldn't read **{filename}**: {exc}",
                    user="numpyai",
                    respond=False,
                )
                continue
            self.active = name
            self.memories.setdefault(
                name, await asyncio.to_thread(self._make_memory, name)
            )
            cols = ", ".join(f"`{c}`" for c in list(self.frame.data.columns)[:6])
            self.chat.send(
                f"Loaded **{filename}** as **@{name}** - "
                f"{len(self.frame.data):,} rows with {cols}. It's now the "
                "active file; mention any file with @ to switch or compare.",
                user="numpyai",
                respond=False,
            )
        self.forget_button.visible = self.memory is not None
        self._render_chips()
        self.table.value = self.frame.data
        self._build_filters()
        self._rebuild_grid()
        self.status.object = ""

    def _build_filters(self) -> None:
        """One widget per low-cardinality text column, plus a date range."""
        self._filter_widgets = []
        widgets = []
        summary = self.frame.metadata["column_summary"]
        for name, info in summary.items():
            if info.get("kind") == "text" and "categories" in info:
                w = pn.widgets.MultiChoice(
                    name=name, options=info["categories"], width=168, margin=(0, 4)
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
                    width=230,
                    margin=(0, 8),
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
            # Filters act on the active file, so only its blocks recompute;
            # a block pinned from another file would silently show the wrong
            # rows if fed this frame.
            if block.source == self.active:
                block.recompute(df)
        self.status.object = f"*{len(df):,} of {len(self.frame.data):,} rows*"

    def drill(self, label: str, block: Block) -> None:
        """A clicked chart element becomes a board-wide filter, or a question.

        Priority: a category filter that knows this label filters the whole
        board instantly and reversibly. A month-shaped label narrows the date
        range. Anything else drafts a drill question into the chat input, so
        the model call stays under the user's control.
        """
        for name, kind, w in self._filter_widgets:
            if kind == "cat" and label in w.options:
                w.value = [] if w.value == [label] else [label]  # click twice to clear
                return
        month = re.match(r"^(\d{4})-(\d{2})", label)
        if month:
            for name, kind, w in self._filter_widgets:
                if kind == "date":
                    start = pd.Timestamp(int(month.group(1)), int(month.group(2)), 1)
                    end = start + pd.offsets.MonthEnd(1)
                    lo, hi = w.start, w.end
                    w.value = (max(pd.Timestamp(lo), start), min(pd.Timestamp(hi), end))
                    return
        widget = self.chat.active_widget
        if widget is not None:
            # value_input prefills without submitting; .value would auto-send
            # and spend a model call the user never asked for.
            widget.value_input = f"Drill into {label}: {block.question}"

    # -- blocks ---------------------------------------------------------------

    def add_block(
        self, result: ChatResult, question: str, source: str | None = "active"
    ) -> Block:
        self._counter += 1
        block = Block(self, result, question, self._counter)
        block.source = self.active if source == "active" else source
        self.blocks.insert(0, block)
        self._rebuild_grid()
        return block

    def remove(self, block: Block) -> None:
        self.blocks.remove(block)
        self._rebuild_grid()

    def clear_board(self) -> None:
        self.blocks.clear()
        self._rebuild_grid()

    def _rebuild_grid(self) -> None:
        n = len(self.blocks)
        self.board_caption.object = (
            f"<span style='font-weight:600;font-size:13.5px;color:#111827'>"
            f"Board</span> <span style='color:#9ca3af;font-size:12px'>"
            f"{n} block{'s' if n != 1 else ''} - click a bar or point to "
            f"filter</span>"
        )
        if not self.blocks:
            self.grid_holder.objects = [
                pn.pane.Markdown(
                    "Nothing pinned yet - ask something and it lands here.",
                    styles={
                        "color": "#9ca3af",
                        "text-align": "center",
                        "font-size": "13px",
                        "padding": "28px 0",
                        "border": "1px dashed #d1d5db",
                        "border-radius": "12px",
                    },
                    sizing_mode="stretch_width",
                )
            ]
            return
        # GridStack renders blank in testing (panel 1.9.3); FlexBox holds the
        # cards reliably. Drag/resize is a follow-up, not worth a broken board.
        for b in self.blocks:
            b.panel.css_classes = (
                ["npi-card", "fresh-card"] if b.fresh else ["npi-card"]
            )
            b.fresh = False
        self.grid_holder.height = None
        self.grid_holder.objects = [
            pn.FlexBox(*[b.panel for b in self.blocks], sizing_mode="stretch_width")
        ]

    # -- chat -----------------------------------------------------------------

    async def _ask(self, question: str) -> ChatResult:
        mentioned = parse_mentions(question, list(self.files))
        self._last_was_multi = len(mentioned) >= 2
        if self._last_was_multi:
            return await asyncio.to_thread(self._multi_chat, question, mentioned)
        if len(mentioned) == 1:
            self.switch(mentioned[0])
        chatting = npi.frame(self._filtered(), max_tries=3)
        chatting.history = self._chat_history  # shared, mutated in place
        chatting.memory = self.memory
        return await asyncio.to_thread(chatting.chat, question)

    def _multi_chat(self, question: str, names: list[str]) -> ChatResult:
        """A question spanning several files: each mentioned frame by name."""
        # "@q1" in the text becomes "q1" so it matches the variable names.
        query = re.sub(r"@([\w.\-]+)", lambda m: m.group(1).lower(), question)
        tables = {n: self.files[n].metadata for n in names}
        result = run_chat(
            query,
            data_vars={n: self.files[n].data for n in names},
            build_prompt=lambda prior: prompt_frames(
                query=query,
                tables=tables,
                prior_feedback=prior,
                history=self._chat_history,
            ),
            generator=self._texter,
            validator=NumpyValidator(),
            max_tries=3,
            verbose=False,
        )
        if result.ok:
            self._chat_history.append((query, result.code, result.description))
            del self._chat_history[:-8]
        return result

    async def _on_ask(self, contents: str, user: str, instance):
        if not HAS_KEY:
            return "No provider API key found - set one in `examples/.env`."
        result = await self._ask(contents)
        if result.ok and isinstance(result.value, str) and not result.code:
            # Advisory answer: already prose, already the answer. No pin, no
            # narration pass - narrating advice would just paraphrase it.
            return pn.pane.Markdown(result.value)
        if result.ok and isinstance(result.value, str):
            # The code step produced evidence (tree rules, group numbers). The
            # code persona cannot narrate values it has not seen, so a second,
            # text-only step turns evidence into an answer - grounded, because
            # the evidence is in its context and it is told to stay inside it.
            narrated = await asyncio.to_thread(
                self._texter.generate_text,
                f"The user asked: {contents}\n\n"
                "Evidence computed directly from their data:\n"
                f"{result.value}\n\n"
                "Answer as the analyst, grounded strictly in this evidence:\n"
                "1. The headline finding, one sentence.\n"
                "2. What drives it, with the actual numbers.\n"
                "3. Caveats a careful analyst would raise about THIS result.\n"
                "3-6 sentences of markdown, no headings for the answer itself. "
                "Do not invent numbers and do not define standard methods. "
                "Then include the evidence verbatim under 'Details:'.",
            )
            return pn.Column(
                pn.pane.Markdown(narrated),
                pn.Accordion(
                    (
                        "how I computed it",
                        pn.pane.Markdown(f"```python\n{result.code}\n```"),
                    ),
                    active=[],
                    stylesheets=[ACCORDION_CSS],
                ),
            )
        if result.ok:
            was_multi = getattr(self, "_last_was_multi", False)
            block = self.add_block(
                result, contents, source=None if was_multi else "active"
            )
            if was_multi and isinstance(result.value, pd.DataFrame):
                # "Join these files" should yield a file, not only a card.
                name = "merged"
                while name in self.files:
                    name += "_2"
                self.files[name] = npi.frame(result.value)
                self.memories.setdefault(name, self._make_memory(name))
                self._render_chips()
                self.chat.send(
                    f"I've also registered the result as **@{name}** "
                    f"({len(result.value):,} rows) - you can chat with it, "
                    "filter it, or compare it like any other file.",
                    user="numpyai",
                    respond=False,
                )
            retried = (
                f" It took {result.attempts} attempts - the first tries were "
                "rejected before one passed review."
                if result.attempts > 1
                else ""
            )
            text = (
                f"{phrase_value(result.value)}{retried}\n\n"
                f"I've pinned it as **#{block.number}** on the board, top left - "
                "it will follow the filters."
            )
            return pn.Column(
                pn.pane.Markdown(text),
                pn.Accordion(
                    (
                        "how I computed it",
                        pn.pane.Markdown(f"```python\n{result.code}\n```"),
                    ),
                    active=[],
                    stylesheets=[ACCORDION_CSS],
                ),
            )
        reasons = "\n".join(f"- {e}" for e in result.errors[-3:])
        return (
            f"I couldn't get an answer that passed review after "
            f"{result.attempts} attempts. Here's what went wrong:\n{reasons}\n\n"
            "Try naming a column from the table below, or rephrasing."
        )

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
                self.status.object = f"*Working on: {q}*"
            result = await self._ask(q)
            if result.ok:
                with pn.io.unlocked():
                    block = self.add_block(result, q)
                    self.chat.send(
                        f"**#{block.number}** *{q}* - {phrase_value(result.value)}",
                        user="numpyai",
                        respond=False,
                    )
        with pn.io.unlocked():
            self.status.object = ""
            self.chat.send(
                "That's a starting point - what would you like to dig into?",
                user="numpyai",
                respond=False,
            )


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------

board = Board()

left = pn.Column(
    pn.pane.Markdown(
        f"<span style='display:inline-block;width:9px;height:9px;"
        f"border-radius:50%;background:{ACCENT};margin-right:7px'></span>"
        "<span style='color:#111827;font-size:17px;font-weight:700'>"
        "numpyai</span> <span style='color:#9ca3af;font-size:17px'>"
        "dashboard</span>",
        margin=(10, 14, 0, 14),
    ),
    pn.pane.Markdown(
        "<span style='color:#9ca3af;font-size:12px'>chat with your "
        "spreadsheets - @mention a file to switch or compare</span>",
        margin=(0, 14, 6, 14),
    ),
    board.file_chips,
    board.chat,
    pn.Row(board.file_input, board.forget_button, margin=(2, 8, 6, 8)),
    width=430,
    styles={"background": "#ffffff", "padding": "6px", **CARD_CSS},
    sizing_mode="stretch_height",
)

filter_bar = pn.Row(
    pn.pane.Markdown(
        "<span style='color:#9ca3af;font-size:11px;font-weight:600;"
        "letter-spacing:.08em'>FILTERS</span>",
        margin=(18, 2, 0, 14),
        align="start",
    ),
    board.filters,
    pn.Spacer(sizing_mode="stretch_width"),
    board.status,
    sizing_mode="stretch_width",
    styles={"background": "#ffffff", "padding": "2px 8px 8px 8px", **CARD_CSS},
)

data_card = pn.Column(
    pn.pane.Markdown(
        "<span style='font-weight:600;font-size:13.5px;color:#111827'>" "Data</span>",
        margin=(8, 14, 0, 14),
    ),
    board.table,
    sizing_mode="stretch_width",
    styles={"background": "#ffffff", "padding": "6px", **CARD_CSS},
)

right = pn.Column(
    filter_bar,
    board.board_header,
    board.grid_holder,
    data_card,
    sizing_mode="stretch_both",
    margin=(0, 4, 0, 14),
)

pn.config.raw_css.append("""
body {
  background: #f6f7f9;
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
    'Helvetica Neue', Arial, sans-serif;
  color: #111827;
}
:root { --design-primary-color: #6366F1; }
.npi-card { transition: box-shadow .15s ease; }
.npi-card:hover { box-shadow: 0 4px 12px rgba(15,23,42,.08) !important; }
@keyframes cardflash {
  0%   { box-shadow: 0 0 0 2px #6366F1; }
  100% { box-shadow: 0 1px 2px rgba(15,23,42,.05); }
}
.fresh-card { animation: cardflash 2s ease-out; }
""")
app = pn.Row(left, right, sizing_mode="stretch_both", margin=12)
app.servable(title="numpyai dashboard")

pn.state.onload(board.autostart)

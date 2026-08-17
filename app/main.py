"""numpyai-dashboard Panel app.

Run with:

    panel serve app/main.py --show

Chat on the left; every answered question becomes a draggable block on the
right. Blocks remember the code that produced them, so the global filter bar
re-executes them against the filtered rows without another model call.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import numpyai_dashboard as npi
from numpyai_dashboard._ai import (
    DEFAULT_MODEL,
    ChatResult,
    NumpyCodeGen,
    _run_coro,
    prompt_frames,
)
from numpyai_dashboard._engine import execute, run_chat
from numpyai_dashboard._validator import NumpyValidator

pn.config.css_files.append(
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
)
pn.config.js_files.update(
    {
        "katex": "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js",
        "sortable": "https://cdn.jsdelivr.net/npm/sortablejs@1.15.2/Sortable.min.js",
        "npi_board": "/assets/board.js",
    }
)
pn.extension("echarts", "tabulator", "filedropper", notifications=True)

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / "examples" / ".env")

SAMPLE = REPO / "examples" / "sample_sales.xlsx"

ACCENT = "#7C9AFF"
PALETTE = ["#7C9AFF", "#4FD1C5", "#F6BD6B", "#F97583", "#6EDDF6", "#C39BF7"]
PEAK = "#F59E0B"
CARD_CSS = {
    "border-radius": "14px",
    "box-shadow": "0 1px 2px rgba(15,23,42,.04), 0 4px 14px rgba(15,23,42,.05)",
    "border": "1px solid rgba(255,255,255,.08)",
}
ACCORDION_CSS = """
.accordion {
  background: #0f1830 !important;
  border: 1px solid rgba(255,255,255,.07) !important;
  box-shadow: none !important; border-radius: 9px;
}
button.accordion-header {
  background: transparent !important; border: none !important;
  color: #93a4c8 !important; font-size: 12px !important;
  padding: 7px 12px !important; font-weight: 500;
}
button.accordion-header:hover { color: #eef2fa !important; }
.accordion .card-button { color: #93a4c8 !important; }
"""

DROPPER_CSS = """
.filepond--root { margin-bottom: 0; }
.filepond--panel-root { background: #182339; border: 1px dashed #d1d5db;
  border-radius: 10px; }
.filepond--drop-label, .filepond--drop-label label {
  color: #9ca3af; font-size: 12.5px; }
"""

TABLE_CSS = """
.tabulator, .tabulator-tableholder, .tabulator-table {
  background: transparent !important; }
.tabulator .tabulator-header {
  background: transparent !important;
  border-bottom: 1px solid rgba(255,255,255,.10) !important; }
.tabulator .tabulator-header .tabulator-col {
  background: transparent !important; border-color: transparent !important; }
.tabulator-col-title {
  font-size: 12px; color: #9aa8c6 !important; font-weight: 600; }
.tabulator-row {
  font-size: 12.5px; color: #dbe3f1 !important;
  background: transparent !important;
  border-color: rgba(255,255,255,.06) !important; }
.tabulator-row .tabulator-cell { border-color: rgba(255,255,255,.06) !important; }
.tabulator-row.tabulator-row-even { background: #0f1830 !important; }
.tabulator-row.tabulator-selectable:hover { background: #1a2542 !important; }
.tabulator-footer {
  background: transparent !important; color: #9aa8c6 !important;
  border-top: 1px solid rgba(255,255,255,.10) !important; }
.tabulator-footer .tabulator-page {
  background: #16203a !important; color: #c3cddf !important;
  border: 1px solid rgba(255,255,255,.10) !important; }
.tabulator-footer .tabulator-page.active { color: #aab8ff !important; }
.tabulator-loader { background: transparent !important; }
"""

AXIS = {
    "axisLine": {"show": False},
    "axisTick": {"show": False},
    "axisLabel": {"color": "#8ea0bf", "fontSize": 11},
}
GRIDLINES = {
    "splitLine": {"lineStyle": {"type": "dashed", "color": "rgba(255,255,255,.08)"}}
}

_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
HAS_KEY = any(os.getenv(v) for v in _KEY_VARS)


class _Starters(BaseModel):
    questions: list[str] = Field(
        description=(
            "Exactly three analysis questions for this dataset, each a "
            "different KIND of analysis - e.g. one headline metric, one "
            "relationship between columns, one distribution, segment "
            "comparison or anomaly check. Never three groupby-sums."
        )
    )


class _BlockTakeaway(BaseModel):
    number: int = Field(description="The block number this takeaway belongs to.")
    takeaway: str = Field(
        description=(
            "One sentence, under 120 characters: what a human should conclude "
            "from this block alone, with its key number."
        )
    )


class _BoardStory(BaseModel):
    story: str = Field(
        description=(
            "The story the whole board tells: 3-6 sentences of markdown, "
            "headline number first, evidence cited as #n, one caveat last."
        )
    )
    takeaways: list[_BlockTakeaway]


_TEX_RE = re.compile(r"\$\$(.+?)\$\$|(?<!\\)\$(.+?)(?<!\\)\$", re.DOTALL)


def _looks_like_tex(tex: str) -> bool:
    """Distinguish $x_i^2$ from $333,978.35 ... $0.14 - money is not math.

    A model narrating revenue writes dollar signs constantly; pairing them
    naively turned a whole sentence into "math" and KaTeX rendered it as red
    error text. Real TeX either carries TeX syntax or is one short token.
    """
    if len(tex) > 160:
        return False
    if re.search(r"[\\^_{}=]", tex):
        return True
    return " " not in tex and not re.fullmatch(r"[\d.,%+\-]+", tex)


def texify(text: str) -> str:
    """Protect $...$ math from markdown, which mangles its underscores.

    Each math span becomes an inert placeholder carrying the raw TeX
    base64-encoded; markdown passes it through untouched and the client
    renders each one with katex.render - no delimiter scanning, so no
    half-rendered red wreckage when markdown has already eaten a token.
    Spans that read as currency or prose stay exactly as written.
    """
    if "$" not in text:
        return text

    def _swap(match: re.Match) -> str:
        display = match.group(1) is not None
        tex = match.group(1) if display else match.group(2)
        if not display and not _looks_like_tex(tex):
            return match.group(0)
        blob = base64.b64encode(tex.encode()).decode()
        return (
            f'<span class="npi-tex" data-tex="{blob}"'
            f' data-display="{int(display)}"></span>'
        )

    return _TEX_RE.sub(_swap, text)


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
            {"offset": 0, "color": "rgba(124,154,255,.35)"},
            {"offset": 1, "color": "rgba(124,154,255,0)"},
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
                "markPoint": {
                    "data": [{"type": "max", "name": "peak"}],
                    "symbolSize": 44,
                    "label": {"fontSize": 10, "color": "#3b2f00", "fontWeight": 600},
                    "itemStyle": {"color": PEAK},
                },
                "markLine": (
                    {
                        "data": [{"type": "average", "name": "avg"}],
                        "lineStyle": {"type": "dashed", "color": "#9ca3af"},
                        "label": {"fontSize": 10, "color": "#9ca3af"},
                        "symbol": "none",
                    }
                    if time_like
                    else None
                ),
            }
        ],
    }


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """A copy formatted for humans: datetimes as dates, not epoch millis."""
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        elif isinstance(out[col].dtype, pd.PeriodDtype):
            out[col] = out[col].astype(str)
    if isinstance(out.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        out.index = out.index.astype(str)
    return out


def render_value(value, chart: str | None = None, on_click=None, height: int = 260):
    """Pick a pane for whatever the generated code produced."""
    if isinstance(value, pd.Series) and np.issubdtype(
        np.asarray(value.to_numpy()).dtype, np.number
    ):
        if chart == "donut" and len(value) <= 8:
            return pn.pane.ECharts(
                {
                    "color": PALETTE,
                    "tooltip": {"trigger": "item"},
                    "legend": {
                        "orient": "vertical",
                        "right": 6,
                        "top": "center",
                        "textStyle": {"color": "#9db0d6", "fontSize": 11},
                    },
                    "series": [
                        {
                            "type": "pie",
                            "radius": ["52%", "78%"],
                            "center": ["38%", "50%"],
                            "label": {"show": False},
                            "itemStyle": {
                                "borderColor": "#121a2e",
                                "borderWidth": 2,
                                "borderRadius": 4,
                            },
                            "data": [
                                {"name": str(k), "value": round(float(v), 2)}
                                for k, v in value.items()
                                if pd.notna(v)
                            ],
                        }
                    ],
                },
                height=height,
                sizing_mode="stretch_width",
                theme="light",
            )
        if len(value) > 30 or chart == "table":
            return pn.widgets.Tabulator(
                _display_frame(value.to_frame()),
                height=height,
                disabled=True,
                sizing_mode="stretch_width",
                stylesheets=[TABLE_CSS],
            )
        spec = _echarts_spec(value)
        if chart in ("bar", "line"):
            spec["series"][0]["type"] = chart
        if chart == "hbar":
            spec["series"][0]["type"] = "bar"
            spec["xAxis"], spec["yAxis"] = spec["yAxis"], spec["xAxis"]
            spec["series"][0]["itemStyle"] = {"borderRadius": [0, 6, 6, 0]}
            spec["series"][0].pop("markPoint", None)
        pane = pn.pane.ECharts(
            spec, height=height, sizing_mode="stretch_width", theme="light"
        )
        if on_click is not None:
            pane.on_event("click", on_click)
        return pane

    if isinstance(value, pd.DataFrame):
        # A (datetime, numeric) pair is a trend: draw the line, don't tabulate
        # epoch milliseconds.
        if (
            chart in (None, "line")
            and value.shape[1] == 2
            and pd.api.types.is_datetime64_any_dtype(value.iloc[:, 0])
            and pd.api.types.is_numeric_dtype(value.iloc[:, 1])
            and 1 < len(value) <= 500
        ):
            series = value.set_index(value.columns[0])[value.columns[1]]
            return pn.pane.ECharts(
                _echarts_spec(series),
                height=height,
                sizing_mode="stretch_width",
                theme="light",
            )
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
            xs = value[x].to_numpy()
            monotonic = bool(np.all(np.diff(xs[~np.isnan(xs)]) >= 0))
            kind = "scatter" if (chart == "scatter" or not monotonic) else "line"
            if chart in ("bar", "line"):
                kind = chart
            spec = {
                "color": PALETTE,
                "tooltip": {"trigger": "axis"},
                "grid": {"left": 56, "right": 16, "top": 14, "bottom": 42},
                "xAxis": {"type": "value", "name": str(x), **AXIS, **GRIDLINES},
                "yAxis": {"type": "value", "name": str(y), **AXIS, **GRIDLINES},
                "series": [
                    {
                        "type": kind,
                        "symbolSize": 7 if kind == "scatter" else 4,
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
                spec, height=height, sizing_mode="stretch_width", theme="light"
            )
        return pn.widgets.Tabulator(
            _display_frame(value),
            pagination="remote" if len(value) > 10_000 else None,
            page_size=8,
            height=height,
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
                    f"color:#9aa8c6'>{key}</span>",
                    margin=(4, 8, 0, 8),
                )
            )
            parts.append(render_value(item, chart))
        return pn.Column(*parts, sizing_mode="stretch_width")

    if isinstance(value, np.ndarray):
        if value.ndim <= 2 and value.size <= 400:
            return pn.widgets.Tabulator(
                pd.DataFrame(value),
                height=height,
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
        big = abs(float(value)) >= 1000
        return pn.indicators.Number(
            value=round(float(value)) if big else round(float(value), 2),
            format="{value:,}",
            font_size="38pt",
            default_color="#eef2fa",
            sizing_mode="stretch_width",
            styles={"padding": "14px 8px"},
        )

    if isinstance(value, str):
        text = texify(value)
        if len(value) <= 40:
            return pn.pane.Markdown(
                text,
                css_classes=["math"],
                styles={
                    "font-size": "27px",
                    "font-weight": "600",
                    "color": "#eef2fa",
                    "padding": "14px 8px",
                },
            )
        return pn.pane.Markdown(text, css_classes=["math"])
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
            options=["auto", "bar", "hbar", "line", "scatter", "donut", "table"],
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
                " <span style='background:rgba(239,68,68,.16);color:#f6a5a5;padding:1px 8px;"
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
            f"<span style='background:#1d2946;color:#9db0d6;padding:1px 7px;"
            f"border-radius:6px;font-size:11px;font-weight:600'>#{number}</span>"
        )
        self.expanded = False
        expand_btn = pn.widgets.Button(
            name="⤢", width=28, height=26, button_type="light", margin=1
        )
        expand_btn.on_click(lambda _e: self._toggle_expand())
        remove_btn = pn.widgets.Button(
            name="✕", width=28, height=26, button_type="light", margin=1
        )
        remove_btn.on_click(lambda _e: board.remove(self))
        self._controls = pn.Row(
            self.chart_choice,
            expand_btn,
            remove_btn,
            css_classes=["card-controls"],
            margin=0,
        )

        initial = phrase_value(result.value)
        self.takeaway = pn.pane.Markdown(
            f"→ {initial}" if initial else "",
            visible=bool(initial),
            css_classes=["math"],
            margin=(0, 12, 2, 12),
            styles={
                "font-size": "12.5px",
                "color": "#dbe3f1",
                "background": "#182339",
                "border-radius": "8px",
                "padding": "5px 10px",
            },
        )
        self._body = pn.Column(sizing_mode="stretch_width")
        self._redraw()

        code_view = pn.Accordion(
            ("code", pn.pane.Markdown(f"```python\n{result.code}\n```")),
            active=[],
            sizing_mode="stretch_width",
            stylesheets=[ACCORDION_CSS],
        )
        self.panel = pn.Column(
            self._controls,
            pn.pane.Markdown(
                f"{number_pill}&nbsp; <span style='font-weight:600;"
                f"font-size:13.5px;color:#eef2fa'>{question}</span>"
                f"{verdict_pill}",
                margin=(2, 34, 0, 10),
                sizing_mode="stretch_width",
            ),
            self.takeaway,
            self._body,
            code_view,
            styles={"background": "#121a2e", "padding": "12px 14px", **CARD_CSS},
            width=470,
            margin=(0, 12, 12, 0),
        )

    def _redraw(self) -> None:
        chart = None if self.chart_choice.value == "auto" else self.chart_choice.value
        self._body.objects = [
            render_value(
                self.result.value,
                chart,
                on_click=self._on_chart_click,
                height=500 if self.expanded else 260,
            )
        ]

    def _toggle_expand(self) -> None:
        self.expanded = not self.expanded
        self._redraw()
        self.board._rebuild_grid()

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
        # Sinks are the JS side-channels (drag order in, file names out for
        # the @ autocomplete); they must exist before the first chip render.
        self._order_sink = pn.widgets.TextInput(
            visible=False, css_classes=["order-sink"], width=0
        )
        self._order_sink.param.watch(self._on_drag_order, "value")
        self._files_sink = pn.widgets.TextInput(
            visible=False, css_classes=["files-sink"], width=0
        )
        self.file_chips = pn.Row(margin=(0, 6))
        self._render_chips()
        self._texter = NumpyCodeGen(
            system_prompt=(
                "You are a sharp, plain-spoken data analyst reading results "
                "computed from the user's own data. Lead with the finding, "
                "quantify it, and flag what a careful analyst would flag - "
                "tiny leaf samples, a split on one salesperson's name, likely "
                "overfitting, confounders. Never open with a textbook "
                "definition; the user asked about THEIR data, not the method. "
                "Inline LaTeX ($...$) renders - use it for formulas and "
                "statistical notation instead of spelling them out."
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
        self.story = pn.pane.Markdown(
            visible=False,
            css_classes=["math"],
            sizing_mode="stretch_width",
            styles={
                "background": "linear-gradient(135deg,#151f38 0%,#121a2e 70%)",
                "border": "1px solid rgba(124,154,255,.22)",
                "padding": "12px 18px",
                "border-radius": "14px",
                "font-size": "13.5px",
                "line-height": "1.62",
                "color": "#c3cddf",
                "max-width": "980px",
            },
        )
        self.board_caption = pn.pane.Markdown(margin=(10, 14, 0, 6))
        clear = pn.widgets.Button(
            name="Clear board",
            button_type="light",
            height=26,
            align="center",
            margin=(6, 6, 0, 0),
        )
        clear.on_click(lambda _e: self.clear_board())
        retell = pn.widgets.Button(
            name="Retell the story",
            button_type="light",
            height=26,
            align="center",
            margin=(6, 4, 0, 0),
        )
        retell.on_click(lambda _e: asyncio.create_task(self.retell()))
        self.board_header = pn.Row(
            self.board_caption,
            self._order_sink,
            self._files_sink,
            pn.Spacer(sizing_mode="stretch_width"),
            retell,
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
                "css_classes": ["math"],
                "show_reaction_icons": False,
                "show_copy_icon": False,
                "show_avatar": False,
                "show_timestamp": False,
                "stylesheets": [
                    """
                    .message {
                      color: #dbe3f1;
                      background: #172138;
                      border: 1px solid rgba(255,255,255,.08);
                      border-radius: 12px;
                      font-size: 13px;
                      line-height: 1.55;
                      box-shadow: none;
                    }
                    """
                ],
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
        self._files_sink.value = ",".join(self.files)
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
                    name="",
                    placeholder=name,
                    options=info["categories"],
                    width=168,
                    margin=(8, 5),
                    css_classes=["flt"],
                )
                self._filter_widgets.append((name, "cat", w))
                widgets.append(w)
            elif info.get("kind") == "datetime" and "min" in info:
                col = self.frame.data[name]
                w = pn.widgets.DatetimeRangePicker(
                    name="",
                    value=(col.min().to_pydatetime(), col.max().to_pydatetime()),
                    enable_time=False,
                    width=215,
                    margin=(8, 5),
                    css_classes=["flt"],
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

    @staticmethod
    def _summarise_value(value, budget: int = 500) -> str:
        """A value as narrator food: structure preserved, size bounded.

        A flat str() truncation starved the narrator - handed a dict of eight
        analyses cut at 220 chars, its caveat was that the material "isn't
        fully displayed". It was right.
        """
        if isinstance(value, dict):
            parts = []
            for key, item in value.items():
                parts.append(f"{key}: {Board._summarise_value(item, 90)}")
            return " | ".join(parts)[:budget]
        if isinstance(value, pd.Series) and len(value):
            bits = ", ".join(f"{k}={v}" for k, v in value.head(6).items())
            more = f" (+{len(value) - 6} more)" if len(value) > 6 else ""
            try:
                peak = f"; peak {value.idxmax()}={value.max()}"
            except (TypeError, ValueError):
                peak = ""
            return f"[{bits}{more}{peak}]"[:budget]
        if isinstance(value, pd.DataFrame):
            return (
                f"table {value.shape[0]}x{value.shape[1]} "
                f"cols={list(value.columns)[:8]}"
            )[:budget]
        return str(value)[:budget]

    def _story_material(self) -> str:
        lines = []
        for block in reversed(self.blocks):  # oldest first: narrative order
            lines.append(
                f"#{block.number} {block.question}\n"
                f"   found: {block.result.description or 'see values'}\n"
                f"   values: {self._summarise_value(block.result.value)}"
            )
        return "\n".join(lines)

    @property
    def _story_agent(self):
        if not hasattr(self, "_story_agent_obj"):
            from pydantic_ai import Agent

            self._story_agent_obj = Agent(
                model=DEFAULT_MODEL,
                output_type=_BoardStory,
                system_prompt=(
                    "You are a sharp, plain-spoken data analyst. You are given "
                    "the blocks of a dashboard and you say what they mean: "
                    "grounded in the numbers provided, never invented."
                ),
            )
        return self._story_agent_obj

    async def retell(self) -> None:
        """One structured call: the board's story plus a takeaway per card."""
        if len(self.blocks) < 2 or not HAS_KEY:
            return
        material = self._story_material()
        prompt = (
            "This dashboard currently shows the following blocks, in the "
            f"order they were made:\n\n{material}\n\n"
            "Return the story of the whole board (headline number first, "
            "evidence cited as #n, one caveat last) AND one takeaway per "
            "block: the single thing a human should conclude from that block, "
            "with its key number. Ground everything strictly in the material; "
            "invent nothing. Inline LaTeX ($...$) renders - use it for any "
            "formula or statistical notation."
        )
        try:
            result = await asyncio.to_thread(
                lambda: _run_coro(self._story_agent.run(prompt)).output
            )
        except Exception:
            return
        by_number = {b.number: b for b in self.blocks}
        with pn.io.unlocked():
            self.story.object = (
                "<span style='color:#9ca3af;font-size:11px;font-weight:600;"
                "letter-spacing:.08em'>THE STORY SO FAR</span>\n\n"
                + texify(result.story)
            )
            self.story.visible = True
            for item in result.takeaways:
                block = by_number.get(item.number)
                if block is not None and item.takeaway:
                    block.takeaway.object = f"→ {texify(item.takeaway)}"
                    block.takeaway.visible = True

    def drill(self, label: str, block: Block) -> None:
        """A clicked chart element becomes a board-wide filter, or a question.

        Priority: a category filter that knows this label filters the whole
        board instantly and reversibly. A month-shaped label narrows the date
        range. Anything else drafts a drill question into the chat input, so
        the model call stays under the user's control.
        """
        for _name, kind, w in self._filter_widgets:
            if kind == "cat" and label in w.options:
                w.value = [] if w.value == [label] else [label]  # click twice to clear
                return
        month = re.match(r"^(\d{4})-(\d{2})", label)
        if month:
            for _name, kind, w in self._filter_widgets:
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

    def _on_drag_order(self, event) -> None:
        """Adopt the order the user dragged into. No rebuild: the DOM already
        shows it, and re-rendering would snap cards mid-gesture."""
        try:
            numbers = [int(n) for n in str(event.new).split(",") if n]
        except ValueError:
            return
        by_number = {b.number: b for b in self.blocks}
        if sorted(numbers) != sorted(by_number):
            return
        self.blocks = [by_number[n] for n in numbers]

    def nudge(self, block: Block, delta: int) -> None:
        """Move a card one slot left or right in the flow."""
        i = self.blocks.index(block)
        j = max(0, min(len(self.blocks) - 1, i + delta))
        if i != j:
            self.blocks[i], self.blocks[j] = self.blocks[j], self.blocks[i]
            self._rebuild_grid()

    def clear_board(self) -> None:
        self.blocks.clear()
        self._rebuild_grid()

    def _rebuild_grid(self) -> None:
        n = len(self.blocks)
        self.board_caption.object = (
            f"<span style='font-weight:600;font-size:13.5px;color:#eef2fa'>"
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
        ordered = sorted(self.blocks, key=lambda b: not b.expanded)
        for b in self.blocks:
            classes = ["npi-card", f"card-{b.number}"]
            if b.fresh:
                classes.append("fresh-card")
            b.panel.css_classes = classes
            b.fresh = False
            # Size expresses content: an expanded card takes the row, a bare
            # metric needs a third of one, charts keep their standard width.
            if b.expanded:
                b.panel.width = 962
            elif isinstance(b.result.value, (int, float, np.floating, np.integer)):
                b.panel.width = 300
            else:
                b.panel.width = 470
        self.grid_holder.height = None
        self.grid_holder.objects = [
            pn.FlexBox(
                *[b.panel for b in ordered],
                sizing_mode="stretch_width",
                css_classes=["board-flex"],
            )
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
            context="\n".join(
                f"- asked: {q}  ->  {d or 'answered'}"
                for q, _c, d in self._chat_history[-6:]
            ),
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
            return pn.pane.Markdown(texify(result.value), css_classes=["math"])
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
                pn.pane.Markdown(texify(narrated), css_classes=["math"]),
                pn.Accordion(
                    (
                        "how I computed it",
                        pn.pane.Markdown(f"```python\n{result.code}\n```"),
                    ),
                    active=[],
                    stylesheets=[ACCORDION_CSS],
                ),
            )
        if result.ok and result.chat_only:
            # A conversational fact: answer inline, keep the board clean.
            return pn.pane.Markdown(
                texify(f"{phrase_value(result.value) or str(result.value)}"),
                css_classes=["math"],
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
        # Template fallback: a story arc of headline, trend, driver.
        out = []
        if nums:
            out.append(f"Overall total {nums[0]}.")
        if nums and dates:
            out.append(f"Monthly total {nums[0]} over {dates[0]}.")
        if nums and cats:
            out.append(f"Total {nums[0]} by {cats[0]}.")
        return out[:3]

    async def _starter_questions(self) -> list[str]:
        """Ask the model for three varied questions; fall back to templates.

        The templates always anchored on the first numeric and first
        categorical column, so every session opened with the same three
        views - the sameness the board was accused of started here.
        """
        summary = self.frame.metadata["column_summary"]
        cols = "\n".join(
            f"- {name}: {info.get('kind')} "
            f"({info.get('categories') or info.get('min', '')})"
            for name, info in summary.items()
        )
        try:
            from pydantic_ai import Agent

            agent = Agent(
                model=DEFAULT_MODEL,
                output_type=_Starters,
                system_prompt=(
                    "You propose sharp opening questions for a data dashboard."
                ),
            )
            result = await asyncio.to_thread(
                lambda: _run_coro(
                    agent.run(
                        f"The dataset has these columns:\n{cols}\n\n"
                        "Propose exactly three short questions, each a "
                        "different kind of analysis (headline metric, "
                        "relationship, distribution, segment gap, anomaly...). "
                        "Plain English, each answerable from these columns."
                    )
                ).output
            )
            questions = [q.strip() for q in result.questions if q.strip()]
            if len(questions) >= 2:
                return questions[:3]
        except Exception:
            pass
        return self.suggestions()

    async def autostart(self) -> None:
        if not HAS_KEY:
            self.status.object = "*No API key - autostart skipped.*"
            return
        for q in await self._starter_questions():
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
        await self.retell()
        with pn.io.unlocked():
            self.status.object = ""
            self.chat.send(
                "That's a starting point - the strip above the board sums up "
                "the story so far. What would you like to dig into?",
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
        "<span style='color:#eef2fa;font-size:17px;font-weight:700'>"
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
    styles={"background": "#121a2e", "padding": "6px", **CARD_CSS},
    sizing_mode="stretch_height",
)

filter_bar = pn.Row(
    pn.pane.Markdown(
        "<span style='color:#9ca3af;font-size:11px;font-weight:600;"
        "letter-spacing:.08em'>FILTERS</span>",
        margin=(0, 2, 0, 14),
        align="center",
    ),
    board.filters,
    pn.Spacer(sizing_mode="stretch_width"),
    board.status,
    align="center",
    sizing_mode="stretch_width",
    styles={"background": "#121a2e", "padding": "4px 10px", **CARD_CSS},
)

data_card = pn.Column(
    pn.pane.Markdown(
        "<span style='font-weight:600;font-size:13.5px;color:#eef2fa'>" "Data</span>",
        margin=(8, 14, 0, 14),
    ),
    board.table,
    sizing_mode="stretch_width",
    styles={"background": "#121a2e", "padding": "6px", **CARD_CSS},
)

right = pn.Column(
    filter_bar,
    board.board_header,
    board.story,
    board.grid_holder,
    data_card,
    sizing_mode="stretch_both",
    margin=(0, 4, 0, 14),
)

pn.config.raw_css.append("""
body {
  background: #0b1220;
  margin: 0;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
    'Helvetica Neue', Arial, sans-serif;
  font-feature-settings: 'cv02', 'tnum';
  color: #eef2fa;
}
:root { --design-primary-color: #7C9AFF; color-scheme: dark; }
.bk-input {
  background: #0f1830 !important; color: #e6eaf3 !important;
  border: 1px solid rgba(255,255,255,.10) !important;
}
.bk-input::placeholder { color: #7f8db0 !important; }
.bk-btn-light, .bk-btn-default {
  background: #16203a !important; color: #c3cddf !important;
  border: 1px solid rgba(255,255,255,.10) !important;
}
.bk-btn-light:hover, .bk-btn-default:hover { background: #1d2946 !important; }
.bk-btn-primary {
  background: #4f6ef7 !important; border-color: #4f6ef7 !important;
}
.choices__list--dropdown, .choices[data-type*=select-multiple] .choices__list {
  background: #141d33 !important; color: #dbe3f1 !important;
}
.choices__item--choice { color: #dbe3f1 !important; }
.choices__item--choice.is-highlighted { background: #243356 !important; }
select.bk-input option { background: #141d33; }
.npi-card { position: relative; transition: box-shadow .15s ease, transform .15s ease; }
.npi-card:hover {
  box-shadow: 0 10px 28px rgba(0,0,0,.45) !important;
  transform: translateY(-2px);
}
.npi-ghost { opacity: .35; }
.markdown, .markdown p, .markdown li, .markdown td, .markdown th,
.markdown em, .markdown strong { color: #dbe3f1; }
html { scrollbar-color: #2a3860 #0e1526; }
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-thumb { background: #24304f; border-radius: 8px; }
*::-webkit-scrollbar-track { background: transparent; }
:host(.flt) .choices__inner, :host(.flt) input.bk-input {
  border: 1px solid rgba(255,255,255,.10); border-radius: 9px; font-size: 12.5px;
  min-height: 32px; background: #0f1830;
}
:host(.flt) .choices__input::placeholder,
:host(.flt) input.bk-input::placeholder { color: #9ca3af; }
.card-controls {
  position: absolute; top: 8px; right: 8px; z-index: 5;
  opacity: 0; transition: opacity .15s ease;
  background: rgba(20,29,51,.95) !important;
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 9px; padding: 2px; box-shadow: 0 2px 10px rgba(15,23,42,.10);
}
:host(.npi-card:hover) .card-controls { opacity: 1; }
@keyframes cardflash {
  0%   { box-shadow: 0 0 0 2px #6366F1; }
  100% { box-shadow: 0 1px 2px rgba(15,23,42,.05); }
}
.fresh-card { animation: cardflash 2s ease-out; }
""")
app = pn.Row(left, right, sizing_mode="stretch_both", margin=12)
app.servable(title="numpyai dashboard")

pn.state.onload(board.autostart)

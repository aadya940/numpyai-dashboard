"""Tests for @mention parsing and the multi-table prompt."""

from __future__ import annotations

import pytest

from numpyai_dashboard._ai import prompt_frames

pd = pytest.importorskip("pandas")


@pytest.fixture
def two_tables():
    from numpyai_dashboard._utils import frame_metadata

    q1 = pd.DataFrame({"region": ["EMEA", "APAC"], "units": [3.0, 5.0]})
    q2 = pd.DataFrame({"region": ["EMEA"], "units": [7.0]})
    return {"q1_sales": frame_metadata(q1), "q2_sales": frame_metadata(q2)}


def test_prompt_frames_describes_every_table(two_tables):
    prompt = prompt_frames("compare units", two_tables)
    assert "`q1_sales` is a pandas DataFrame with 2 rows" in prompt
    assert "`q2_sales` is a pandas DataFrame with 1 rows" in prompt
    assert "df['" not in prompt  # tables go by their own names, not df


def test_prompt_frames_examples_use_real_names(two_tables):
    prompt = prompt_frames("compare units", two_tables)
    assert "q1_sales['units'].sum() - q2_sales['units'].sum()" in prompt


def test_prompt_frames_carries_feedback_and_history(two_tables):
    prompt = prompt_frames(
        "x",
        two_tables,
        prior_feedback="wrong table",
        history=[("earlier q", "code", "desc")],
    )
    assert "wrong table" in prompt
    assert "earlier q" in prompt


# ---- the app-side parser (needs panel installed) ---------------------------

pn = pytest.importorskip("panel")


@pytest.fixture(scope="module")
def app_module(tmp_path_factory):
    import dotenv

    dotenv.load_dotenv = lambda *a, **k: None
    import app.main as m

    return m


NAMES = ["sample_sales", "q2_sales"]


def test_exact_and_fuzzy_mentions(app_module):
    parse = app_module.parse_mentions
    assert parse("@sample_sales total", NAMES) == ["sample_sales"]
    assert parse("what about @q2?", NAMES) == ["q2_sales"]
    assert parse("@sample vs @q2", NAMES) == NAMES


def test_unknown_and_absent_mentions(app_module):
    parse = app_module.parse_mentions
    assert parse("meet @ 3pm about totals", NAMES) == []
    assert parse("no mentions here", NAMES) == []
    assert parse("@nonexistent file", NAMES) == []


def test_slug_is_identifier_safe(app_module):
    assert app_module.slug("Q1 Sales (final)") == "q1_sales_final"
    assert app_module.slug("2024 data") == "t_2024_data"

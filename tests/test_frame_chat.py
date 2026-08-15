"""Tests for frame.chat, with the model stubbed out. No LLM, no API key."""

from __future__ import annotations

import pytest

from numpyai_dashboard import ChatResult, frame
from numpyai_dashboard._ai import prompt_frame

pd = pytest.importorskip("pandas")

from test_chat import FakeCodeGen  # noqa: E402


@pytest.fixture
def sales():
    return pd.DataFrame(
        {
            "region": ["EMEA", "APAC", "EMEA", "AMER"],
            "units": [3.0, 5.0, 7.0, 2.0],
            "when": pd.to_datetime(
                ["2023-01-01", "2023-02-01", "2023-05-01", "2023-06-01"]
            ),
        }
    )


def make(sales, script, **kwargs):
    f = frame(sales, **kwargs)
    f._code_generator = FakeCodeGen(script)
    return f


GROUPBY = [
    (
        "output = df.groupby('region')['units'].sum()\nmetadata = 'units per region'",
        "groups",
        True,
        "",
    )
]


# --------------------------------------------------------------------------
# the frame is what the generated code sees
# --------------------------------------------------------------------------


def test_chat_can_group_by_a_text_column(sales):
    result = make(sales, GROUPBY).chat("units by region")
    assert isinstance(result, ChatResult)
    assert result.ok is True
    assert result.value["EMEA"] == 10.0
    assert result.value["APAC"] == 5.0


def test_chat_can_filter_by_a_date_column(sales):
    script = [
        (
            "output = df[df['when'] > '2023-03-01']\nmetadata = 'later orders'",
            "filters",
            True,
            "",
        )
    ]
    result = make(sales, script).chat("orders after March")
    assert len(result.value) == 2


def test_numpy_is_available_alongside_pandas(sales):
    script = [
        (
            "region = df['region'].to_numpy()\n"
            "output = float(np.nansum(df['units'].to_numpy()[region == 'EMEA']))\n"
            "metadata = 'EMEA units via numpy'",
            "numpy",
            True,
            "",
        )
    ]
    assert make(sales, script).chat("EMEA units").value == 10.0


def test_pandas_is_bound_as_pd(sales):
    script = [
        ("output = pd.Series([1, 2]).sum()\nmetadata = 'pd is bound'", "pd", True, "")
    ]
    assert make(sales, script).chat("check pd").value == 3


# --------------------------------------------------------------------------
# metadata and prompt
# --------------------------------------------------------------------------


def test_metadata_describes_every_column(sales):
    summary = frame(sales).metadata["column_summary"]
    assert summary["region"]["categories"] == ["EMEA", "APAC", "AMER"]
    assert summary["units"]["kind"] == "numeric"
    assert summary["when"]["kind"] == "datetime"


def test_metadata_follows_a_filtered_frame(sales):
    """Stale metadata would let the model answer about rows that are gone."""
    f = frame(sales)
    assert f.metadata["rows"] == 4
    f.data = sales[sales["region"] == "EMEA"]
    assert f.metadata["rows"] == 2
    assert f.metadata["column_summary"]["region"]["categories"] == ["EMEA"]


def test_prompt_lists_columns_and_categories(sales):
    prompt = prompt_frame("units by region", frame(sales).metadata)
    assert "df['region']" in prompt
    assert "'EMEA'" in prompt
    assert "df['when']" in prompt
    assert "datetime64" in prompt


def test_prompt_carries_the_rejection_reason(sales):
    prompt = prompt_frame("x", frame(sales).metadata, prior_feedback="wrong column")
    assert "wrong column" in prompt


# --------------------------------------------------------------------------
# proxying and misuse
# --------------------------------------------------------------------------


def test_frame_proxies_to_the_dataframe(sales):
    f = frame(sales)
    assert f.shape == (4, 3)
    assert f["units"].sum() == 17.0
    assert list(f.columns) == ["region", "units", "when"]


def test_data_exposes_the_real_dataframe(sales):
    assert isinstance(frame(sales).data, pd.DataFrame)


def test_rejects_a_non_dataframe():
    with pytest.raises(TypeError, match="must be a pandas.DataFrame"):
        frame([1, 2, 3])


def test_rejects_a_non_string_query(sales):
    with pytest.raises(TypeError, match="query must be a string"):
        make(sales, GROUPBY).chat(42)

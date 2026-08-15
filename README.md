<p align="center">
<img src="logo.png" alt="numpyai-dashboard logo" width="500">
</p>

# numpyai-dashboard

Load spreadsheets and CSVs into [NumPy](https://github.com/numpy/numpy) and explore
them in plain English.

Built on [Pydantic AI](https://ai.pydantic.dev/), so Google Gemini, OpenAI,
Anthropic or any other supported model works without touching the library code.

Forked from [numpyai](https://github.com/aadya940/numpyai). It installs as
`numpyai_dashboard`, so it will not collide with an existing `numpyai` install.

## Features

- Load a spreadsheet or CSV into a DataFrame with one call.
- Ask questions in English; the library generates and executes NumPy code for you.
- `numpyai_dashboard.Diagnosis` suggests analysis steps for your data.
- `numpyai_dashboard.NumpyAISession` chats over multiple arrays at once.
- Generated code is syntax-checked and independently validated before returning.
- Automatic retries with error context.
- Verbose mode (`verbose=True`) prints every intermediate step.

## Installation

```sh
pip install "numpyai-dashboard[all]"
```

Or install only the pieces you need:

```sh
pip install "numpyai-dashboard[excel]"     # .xlsx/.xls/.xlsb/.ods loading
pip install "numpyai-dashboard[csv]"       # .csv/.tsv loading
pip install "numpyai-dashboard[google]"    # Google Gemini
pip install "numpyai-dashboard[openai]"    # OpenAI
pip install "numpyai-dashboard[anthropic]" # Anthropic Claude
```

### From source

```sh
git clone https://github.com/aadya940/numpyai-dashboard
cd numpyai-dashboard
pip install -e ".[all,dev]"
```

## Setup

Set the API key for your chosen provider. Pydantic AI reads standard env vars:

| Provider  | Environment variable |
| --------- | -------------------- |
| Google    | `GEMINI_API_KEY`     |
| OpenAI    | `OPENAI_API_KEY`     |
| Anthropic | `ANTHROPIC_API_KEY`  |

```sh
export GEMINI_API_KEY=...
```

## Usage

### Loading

`read_excel` handles `.xlsx`, `.xls`, `.xlsb` and `.ods`; `read_csv` handles
delimited text and forwards any extra keyword to `pandas.read_csv`.

```python
import numpy as np
import numpyai_dashboard as npi

df = npi.read_excel("sales.xlsx")         # or sheet="Q3", header=False, n_rows=1000
df = npi.read_csv("sales.csv.gz", usecols=["region", "units"])
```

Both return a `frame`: a DataFrame that can answer questions. Attribute and item
access pass through, so `df.head()` and `df["revenue"] = ...` work as usual, and
`df.data` is the real DataFrame that a Panel `Tabulator` wants.

CSV carries no types, so pass `parse_dates=["order_date"]` for real `datetime64`.

### Asking questions

Every column is visible to the model, with its dtype, range, and the distinct
values of the categorical ones, so questions can name columns and span types.
Both pandas and NumPy are in scope and the model picks whichever fits.

```python
df["revenue"] = df["units"] * df["unit_price"] * (1 - df["discount"].fillna(0))

result = df.chat("Total revenue by region since March.")

result.value        # the answer
result.code         # the code that produced it
result.description  # the model's own summary
result.judgment     # verdict, and the reason if rejected
result.attempts     # how many tries it took
result.ok           # False if every attempt was rejected
```

Failure never raises: `ok` is False, `value` is None, and `errors` holds one
entry per attempt.

`npi.array(...)` is the same interface over a NumPy array, showing the model one
homogeneous array rather than a table:

```python
arr = npi.array(np.array([[1, 2, 3], [4, 5, np.nan]]))
arr.chat("Replace missing values with the column mean.").value
```

Pass any Pydantic AI spec via `model=`, such as `model="openai:gpt-4o"`, or a
pre-configured `pydantic_ai.models.Model`.

### Several arrays at once

`NumpyAISession` exposes each array to the model as `arr1`, `arr2`, ...
`Diagnosis` suggests analysis steps rather than computing an answer.

```python
sess = npi.NumpyAISession([np.array([[1, 2, 3]]), np.random.random((1, 3))])
sess.chat("Impute the first array with the mean of the second.").value

npi.Diagnosis(sess).steps(task="Give me 5 steps to analyse this data.")
```

## Supported LLM providers

Anything Pydantic AI supports: Google (Gemini), OpenAI, Anthropic, Groq, Mistral,
Ollama, and OpenAI-compatible endpoints. See the
[Pydantic AI model docs](https://ai.pydantic.dev/models/) for the full list.

## Examples

Runnable notebooks live in [`examples/`](examples). Start with
[`test_all_functionality_excel.ipynb`](examples/test_all_functionality_excel.ipynb),
which loads the bundled `sample_sales.xlsx` and walks from spreadsheet to
DataFrame to natural-language questions.

They expect `pip install -e ".[all,dev]"` and a provider key in `examples/.env`.

## Contributing

- Format with `black` and lint with `ruff`.
- Add tests under `tests/`.
- Public API (`array`, `NumpyAISession`, `Diagnosis`, `read_excel`, `read_csv`)
  should stay stable.

## License

MIT, see [LICENSE](LICENSE). Forked from
[numpyai](https://github.com/aadya940/numpyai), also MIT.

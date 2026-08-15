<p align="center">
<img src="logo.png" alt="numpyai-dashboard logo" width="500">
</p>

# numpyai-dashboard

Ask questions about spreadsheets, CSVs and
[NumPy](https://github.com/numpy/numpy) arrays in plain English.

Forked from [numpyai](https://github.com/aadya940/numpyai). It installs as
`numpyai_dashboard`, so it will not collide with an existing `numpyai` install.

## Features

- Load a spreadsheet or CSV with one call.
- Ask questions in English and get the answer, plus the code behind it.
- Wrong answers are caught and retried rather than returned.
- `verbose=True` shows every step.

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

```python
import numpy as np
import numpyai_dashboard as npi

df = npi.read_excel("sales.xlsx")         # or sheet="Q3", header=False, n_rows=1000
df = npi.read_csv("sales.csv.gz", usecols=["region", "units"])
```

`read_excel` handles `.xlsx`, `.xls`, `.xlsb` and `.ods`. `read_csv` takes any
`pandas.read_csv` keyword, and needs `parse_dates=[...]` for real dates.

Use it as a DataFrame. `df.data` gives the underlying one.

### Asking questions

```python
df["revenue"] = df["units"] * df["unit_price"] * (1 - df["discount"].fillna(0))

result = df.chat("Total revenue by region since March.")

result.value        # the answer
result.code         # the code behind it
result.description  # one-line summary
result.judgment     # verdict, and why if rejected
result.attempts     # tries taken
result.ok           # False if it could not answer
```

Questions can name any column and mix types. Failure never raises: `ok` is
False, `value` is None, and `errors` holds one entry per attempt.

Same interface over a NumPy array:

```python
arr = npi.array(np.array([[1, 2, 3], [4, 5, np.nan]]))
arr.chat("Replace missing values with the column mean.").value
```

Pick a model with `model="openai:gpt-4o"`, or pass a `pydantic_ai.models.Model`.

### Several inputs at once

```python
sess = npi.NumpyAISession([np.array([[1, 2, 3]]), npi.read_excel("sales.xlsx")])
sess.chat("Compare the first array against the revenue column.").value

npi.Diagnosis(sess).steps(task="Give me 5 steps to analyse this data.")
```

Arrays and tables can be mixed; they are named `arr1`, `df2`, ... by position.
`Diagnosis` suggests steps instead of computing an answer.

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

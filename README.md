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

### Loading a spreadsheet

Requires `numpyai-dashboard[excel]`. Reads `.xlsx`, `.xls`, `.xlsb` and `.ods`.

```python
import numpyai_dashboard as npi

df = npi.read_excel("sales.xlsx")     # or sheet="Q3", header=False, n_rows=1000
print(df.columns.tolist())            # ['region', 'order_date', 'units', 'price']
```

You get a `frame`: a DataFrame that can answer questions. Attribute and item
access pass straight through, so `df.head()`, `df["units"]` and
`df["revenue"] = ...` work as usual. `df.data` is the real DataFrame, which is
what a Panel `Tabulator` wants.

Reading goes through [fastexcel](https://github.com/ToucanToco/fastexcel) and the
Rust [calamine](https://github.com/tafia/calamine) parser, which measures about 3x
faster and half the memory of reading calamine from Python.

### Loading delimited text

Requires `numpyai-dashboard[csv]`. `read_csv` mirrors `read_excel`'s contract, and
forwards any other keyword to `pandas.read_csv`.

```python
df = npi.read_csv("sales.csv")            # or header=False, n_rows=1000
df = npi.read_csv("sales.tsv")            # tab inferred from the extension
df = npi.read_csv("sales.csv.gz", usecols=["region", "units"])
```

Delimited text carries no types, so pass `parse_dates=["order_date"]` through to
pandas for real `datetime64` columns.

### Asking about a table

Every column is visible to the model, with its dtype, its range, and the distinct
values of the categorical ones. So questions can name columns and span types:

```python
df["revenue"] = df["units"] * df["unit_price"] * (1 - df["discount"].fillna(0))

df.chat("Total revenue by region since March.").value
# region
# AMER     58373.70
# APAC     75876.63
# EMEA     70421.71
# LATAM    65757.28
```

Both pandas and NumPy are in scope, so the model picks whichever suits the
question:

```python
output = df.groupby("region")["revenue"].sum()                       # pandas
output = np.nansum(rev[df["region"].to_numpy() == "EMEA"])           # numpy
```

For a NumPy array rather than a table, `npi.array(...)` works the same way and
shows the model one homogeneous array instead.

### What a question returns

`chat` returns a `ChatResult`, not a bare value, so the code and the verdict are
available to a caller rather than only printed.

```python
result = arr.chat("Total revenue after discount.")

result.value        # the answer
result.code         # the NumPy that produced it
result.description  # the model's own summary
result.judgment     # verdict, and the reason if rejected
result.attempts     # how many tries it took
result.ok           # False if every attempt was rejected
```

Failure never raises. `result.ok` is False, `result.value` is None, and
`result.errors` holds one entry per attempt.

### Single array

```python
import numpy as np
import numpyai_dashboard as npi

data = np.array([[1, 2, 3, 4, 5, np.nan], [np.nan, 3, 5, 3.1415, 2, 2]])
arr = npi.array(data)  # defaults to google:gemini-2.5-flash

print(arr.chat("Compute the height and width of the image using NumPy.").value)
# Expected output: (2, 6)
```

### Choosing a model

Pass any Pydantic AI model spec via `model=`:

```python
npi.array(data, model="anthropic:claude-sonnet-4-5")
npi.array(data, model="openai:gpt-4o")
npi.array(data, model="google:gemini-2.5-pro")
```

A pre-configured `pydantic_ai.models.Model` instance also works.

### Multiple arrays

```python
import numpy as np
import numpyai_dashboard as npi

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.random.random((2, 3))

sess = npi.NumpyAISession([arr1, arr2])
result = sess.chat("Impute the first array with the mean of the second array.")
imputed = result.value
```

### Diagnosis

```python
sess = npi.NumpyAISession([arr1, arr2])
diag = npi.Diagnosis(sess)
steps = diag.steps(
    task="Give me exactly 7 pithy steps to select an ML model for this data."
)
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

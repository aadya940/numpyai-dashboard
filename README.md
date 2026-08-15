<p align="center">
<img src="logo.png" alt="numpyai-dashboard logo" width="500">
</p>

# numpyai-dashboard

Load spreadsheets into [NumPy](https://github.com/numpy/numpy) and explore them in
plain English.

Built on [Pydantic AI](https://ai.pydantic.dev/), so Google Gemini, OpenAI,
Anthropic or any other supported model works without touching the library code.

**Status:** early. The spreadsheet loading layer works; the dashboard layer is next.

Forked from [numpyai](https://github.com/aadya940/numpyai). It installs as
`numpyai_dashboard`, so it will not collide with an existing `numpyai` install.

## Features

- Load a spreadsheet into a DataFrame with one call, via a Rust reader. Text, dates
  and numbers are all preserved, each with its own dtype.
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

Requires `numpyai-dashboard[excel]`. Reads `.xlsx`, `.xls`, `.xlsb` and `.ods`, and
returns a `pandas.DataFrame`.

```python
import numpyai_dashboard as npi

df = npi.read_excel("sales.xlsx")     # or sheet="Q3", header=False, n_rows=1000
print(df.columns.tolist())            # ['region', 'date', 'units', 'price']
```

Every column is kept, with its type inferred by the reader:

| In the sheet | Becomes | Blank cells |
| --- | --- | --- |
| numbers | `float64` / `int64` | `NaN` |
| dates and datetimes | `datetime64` | `NaT` |
| `TRUE` / `FALSE` | `bool` | null |
| anything else | string | null |

Numeric columns hand off to NumPy for free, so you can mix the two freely:

```python
units = df["units"].to_numpy()
np.nansum(units[df["region"].to_numpy() == "EMEA"])
```

Reading is delegated to [fastexcel](https://github.com/ToucanToco/fastexcel), which
wraps the Rust [calamine](https://github.com/tafia/calamine) parser and emits Arrow
data directly. Type inference happens in Rust and the data crosses into Python as
columnar buffers rather than one object per cell, which measures about 3x faster and
roughly half the memory of driving calamine from Python.

### Single array

```python
import numpy as np
import numpyai_dashboard as npi

data = np.array([[1, 2, 3, 4, 5, np.nan], [np.nan, 3, 5, 3.1415, 2, 2]])
arr = npi.array(data)  # defaults to google:gemini-2.5-flash

print(arr.chat("Compute the height and width of the image using NumPy."))
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
imputed = sess.chat("Impute the first array with the mean of the second array.")
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
- Public API (`array`, `NumpyAISession`, `Diagnosis`, `read_excel`) should stay stable.

## License

MIT, see [LICENSE](LICENSE). Forked from
[numpyai](https://github.com/aadya940/numpyai), also MIT.

<p align="center">
<img src="logo.png" alt="numpyai-dashboard logo" width="500">
</p>

# numpyai-dashboard

Load spreadsheets into [NumPy](https://github.com/numpy/numpy) and explore them in
plain English.

`numpyai-dashboard` reads `.xlsx`/`.xls`/`.xlsb`/`.ods` straight into a NumPy array,
carries your column names through to the model, and lets you ask questions about the
data in natural language. It is built on [Pydantic AI](https://ai.pydantic.dev/), so
Google Gemini, OpenAI, Anthropic or any other supported model works without touching
the library code.

> **Status:** early. The spreadsheet loading layer is in place; the dashboard/
> visualization layer is the next milestone.
>
> This project is a fork of [numpyai](https://github.com/aadya940/numpyai) and
> currently shares most of its core. It installs as `numpyai_dashboard`, so it will
> not collide with an existing `numpyai` install.

## Features

- Load a spreadsheet into a NumPy array with one call, column names included.
- Ask questions in English; the library generates and executes NumPy code for you.
- `numpyai_dashboard.Diagnosis` suggests analysis steps for your data.
- `numpyai_dashboard.NumpyAISession` chats over multiple arrays at once.
- Generated code is syntax-checked and independently validated before returning.
- Automatic retries with error context.
- Verbose mode (`verbose=True`) prints every intermediate step.
- Provider-agnostic - any Pydantic AI model spec works.

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

| Provider  | Environment variable  |
| --------- | --------------------- |
| Google    | `GEMINI_API_KEY`      |
| OpenAI    | `OPENAI_API_KEY`      |
| Anthropic | `ANTHROPIC_API_KEY`   |

```sh
export GEMINI_API_KEY=...
```

## Usage

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

You can also pass a pre-configured `pydantic_ai.models.Model` instance for full control.

### Loading a spreadsheet

Requires `numpyai-dashboard[excel]`. Reads `.xlsx`, `.xls`, `.xlsb` and `.ods` via
[python-calamine](https://github.com/dimastbk/python-calamine).

```python
import numpyai_dashboard as npi

arr = npi.read_excel("sales.xlsx")          # or sheet="Q3", header=False
print(arr.columns)                          # ['units', 'unit_price', 'discount']
print(arr.chat("Total revenue after discount."))
```

Column names are passed to the model, so you can refer to them in plain English
rather than by index.

Because the array handed to the model is a homogeneous `float64` matrix, only
columns that convert cleanly are kept:

| In the sheet | Becomes |
| --- | --- |
| numbers, numeric text | `float64` |
| blank cells | `NaN` |
| `TRUE` / `FALSE` | `1.0` / `0.0` |
| text, dates | **dropped**, with a `UserWarning` naming each one |

Nothing is dropped silently — if a column you needed went missing, the warning
tells you which and why.

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

Anything Pydantic AI supports - Google (Gemini), OpenAI, Anthropic, Groq, Mistral,
Ollama, and OpenAI-compatible endpoints. See the
[Pydantic AI model docs](https://ai.pydantic.dev/models/) for the full list.

## Contributing

- Format with `black` and lint with `ruff`.
- Add tests under `tests/`.
- Public API surface (`array`, `NumpyAISession`, `Diagnosis`, `read_excel`) should
  stay stable.

## License

MIT - see [LICENSE](LICENSE). Forked from
[numpyai](https://github.com/aadya940/numpyai), also MIT.

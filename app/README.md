# Dashboard app

A Panel app over `numpyai_dashboard`. Chat on the left; every answered
question becomes a draggable, resizable block on the right, with the code
that produced it one click away. The filter bar re-executes every block
against the filtered rows without another model call.

## Run

```sh
pip install -e ".[ui,google]"
export GEMINI_API_KEY=...        # or put it in examples/.env
panel serve app/main.py --show --websocket-max-message-size 524288000 \\
  --static-dirs assets=app/assets
```

Loads `examples/sample_sales.xlsx` by default; upload any `.xlsx`, `.xls`,
`.xlsb`, `.ods`, `.csv` or `.tsv` to replace it. On load it builds a small
starter dashboard from the file's own columns, streaming blocks in as the
model answers.

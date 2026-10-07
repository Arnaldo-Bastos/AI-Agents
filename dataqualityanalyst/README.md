# Data Quality CrewAI Agent

The profiler calculates dataset and column measurements with Pandas/NumPy.
One CrewAI agent interprets those measurements and writes a standalone HTML
report. The HTML document is validated before saving. No separate execution
plan, action JSON or diagnostic analysis text is generated.

## Install

Requires Python >=3.10,<3.14. From this directory:

```powershell
uv sync
```

## Configure

The `.env` file selects the model. The default is `ollama/llama3.2`.

```env
MODEL=ollama/llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

Run Ollama and download the selected model before executing the agent.
Other CrewAI-supported providers can be selected with `MODEL` and their
required credentials.

## Run

```powershell
uv run python main.py "https://example.com/data/dataset.csv"
```

The dataset argument must be an HTTP(S) URL. The profiler reads CSV, TSV, Excel,
Parquet, JSON and JSONL/NDJSON files, and ZIP/TAR archives containing exactly one
supported data file, or an explicitly selected file when there are several:


For nested archives, separate each archive path with `::`. For example, The Census Income KDD
ZIP contains `census.tar.gz`, which contains separate training and test files:

```powershell
uv run python main.py "https://archive.ics.uci.edu/static/public/117/census+income+kdd.zip" --archive-member "census.tar.gz::census-income.data"
```

Select `census.tar.gz::census-income.test` for the test dataset. Comma-delimited
`.data` and `.test` files are read without a header, with generated column names
(`column_1`, `column_2`, ...). Metadata in `.names` files is not imported.
Archives are read in memory, including compressed TAR files, with at most eight
archive levels.

Use the exact internal path listed in the error. Files are analyzed individually;
participant files are not automatically concatenated. `.txt` members require
explicit selection because they may contain documentation or custom recordings.
CSV-like files automatically detect common delimiters,
including comma, semicolon and tabulation. CSV loading preserves textual
missing-value tokens and blank strings.

## Report

The output is a standalone HTML report with Executive Summary, Dataset Overview,
Prioritized Findings, Recommendations, Positive Observations, Methodology and
Limitations, ending with **Actions Before Analysis**. This final section contains
practical recommendations with exact columns, record selection, operation,
parameters and verification. Unknown domain decisions must be explicit.
The agent recommends treatments and does not modify the dataset.

For every eligible continuous numeric feature with detected IQR outliers, the
prompts require an embedded SVG histogram in an **Outlier Histograms** section
before the final actions. Charts use measured bins, highlight outlier frequencies
and show the IQR bounds, outlier count and percentage. The LLM infers likely
feature meaning from the name and measured profile, and does not create IQR
outlier charts for boolean/binary values, identifiers, codes, categorical
encodings, ranks, dates/timestamps or geographic coordinates. For example,
`lat`, `latitude`, `lon` and `longitude` are treated as likely geographic
coordinates unless the profile establishes otherwise. The semantic check also
excludes numeric-looking identifiers such as `cc_num`, card/account/customer or
transaction identifiers, invoice/order numbers, CPF/SSN, phone and ZIP/postal
codes. When a numeric field is ambiguous, it is excluded conservatively.
Outliers are statistical anomalies, not confirmed errors.

Charts are static inline SVGs, not JavaScript-generated graphics. Each chart
includes aligned bars and axes in the same plot coordinate system, measured
numeric ticks on both axes, IQR bound markers when visible, labels and a legend
within its `viewBox`.

The model receives measured column profiles, flags and aggregated histogram
arrays, without raw row previews. HTML generation is performed by the model; the guardrail
checks for a complete document, ordered head/body structure, a closed plain-text
title, required section terms and absence of scripts, with two retries.
This validation does not establish the correctness of every recommendation.

## Files

- `main.py`: LLM configuration, profiling, HTML report task and validation, saving.
- `tools/data_quality_profiler.py`: measured data-quality evidence.
- `models.py`: optional structured audit schema reference, unused by this flow.

## Analytical limitations

Outliers and rare categories are not automatically errors. Business validity
requires domain rules. LLM interpretations need review; the report includes
profiler evidence so recommendations can be checked against measurements.
A local model may take several minutes to analyze a large profile.

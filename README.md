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

## Run

```powershell
uv run python main.py "https://example.com/data/telco_churn_dataset.csv"
```

The default output is `output/data_quality_report.html`. Change it with:

```powershell
uv run python main.py "https://example.com/data/dataset.zip" --output output/my_report.html
```

Public Sciebo links of the form `https://host.sciebo.de/s/token` are automatically
resolved to `/download`, including links with `?opendetails=`.

The dataset argument must be an HTTP(S) URL. The profiler reads CSV, TSV, Excel,
Parquet, JSON and JSONL/NDJSON files, and ZIP archives containing exactly one
supported data file, or an explicitly selected file when there are several:

```powershell
uv run python main.py "https://example.com/WESAD.zip" --archive-member "WESAD/S10/S10_quest.csv"
```

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

The model receives measured column profiles and flags, without raw row previews
or histogram arrays. HTML generation is performed by the model; the guardrail
checks for a complete document and required section terms, with two retries.
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

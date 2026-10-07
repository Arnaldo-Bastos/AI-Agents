#-----------------------------------------------------------------------------
#                                       IMPORTS
import argparse
import os
import re
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from crewai import (
                       Agent,
                       Crew,
                       LLM,
                       Process,
                       Task,
                       TaskOutput,
                   )
from tools.data_quality_profiler import DatasetQualityProfilerTool
from tools.histograms import prepare_histograms, insert_histograms
#-----------------------------------------------------------------------------


load_dotenv()  # to load the API's key from LLMs models


def build_llm() -> LLM:
    """
        Build the CrewAI LLM from environment variables.
        Examples:
            MODEL=openai/gpt-4.1-mini
            MODEL=ollama/llama3.1:8b
    """ 

    #model = os.getenv("MODEL", "ollama/llama3.2")
    model = os.getenv("MODEL", "openai/gpt-4.1-mini")
    kwargs = {}
    if model.startswith("ollama/"):
        kwargs["base_url"] = os.getenv(
                                        "OLLAMA_BASE_URL",
                                        "http://localhost:11434",
                                      )

    return LLM(
                model = model,
                **kwargs,
              )


profiler_tool = DatasetQualityProfilerTool()


def build_agent() -> Agent:

    return Agent(
                  role = ("Senior Data Quality Analyst and " "Data Reliability Specialist"),
                  goal = """
                             Autonomously investigate datasets, identify measurable
                             data-quality risks, distinguish confirmed problems from
                             statistical anomalies, and recommend safe and technically
                             appropriate treatments without inventing facts.
                             Include measured histograms only for continuous
                             numeric features whose identified outliers presence.
                         """,
                  backstory = """
                                  You specialize in data quality, exploratory analysis and reliable analytical pipelines.
                                  Never estimate statistics that can be obtained from a tool.
                                  Distinguish evidence from interpretation.
                                  Outliers are not automatically errors.
                                  Never create histograms for boolean or binary features, even when encoded numerically.
                                  Try to identify the feature meaning by their names and just generate the histograms if it makes sense.
                                  When a feature's meaning is ambiguous, conservatively omit it from IQR outlier findings and charts rather than guessing.
                                  Missing values are not automatically candidates for imputation.
                                  Rare categories are not automatically errors.
                                  Never invent business rules.
                                  If domain knowledge is required, explicitly say so.
                                  Recommendations must be supported by evidence and explain potential analytical consequences.
                                  Keep recommendations concise: state the finding, its impact and the needed decision.
                              """,
                  llm = build_llm(),
                  tools = [],
                  reasoning = False,
                  max_iter = 15,
                  allow_delegation = False,
                  verbose = True,
                  cache = True,
                  respect_context_window = True,
                )


def validate_html_report(result: TaskOutput,) -> tuple[bool, Any]:

    # Deterministically validate the final report before saving it.

    # Extract only a complete document, discarding prose and outer code fences.
    # Add the deterministic HTML5 declaration instead of asking the LLM to retry
    # an otherwise valid report just because it omitted the declaration.
    document = re.search(
                          r"<html\b[^>]*>.*?</html\s*>",
                          result.raw,
                          flags = (re.IGNORECASE)
                          | (re.DOTALL),
                        )
    if document is None:
        return (
                 False,
                 "Return a complete HTML report enclosed in <html>...</html>, " "including Executive Summary, Dataset Overview, Prioritized " "Findings, Recommended Treatments, Methodology and a final Actions Before Analysis section. " "Do not return an explanation or a partial HTML fragment.",
               )

    html = (
             "<!DOCTYPE html>\n" + document.group(0)
                                           .strip()
           )
    lower = html.lower()

    structure = re.fullmatch(
        r"<html\b[^>]*>\s*<head\b[^>]*>(.*?)</head\s*>\s*"
        r"<body\b[^>]*>.*?</body\s*>\s*</html\s*>",
        document.group(0).strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if structure is None:
        return (
            False,
            "Return a complete HTML document with <head>...</head> followed by "
            "<body>...</body> inside <html>...</html>. Close every tag explicitly.",
        )

    head = structure.group(1)
    if not re.search(r"<title\b[^>]*>[^<]*</title\s*>", head, flags=re.IGNORECASE):
        return (
            False,
            "Include <title>Report title</title> inside <head>. The title must contain "
            "only escaped plain text and must close with </title> before <style> or "
            "any other element. An unclosed title makes the report appear blank.",
        )

    required_terms = [
                       "executive",
                       "dataset",
                       "finding",
                       "recommend",
                       "methodology",
                       "actions before analysis",
                     ]

    missing = [term for term in required_terms if term not in lower]

    if missing:
        return (
                 False,
                 f"Missing required report sections: {missing}",
               )

    if re.search(r"<script\b", html, flags=re.IGNORECASE):
        return (
                 False,
                 "Do not use JavaScript in the report. Render each eligible histogram as a " "complete static inline SVG with bars, axes, ticks, labels and legend already " "positioned inside its viewBox.",
               )

    return True, html
def main() -> None:

    parser = argparse.ArgumentParser(
                                      description = ("Autonomous CrewAI agent for dataset data-quality auditing.")
                                    )

    parser.add_argument(
                         "dataset_url",
                         help = ("HTTP(S) URL of a CSV, TSV, DATA/TEST, Excel, Parquet, JSON, JSONL or ZIP/TAR dataset."),
                       )

    parser.add_argument(
                         "--output",
                         default = "output/data_quality_report.html",
                         help = "Path for the generated HTML report.",
                       )

    parser.add_argument(
                         "--archive-member",
                         help = "Exact path in a ZIP/TAR archive. Separate nested archive paths with :: (outer.tar.gz::data.csv)."
                       )
    args = parser.parse_args()

    report_path = str(
                       Path(args.output)
                     )

    agent = build_agent()

    try:
        profile = profiler_tool.run(
                                     dataset_url = args.dataset_url,
                                     archive_member = args.archive_member,
                                     include_value_examples = False
                                   )
    except ValueError as exc:
        parser.exit(2, f"Dataset error: {exc}\n")
    columns, histogram_charts = prepare_histograms(profile.columns_profile)

    def validate_generated_report(result: TaskOutput) -> tuple[bool, Any]:
        if re.search(r"<svg\b", result.raw, flags=re.IGNORECASE):
            return False, "Use the supplied histogram_token instead of writing SVG yourself."
        valid, html = validate_html_report(result)
        if not valid:
            return valid, html
        if re.search(r"<h2\b[^>]*>\s*Outlier Histograms\s*</h2>", html, re.IGNORECASE):
            if not any(token in html for token in histogram_charts):
                return False, "The Outlier Histograms section must contain the exact supplied histogram_token for each eligible column."
        try:
            return True, insert_histograms(html, histogram_charts)
        except ValueError as exc:
            return False, str(exc)

    audit_task = Task(
                       description = (
                                       """
                                           - Produce a complete standalone HTML data-quality report from the measured profile below.
                                           - Return only <html>...</html>, with inline CSS and escaped dataset paths and column names.
                                           - Use the complete structure <html><head><meta charset="UTF-8">
                                             <title>Data Quality Report</title><style>...</style></head><body>...</body></html>.
                                           - The title must contain only escaped plain text. Always close </title> before
                                             opening <style>; close </style> and </head> before opening <body>.
                                           - Check that all non-void HTML and SVG elements have matching closing tags
                                             and that the report content is inside <body> before returning the document.
                                           - Include this sections: Executive Summary, Dataset Overview, Prioritized Findings,
                                             Recommendations, Positive Observations, and Methodology and Limitations.
                                           - Include all measured flags with column, severity and evidence.
                                           - Do not invent statistics, columns, thresholds, validity rules, images or 
                                             claims of completed cleanup.
                                           - Do not modify the dataset. 
                                           - Do not produce separate execution plans or files.
                                           - Do not use code fences or return a plan instead of the report.
                                           HISTOGRAM ELIGIBILITY AND DATA:
                                           - Include Outlier Histograms only for semantically eligible continuous numeric
                                             measurements with iqr_outliers.count > 0 and a supplied histogram_token.
                                           - For each eligible feature, a histogram is mandatory; do not replace it with tables,
                                             prose or recommendations to plot later. Use the exact measured column name.
                                           - Never create an outlier chart or label values as outliers for boolean/binary features, 
                                             including 0/1 encodings, or for identifiers, codes, categorical encodings, ranks, 
                                             dates/timestamps, or geographic coordinates.
                                           - Do not create the Outlier Histograms section when no eligible feature exists.
                                           - Charts are rendered deterministically by Python after your response.
                                             Do not write SVG, chart coordinates, bars, chart captions or chart placeholders
                                             of your own. Do not use JavaScript or external images for charts.
                                           - Each candidate column includes an exact histogram_token in the profile.
                                             For every semantically eligible measurement, copy its histogram_token once
                                             on its own line inside the Outlier Histograms section, before Actions Before Analysis.
                                             The token is replaced by a complete measured stacked histogram with axes,
                                             a centered legend, and a caption containing counts, percentage and IQR bounds.
                                           - A column without histogram_token must not have a histogram. A token only
                                             establishes that chart data exists; still exclude ambiguous features,
                                             identifiers, categorical encodings and other ineligible measurements.
                                           - Never replace a supplied token with prose, comments describing bars,
                                             a table, a hand-written SVG, or a promise to plot later.
                                           - End with an h2 section titled Actions Before Analysis. Give concise numbered treatment 
                                             recommendations: exact column, reproducible record selection, operation, parameters and verification.
                                           - Recommend mean or median imputation only for appropriate numeric features with measured nulls,
                                             specifying the finite non-null reference population.
                                           - Distinguish row deletion from value replacement.
                                           - Outliers and rare categories are not automatically errors.
                                           - If a mutation requires domain knowledge, name the exact missing decision.
                                       """
                                       + profile.model_copy(
                                                             update = {
                                                                        "columns_profile": columns,
                                                                        "preview_columns": [],
                                                                        "preview_rows": [],
                                                                      }
                                                           )
                                                .model_dump_json()
                                     ),
                       expected_output = ("""
                                              A complete standalone HTML audit with the exact supplied histogram_token
                                              for each eligible continuous numeric measurement with detected IQR outliers.
                                              Python replaces the tokens with measured static SVG histograms.
                                              Do not write SVG yourself. End with Actions Before Analysis.
                                          """),
                       agent = agent,
                       guardrail = validate_generated_report,
                       guardrail_max_retries = 2,
                     )

    crew = Crew(
                 agents = [agent],
                 tasks = [
                           audit_task,
                         ],
                 process = Process.sequential,
                 verbose = True,
                 cache = True,
                 memory = False,
                 planning = False,
               )

    try:
        result = crew.kickoff()
    except Exception as exc:
        parser.exit(2, f'Audit failed: {exc}\n')
    report = insert_histograms(result.raw, histogram_charts)
    valid, report = validate_html_report(
                                          TaskOutput(
                                                      description = "Rendered report",
                                                      expected_output = "HTML",
                                                      raw = report,
                                                      agent = agent.role,
                                                    )
                                        )
    if not valid:
        raise ValueError(report)
    destination = Path(report_path)
    destination.parent.mkdir(parents = True, exist_ok = True)
    destination.write_text(report, encoding = "utf-8")

    print("\nData-quality analysis completed.")
    print(f"HTML report: {Path(report_path).resolve()}")


if __name__ == "__main__":
    main()

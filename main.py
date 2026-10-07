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
#-----------------------------------------------------------------------------


load_dotenv() # to load the API's key from LLMs models


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
                         """,
                  backstory = """
                                  You specialize in data quality, exploratory analysis and reliable analytical pipelines.
                                  Never estimate statistics that can be obtained from a tool.
                                  Distinguish evidence from interpretation.
                                  Outliers are not automatically errors.
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
        flags=re.IGNORECASE | re.DOTALL,
    )
    if document is None:
        return (
                 False,
                 "Return a complete HTML report enclosed in <html>...</html>, "
                 "including Executive Summary, Dataset Overview, Prioritized "
                 "Findings, Recommended Treatments, Methodology and a final Actions Before Analysis section. "
                 "Do not return an explanation or a partial HTML fragment.",
               )

    html = "<!DOCTYPE html>\n" + document.group(0).strip()
    lower = html.lower()

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

    return True, html
def main() -> None:

    parser = argparse.ArgumentParser(description = ("Autonomous CrewAI agent for dataset data-quality auditing."))

    parser.add_argument(
                         "dataset_url",
                         help = ("HTTP(S) URL of a CSV, TSV, Excel, Parquet, JSON, JSONL or ZIP dataset."),
                       )

    parser.add_argument(
                         "--output",
                         default = "output/data_quality_report.html",
                         help = "Path for the generated HTML report.",
                       )

    parser.add_argument("--archive-member", help="Exact internal path of the data file to read from a ZIP archive.")
    args = parser.parse_args()

    report_path = str(
                       Path(args.output)
                     )

    agent = build_agent()

    try:
        profile = profiler_tool.run(dataset_url=args.dataset_url, archive_member=args.archive_member, include_value_examples=False)
    except ValueError as exc:
        parser.exit(2, f"Dataset error: {exc}\n")
    audit_task = Task(
        description=(
            "Produce a complete standalone HTML data-quality report from the measured "
            "profile below. Return only <html>...</html>, with inline CSS and escaped "
            "dataset paths and column names. Include Executive Summary, Dataset Overview, "
            "Prioritized Findings, Recommendations, Positive Observations, and Methodology "
            "and Limitations. Include all measured flags with column, severity and evidence. "
            "Show measured column counts in a readable table. Do not invent statistics, "
            "columns, thresholds, validity rules, images or claims of completed cleanup. "
            "End with an h2 section titled Actions Before Analysis. Give concise numbered "
            "treatment recommendations: exact column, reproducible record selection, "
            "operation, parameters and verification. Recommend mean or median imputation "
            "only for appropriate numeric features with measured nulls, specifying the "
            "finite non-null reference population. Distinguish row deletion from value "
            "replacement. Outliers and rare categories are not automatically errors. "
            "If a mutation requires domain knowledge, name the exact missing decision. "
            "Do not modify the dataset. Do not produce separate execution plans or files. "
            "Do not use code fences or return a plan instead of the report.\n\n"
            + profile.model_copy(update={
                "preview_columns": [],
                "preview_rows": [],
                "columns_profile": [
                    {key: value for key, value in column.items() if key != "outlier_histogram"}
                    for column in profile.columns_profile
                ],
            }).model_dump_json()
        ),
        expected_output="A complete standalone HTML audit ending with Actions Before Analysis.",
        agent=agent,
        guardrail=validate_html_report,
        guardrail_max_retries=2,
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
    report = result.raw
    valid, report = validate_html_report(TaskOutput(
        description="Rendered report", expected_output="HTML", raw=report,
        agent=agent.role,
    ))
    if not valid:
        raise ValueError(report)
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")

    print("\nData-quality analysis completed.")
    print(f"HTML report: {Path(report_path).resolve()}")


if __name__ == "__main__":
    main()

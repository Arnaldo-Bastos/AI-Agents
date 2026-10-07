#-----------------------------------------------------------------------------
#                                       IMPORTS
from typing import Literal
from pydantic import BaseModel, Field
#-----------------------------------------------------------------------------


Severity = Literal[
                      "critical",
                      "high",
                      "medium",
                      "low",
                      "info",
                  ]

Dimension = Literal[
                       "schema",
                       "completeness",
                       "validity",
                       "consistency",
                       "uniqueness",
                       "distribution",
                       "anomaly",
                       "categorical_quality",
                       "temporal_quality",
                       "other",
                   ]


class Finding(BaseModel):
    severity: Severity
    dimension: Dimension
    column: str | None = None
    title: str
    evidence: str = Field(description="Exact measurable evidence obtained from the profiling tool.")
    interpretation: str
    recommendation: str
    treatment_example: str | None = None
    caveat: str | None = None


class DimensionAssessment(BaseModel):
    dimension: Dimension

    status: Literal[
                       "good",
                       "attention",
                       "poor",
                       "not_assessable",
                   ]

    explanation: str


class AuditAssessment(BaseModel):
    dataset_path: str
    rows: int
    columns: int
    executive_summary: str
    dimensions: list[DimensionAssessment]
    findings: list[Finding]
    positive_observations: list[str]
    limitations: list[str]

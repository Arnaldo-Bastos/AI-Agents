#-----------------------------------------------------------------------------
#                                       IMPORTS
from __future__ import annotations
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Type
import re
import tarfile
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile
import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
#-----------------------------------------------------------------------------


# Tokens that may represent missing/unusual information.
MISSING_TOKENS = {
                   "na",
                   "n/a",
                   "null",
                   "none",
                   "nil",
                   "nan",
                   "unknown",
                   "?",
                   "-",
                   "--",
                 }

# Function to convert values in percentage format:
def pct(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 3)


def safe_nunique(series: pd.Series) -> int:
    """Count distinct values even if a column contains non-hashable objects."""
    try:
        return int(series.nunique(dropna = True))
    except TypeError:
        return int(
                    series.astype(str)
                          .nunique(dropna = True)
                  )


def pattern_signature(value: str) -> str:
    """
    Convert a concrete value into a simplified format signature.

    Examples:
        2026-01-15 -> 9999-99-99
        ABC-123    -> AAA-999
    """
    value = str(value).strip()
    value = re.sub(
                    r"[A-Za-z]",
                    "A",
                    value
                  )
    value = re.sub(
                    r"\d",
                    "9",
                    value
                  )
    return value[: 80]


class DatasetProfilerInput(BaseModel):
    archive_member: (str) | (None) = Field(
                                            default = None,
                                            description = "Exact path in a ZIP/TAR archive; separate nested archive paths with ::."
                                          )
    dataset_url: str = Field(
                              ...,
                              description = ("HTTP(S) URL of the dataset to be audited. Supported formats: CSV, TSV, headerless DATA/TEST, Excel, Parquet, JSON, JSONL and ZIP/TAR archives, including nested archives."),
                            )

    max_categories: int = Field(
                                 default = 50,
                                 ge = 5,
                                 le = 200,
                                 description = ("Maximum number of distinct values for treating a text column as low-cardinality categorical data."),
                               )

    include_value_examples: bool = Field(
                                          default = False,
                                          description = ("Whether raw unusual/category value examples may be returned. Keep False for safer handling of potentially sensitive data."),
                                        )


class DatasetProfilerResult(BaseModel):
    preview_columns: list[str] = Field(default_factory = list)
    preview_rows: list[
                        list[
                              (str)
                              | (None)
                            ]
                      ] = Field(default_factory = list)
    dataset_path: str
    rows: int
    columns: int
    memory_mb: float
    duplicate_rows: int
    duplicate_row_pct: float
    duplicate_column_names: list[str]
    null_cells: int
    blank_cells: int
    effective_missing_cells: int
    effective_missing_pct: float
    global_flags: list[dict[str, Any]]
    columns_profile: list[dict[str, Any]]


class DatasetQualityProfilerTool(BaseTool):
    name: str = "profile_dataset_quality"

    description: str = """
                            Performs a deterministic and comprehensive data-quality profile of a dataset.
                            Use this tool whenever factual measurements are needed about schema,
                            column types, missing values, blank values, duplicate rows, mixed types,
                            numeric values stored as strings, date-like strings, whitespace/case
                            inconsistencies, cardinality, rare categories, infinite values,
                            constant columns, near-constant columns and numeric outliers.
                            Never estimate these metrics yourself. Use this tool.
                       """

    args_schema: Type[BaseModel] = DatasetProfilerInput

    @staticmethod
    def _download_url(dataset_url: str) -> str:
      """Resolve public Sciebo share pages to their download endpoint."""
      parsed = urlparse(dataset_url)
      host = (
               (parsed.hostname)
               or ('')
             ).lower()
      if (host == 'sciebo.de' or host.endswith('.sciebo.de')) and re.fullmatch(r'/s/[^/]+/?', parsed.path):
        return urlunparse(
                           parsed._replace(
                                            path = parsed.path.rstrip('/') + '/download',
                                            query = '',
                                            fragment = ''
                                          )
                         )
      return dataset_url

    def _download_dataset(self, dataset_url: str) -> tuple[bytes, str]:
      parsed = urlparse(dataset_url)
      if parsed.scheme not in {"http", "https"}:
        raise ValueError("dataset_url must be an HTTP(S) URL, not a local file path.")

      request = Request(
                         self._download_url(dataset_url),
                         headers = {"User-Agent": "data-quality-profiler/1.0"}
                       )
      with urlopen(request, timeout=60) as response:
        content_type = response.headers.get_content_type()
        first = response.read(1024)
        prefix = (
                   first.lstrip(b'\xef\xbb\xbf \t\r\n')
                        .lower()
                 )
        if content_type in {'text/html', 'application/xhtml+xml'} or prefix.startswith((b'<!doctype html', b'<html')):
          raise ValueError(
                            f"Download returned an HTML page at {response.geturl()}. Use a public direct download link; check whether the share requires a password or login."
                          )
        return first + response.read(), content_type

    @staticmethod
    def _archive_member(content: bytes, archive_member: str | None = None,
                        _prefix: str = "", _depth: int = 0) -> tuple[bytes, str] | None:
      if _depth >= 8:
        raise ValueError("Archive nesting exceeds the supported limit of 8 levels.")
      try:
        archive = ZipFile(BytesIO(content))
        archive_kind = "ZIP"
      except BadZipFile:
        try:
          archive = tarfile.open(fileobj = BytesIO(content), mode = "r:*")
          archive_kind = "TAR"
        except tarfile.TarError:
          archive = None
      if archive is None:
        if archive_member is not None:
          raise ValueError(
                            "The response is not a valid ZIP/TAR archive. Use an archive download URL with --archive-member; for a direct data file, omit --archive-member."
                          ) from None
        return None

      supported = {
                    ".csv",
                    ".tsv",
                    ".txt",
                    ".data",
                    ".test",
                    ".xlsx",
                    ".xls",
                    ".parquet",
                    ".json",
                    ".jsonl",
                    ".ndjson"
                  }
      def is_archive(name: str) -> bool:
        return name.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz"))

      with archive:
        if archive_kind == "ZIP":
          entries = [
                      (member.filename, member)
                      for member in archive.infolist()
                      if not member.is_dir()
                    ]
        else:
          # Read regular files in memory; never extract paths or follow links.
          entries = [
                      (member.name, member)
                      for member in archive.getmembers()
                      if member.isfile()
                    ]
        members = [
                    (name, member)
                    for name, member in entries
                    if (
                         (Path(name).suffix.lower() in supported)
                         or (is_archive(name))
                       )
                    and (not name.startswith('__MACOSX/'))
                  ]
        nested_selection = None
        if archive_member is not None:
          selected, separator, remainder = archive_member.partition("::")
          nested_selection = remainder if separator else None
          matches = [
                      (name, member)
                      for name, member in members
                      if name == selected
                    ]
          if len(matches) != 1:
            names = ", ".join(_prefix + name for name, _ in members)
            raise ValueError(
                              f"{archive_kind} member must identify exactly one supported file: {archive_member}. Available files: {names or 'none'}"
                            )
          name, member = matches[0]
        else:
          # Text files may be documentation or instrument-specific recordings.
          # Require explicit selection rather than guessing their tabular schema.
          candidates = [
                         (name, member)
                         for name, member in members
                         if Path(name).suffix.lower() != '.txt'
                       ]
          if len(candidates) != 1:
            names = ", ".join(
                               _prefix + name
                               for name, _ in (candidates)
                               or (members)
                             )
            raise ValueError(
                              f'{archive_kind} requires file selection. Use --archive-member "{_prefix}internal/path.csv". Available files: {names or "none"}'
                            )
          name, member = candidates[0]
        if archive_kind == "ZIP":
          source = archive.read(member)
        else:
          with archive.extractfile(member) as stream:
            source = stream.read()
        full_name = _prefix + name
        if is_archive(name):
          result = DatasetQualityProfilerTool._archive_member(
                                                               source,
                                                               nested_selection,
                                                               full_name + "::",
                                                               _depth + 1
                                                             )
          if result is None:
            raise ValueError(f"Invalid nested ZIP/TAR archive: {full_name}")
          return result
        if nested_selection is not None:
          raise ValueError(f"Cannot select a nested member inside a data file: {full_name}")
        return source, full_name

    def _load_dataset(self, dataset_url: str, archive_member: str | None = None) -> pd.DataFrame:
      content, content_type = self._download_dataset(dataset_url)
      if not content:
        raise ValueError("The dataset URL returned an empty response.")
      prefix = (
                 content[: 1024].lstrip(b'\xef\xbb\xbf \t\r\n')
                                .lower()
               )
      if content_type in {'text/html', 'application/xhtml+xml'} or prefix.startswith((b'<!doctype html', b'<html')):
        raise ValueError(
                          "The dataset URL returned an HTML page, not a data file. Use the direct download link; a dataset landing page or login page cannot be analyzed."
                        )
      member = self._archive_member(content, archive_member)
      if member is None and (urlparse(dataset_url).path.lower().endswith(('.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz')) or content_type in {'application/zip', 'application/x-zip-compressed', 'application/x-tar'}):
        raise ValueError(
                          "The URL indicates a ZIP/TAR archive, but the response is not valid. Check the download link or whether the archive is incomplete."
                        )
      source = member[0] if member else content
      filename = member[1] if member else Path(urlparse(dataset_url).path).name
      suffix = Path(filename).suffix.lower()

      if not suffix and content_type == "text/tab-separated-values":
        suffix = ".tsv"

      if suffix in {".csv", ".tsv", ".txt"}:
        # Python's CSV engine sniffs comma, semicolon, tab and other common delimiters.
        return pd.read_csv(
                            BytesIO(source),
                            sep = None,
                            engine = "python",
                            keep_default_na = False
                          )

      if suffix in {".data", ".test"}:
        # UCI-style comma-delimited files have no header row.
        df = pd.read_csv(
                          BytesIO(source),
                          header = None,
                          keep_default_na = False
                        )
        df.columns = [f"column_{index + 1}" for index in range(len(df.columns))]
        return df

      if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(BytesIO(source))

      if suffix == ".parquet":
        return pd.read_parquet(BytesIO(source))

      if suffix == ".json":
        return pd.read_json(BytesIO(source))

      if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(BytesIO(source), lines = True)

      raise ValueError(f"Unsupported dataset format in URL: {filename or content_type}")

    def _run(
                self,
                dataset_url: str,
                max_categories: int = 50,
                include_value_examples: bool = False,
                archive_member: str | None = None,
            ) -> DatasetProfilerResult:
        df = self._load_dataset(dataset_url, archive_member)

        rows = len(df)
        columns = len(df.columns)

        memory_mb = round(
                           df.memory_usage(deep = True)
                             .sum() / 1024 ** 2,
                           3,
                         )

        # -------------------------------
        # Dataset-level checks
        # -------------------------------
        try:
            duplicate_rows = int(
                                  df.duplicated()
                                    .sum()
                                )
        except TypeError:
            # Fallback for columns containing lists/dicts.
            duplicate_rows = int(
                                  df.astype(str)
                                    .duplicated()
                                    .sum()
                                )

        duplicate_column_names = [str(column) for column in df.columns[df.columns.duplicated()]]

        global_flags: list[dict[str, Any]] = []

        if rows == 0:
            global_flags.append(
                                 {
                                   "code": "EMPTY_DATASET",
                                   "severity_hint": "critical",
                                   "evidence": "The dataset contains zero rows.",
                                 }
                               )

        if duplicate_rows > 0:
            global_flags.append(
                                 {
                                   "code": "DUPLICATE_ROWS",
                                   "severity_hint": "medium",
                                   "evidence": (f"{duplicate_rows} duplicated rows " f"({pct(duplicate_rows, rows)}%)."),
                                 }
                               )

        if duplicate_column_names:
            global_flags.append(
                                 {
                                   "code": "DUPLICATE_COLUMN_NAMES",
                                   "severity_hint": "high",
                                   "evidence": ("Duplicated column names: " f"{duplicate_column_names}"),
                                 }
                               )

        # -------------------------------
        # Column-level checks
        # -------------------------------
        columns_profile: list[dict[str, Any]] = []
        total_null_cells = 0
        total_blank_cells = 0

        for position, column in enumerate(df.columns):
            series = df.iloc[:, position]
            column_name = str(column)

            profile: dict[str, Any] = {
                                        "column_position": position,
                                        "column": column_name,
                                        "pandas_dtype": str(series.dtype),
                                        "flags": [],
                                      }

            null_count = int(
                              series.isna()
                                    .sum()
                            )
            non_null = series.dropna()
            non_null_count = len(non_null)
            unique_count = safe_nunique(series)

            profile.update(
                            {
                              "non_null_count": non_null_count,
                              "null_count": null_count,
                              "null_pct": pct(null_count, rows),
                              "unique_count": unique_count,
                              "unique_pct": pct(unique_count, non_null_count),
                            }
                          )

            total_null_cells += null_count

            if null_count > 0:
                severity = ("high" if profile["null_pct"] >= 40 else "medium")

                profile["flags"].append(
                                         {
                                           "code": "NULL_VALUES",
                                           "severity_hint": severity,
                                           "evidence": (f"{null_count} null values " f"({profile['null_pct']}%)."),
                                         }
                                       )

            # Runtime type mix
            python_types = Counter(
                                    type(value).__name__
                                    for value in non_null
                                  )
            profile["python_types"] = dict(python_types.most_common())

            if len(python_types) > 1:
                profile["flags"].append(
                                         {
                                           "code": "MIXED_PYTHON_TYPES",
                                           "severity_hint": "medium",
                                           "evidence": ("Multiple runtime types detected: " f"{dict(python_types)}"),
                                         }
                                       )

            # Constant / near-constant
            if non_null_count > 0:
                try:
                    value_counts = non_null.value_counts(dropna = True)
                except TypeError:
                    value_counts = (
                                     non_null.astype(str)
                                             .value_counts(dropna = True)
                                   )

                top_frequency_share = (value_counts.iloc[0] / non_null_count if len(value_counts) else 0)

                profile["top_frequency_pct"] = round(
                                                      top_frequency_share * 100,
                                                      3,
                                                    )

                if unique_count == 1:
                    profile["flags"].append(
                                             {
                                               "code": "CONSTANT_COLUMN",
                                               "severity_hint": "medium",
                                               "evidence": ("Only one distinct non-null value exists."),
                                             }
                                           )
                elif top_frequency_share >= 0.99:
                    profile["flags"].append(
                                             {
                                               "code": "NEAR_CONSTANT_COLUMN",
                                               "severity_hint": "low",
                                               "evidence": ("Most frequent value represents " f"{profile['top_frequency_pct']}% " "of non-null records."),
                                             }
                                           )

            # -------------------------------
            # Text checks
            # -------------------------------
            is_textual = (
                           (pd.api.types.is_object_dtype(series.dtype))
                           or (pd.api.types.is_string_dtype(series.dtype))
                         )

            if is_textual and non_null_count:
                text = non_null.astype(str)
                stripped = text.str.strip()
                normalized = stripped.str.casefold()

                blank_count = int(
                                   stripped.eq("")
                                           .sum()
                                 )
                total_blank_cells += blank_count

                profile["blank_count"] = blank_count
                profile["blank_pct"] = pct(blank_count, rows)

                if blank_count:
                    profile["flags"].append(
                                             {
                                               "code": "BLANK_STRINGS",
                                               "severity_hint": "medium",
                                               "evidence": (f"{blank_count} blank/empty strings " f"({profile['blank_pct']}%)."),
                                             }
                                           )

                whitespace_count = int(
                                        text.ne(stripped)
                                            .sum()
                                      )
                profile["leading_trailing_whitespace_count"] = (whitespace_count)

                if whitespace_count:
                    profile["flags"].append(
                                             {
                                               "code": "WHITESPACE_INCONSISTENCY",
                                               "severity_hint": "low",
                                               "evidence": (f"{whitespace_count} values contain leading or trailing whitespace."),
                                             }
                                           )

                placeholder_count = int(
                                         normalized.isin(MISSING_TOKENS)
                                                   .sum()
                                       )
                profile["potential_missing_token_count"] = (placeholder_count)

                if placeholder_count:
                    profile["flags"].append(
                                             {
                                               "code": "POTENTIAL_MISSING_TOKENS",
                                               "severity_hint": "medium",
                                               "evidence": (f"{placeholder_count} values use tokens commonly associated with missing data."),
                                             }
                                           )

                original_unique = stripped.nunique()
                normalized_unique = normalized.nunique()
                collisions = original_unique - normalized_unique

                profile["case_or_spacing_collisions"] = int(collisions)

                if collisions > 0:
                    profile["flags"].append(
                                             {
                                               "code": "CASE_OR_SPACING_INCONSISTENCY",
                                               "severity_hint": "low",
                                               "evidence": (f"{collisions} distinct representations collapse after trim + case normalization."),
                                             }
                                           )

                lengths = stripped.str.len()
                profile["string_length"] = {
                                             "min": int(lengths.min()),
                                             "median": float(lengths.median()),
                                             "max": int(lengths.max()),
                                           }

                patterns = (
                             stripped[stripped.ne("")].head(5000)
                                                      .map(pattern_signature)
                                                      .value_counts()
                                                      .head(10)
                           )
                profile["top_format_patterns"] = patterns.to_dict()

                candidates = stripped[
                                       (~stripped.eq(""))
                                       & (~normalized.isin(MISSING_TOKENS))
                                     ]

                # Numeric represented as text
                if len(candidates):
                    numeric_parsed = pd.to_numeric(
                                                    candidates,
                                                    errors = "coerce",
                                                  )
                    numeric_like_ratio = (
                                           numeric_parsed.notna()
                                                         .mean()
                                         )

                    profile["numeric_like_pct"] = round(
                                                         numeric_like_ratio * 100,
                                                         3,
                                                       )

                    if numeric_like_ratio >= 0.90:
                        profile["flags"].append(
                                                 {
                                                   "code": "NUMERIC_STORED_AS_TEXT",
                                                   "severity_hint": "medium",
                                                   "evidence": (f"{profile['numeric_like_pct']}% of meaningful values can be parsed as numeric."),
                                                 }
                                               )

                # Date represented as text
                date_candidates = candidates[
                                              candidates.str.contains(
                                                                       r"[-/:]",
                                                                       regex = True,
                                                                       na = False,
                                                                     )
                                            ].head(1000)

                if len(date_candidates) >= 5:
                    parsed_dates = pd.to_datetime(
                                                   date_candidates,
                                                   errors = "coerce",
                                                   format = "mixed",
                                                 )
                    date_ratio = (
                                   parsed_dates.notna()
                                               .mean()
                                 )
                    profile["date_like_pct"] = round(
                                                      date_ratio * 100,
                                                      3,
                                                    )

                    if date_ratio >= 0.80:
                        profile["flags"].append(
                                                 {
                                                   "code": "DATE_STORED_AS_TEXT",
                                                   "severity_hint": "medium",
                                                   "evidence": (f"{profile['date_like_pct']}% of date-looking values parse successfully as dates."),
                                                 }
                                               )

                # Rare categories
                if 1 < unique_count <= max_categories and len(candidates):
                    category_counts = candidates.value_counts()
                    category_frequency = (category_counts / len(candidates))
                    rare_categories = category_frequency[category_frequency < 0.01]

                    profile["rare_category_count"] = int(
                                                          len(rare_categories)
                                                        )

                    if len(rare_categories):
                        profile["flags"].append(
                                                 {
                                                   "code": "RARE_CATEGORIES",
                                                   "severity_hint": "info",
                                                   "evidence": (f"{len(rare_categories)} categories occur in less than 1% of records."),
                                                 }
                                               )

                        if include_value_examples:
                            profile["rare_category_examples"] = [str(x) for x in rare_categories.index[: 10]]

                # High-cardinality text
                if (
                    unique_count > 50
                    and profile["unique_pct"] >= 90
                ):
                    profile["flags"].append(
                                             {
                                               "code": "HIGH_CARDINALITY",
                                               "severity_hint": "info",
                                               "evidence": (f"{profile['unique_pct']}% of non-null values are unique."),
                                             }
                                           )

            # -------------------------------
            # Numeric checks
            # -------------------------------
            is_numeric = (
                           (pd.api.types.is_numeric_dtype(series))
                           and (not pd.api.types.is_bool_dtype(series))
                         )

            if is_numeric:
                numeric = pd.to_numeric(series, errors = "coerce")
                numeric_array = numeric.astype("float64")

                inf_count = int(
                                 np.isinf(numeric_array.to_numpy())
                                   .sum()
                               )
                profile["infinite_count"] = inf_count

                if inf_count:
                    profile["flags"].append(
                                             {
                                               "code": "INFINITE_VALUES",
                                               "severity_hint": "high",
                                               "evidence": (f"{inf_count} infinite numeric values."),
                                             }
                                           )

                finite = (
                           numeric_array.replace(
                                                  [
                                                    np.inf,
                                                    -np.inf
                                                  ],
                                                  np.nan
                                                )
                                        .dropna()
                         )

                if len(finite):
                    q1 = float(finite.quantile(0.25))
                    median = float(finite.median())
                    q3 = float(finite.quantile(0.75))
                    iqr = q3 - q1

                    profile["numeric_summary"] = {
                                                   "min": float(finite.min()),
                                                   "q1": q1,
                                                   "median": median,
                                                   "q3": q3,
                                                   "max": float(finite.max()),
                                                   "mean": float(finite.mean()),
                                                   "std": (float(finite.std()) if len(finite) > 1 else 0.0),
                                                 }

                    # Do not apply the IQR rule to obviously discrete
                    # low-cardinality numeric columns.
                    if iqr > 0 and finite.nunique() >= 10:
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr

                        outlier_mask = (
                                         (finite < lower)
                                         | (finite > upper)
                                       )
                        outlier_count = int(outlier_mask.sum())

                        profile["iqr_outliers"] = {
                                                    "lower_bound": float(lower),
                                                    "upper_bound": float(upper),
                                                    "count": outlier_count,
                                                    "pct": pct(outlier_count, len(finite)),
                                                  }

                        if outlier_count:
                            # Aggregated chart data only; never expose raw rows.
                            bin_count = min(60, max(10, int(np.ceil(np.sqrt(len(finite))))))
                            counts, edges = np.histogram(finite.to_numpy(), bins = bin_count)
                            outlier_counts, _ = np.histogram(
                                                              finite[outlier_mask].to_numpy(),
                                                              bins = edges,
                                                            )
                            profile["outlier_histogram"] = {
                                                             "counts": counts.tolist(),
                                                             "edges": edges.tolist(),
                                                             "outlier_counts": outlier_counts.tolist(),
                                                           }
                            profile["flags"].append(
                                                     {
                                                       "code": "IQR_OUTLIERS",
                                                       "severity_hint": "info",
                                                       "evidence": (f"{outlier_count} values " f"({profile['iqr_outliers']['pct']}%) fall outside the 1.5 * IQR bounds."),
                                                     }
                                                   )

            columns_profile.append(profile)

        effective_missing_cells = (total_null_cells + total_blank_cells)
        total_cells = rows * columns

        return DatasetProfilerResult(
                                      preview_columns = [str(column) for column in df.columns],
                                      preview_rows = [
                                                       [
                                                         None if (pd.api.types.is_scalar(value))
                                                         and (pd.isna(value)) else str(value)
                                                         for value in row
                                                       ]
                                                       for row in df.head(5)
                                                                    .itertuples(index = False, name = None)
                                                     ],
                                      dataset_path = dataset_url,
                                      rows = rows,
                                      columns = columns,
                                      memory_mb = memory_mb,
                                      duplicate_rows = duplicate_rows,
                                      duplicate_row_pct = pct(duplicate_rows, rows),
                                      duplicate_column_names = duplicate_column_names,
                                      null_cells = total_null_cells,
                                      blank_cells = total_blank_cells,
                                      effective_missing_cells = effective_missing_cells,
                                      effective_missing_pct = pct(
                                                                   effective_missing_cells,
                                                                   total_cells,
                                                                 ),
                                      global_flags = global_flags,
                                      columns_profile = columns_profile,
                                    )

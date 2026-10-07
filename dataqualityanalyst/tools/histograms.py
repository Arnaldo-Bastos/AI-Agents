"""Render measured histogram data without asking the language model to draw bars."""

from html import escape
import math
import re


def render_histogram(column: dict) -> str:
    histogram = column["outlier_histogram"]
    edges, counts, outliers = (histogram[k] for k in ("edges", "counts", "outlier_counts"))
    if (not counts or len(edges) != len(counts) + 1 or len(outliers) != len(counts)
            or not all(math.isfinite(v) for v in edges + counts + outliers)
            or not all(a < b for a, b in zip(edges, edges[1:]))
            or not all(0 <= o <= c for c, o in zip(counts, outliers))
            or max(counts) <= 0 or sum(outliers) != column["iqr_outliers"]["count"]):
        raise ValueError(f"Invalid measured histogram for {column['column']}")
    name = escape(column["column"], quote=True)
    xmin, xmax, ymax = edges[0], edges[-1], max(counts)
    x = lambda v: 90 + 670 * (v - xmin) / (xmax - xmin)
    height = lambda c: 230 * c / ymax
    parts = [f'<figure data-measured-histogram="true" style="margin:24px 0">'
             f'<h3>{name}</h3><svg xmlns="http://www.w3.org/2000/svg" '
             f'viewBox="0 0 800 400" width="800" height="400" '
             f'style="max-width:100%;height:auto" role="img" aria-label="Histogram: {name}">'
             f'<title>Distribution of {name}</title>'
             '<g font-family="Arial,sans-serif" font-size="12" fill="#334155">'
             '<rect x="250" y="36" width="16" height="16" fill="#2563eb"/>'
             '<text x="275" y="49">Non-outliers</text>'
             '<rect x="445" y="36" width="16" height="16" fill="#ea580c"/>'
             '<text x="470" y="49">Outliers</text>']
    for i in range(5):
        value = ymax * i / 4
        y = 310 - height(value)
        parts.append(f'<line x1="90" y1="{y:.6f}" x2="760" y2="{y:.6f}" stroke="#e2e8f0"/>'
                     f'<text x="78" y="{y + 4:.6f}" text-anchor="end">{value:.4g}</text>')
        value = xmin + (xmax - xmin) * i / 4
        xpos = x(value)
        parts.append(f'<line x1="{xpos:.6f}" y1="310" x2="{xpos:.6f}" y2="316" stroke="#475569"/>'
                     f'<text x="{xpos:.6f}" y="337" text-anchor="middle">{value:.4g}</text>')
    for i, (count, outlier) in enumerate(zip(counts, outliers)):
        left, width = x(edges[i]), x(edges[i + 1]) - x(edges[i])
        for label, portion, top, color in (
            ("Non-outliers", count - outlier, 310 - height(count - outlier), "#2563eb"),
            ("Outliers", outlier, 310 - height(count), "#ea580c"),
        ):
            if portion:
                parts.append(f'<rect data-bin="{i}" data-series="{label}" x="{left:.6f}" '
                             f'y="{top:.6f}" width="{width:.6f}" height="{height(portion):.6f}" '
                             f'fill="{color}" stroke="#fff" stroke-width="0.3">'
                             f'<title>{edges[i]:.6g} to {edges[i + 1]:.6g}: {portion} {label}</title></rect>')
    parts.append('<path d="M90 80V310H760" fill="none" stroke="#475569"/>'
                 '<text x="425" y="375" text-anchor="middle">Value (see column heading)</text>'
                 '<text x="24" y="195" transform="rotate(-90 24 195)" text-anchor="middle">Frequency</text>'
                 '</g></svg>')
    bounds = column["iqr_outliers"]
    note = " Some outlier portions are smaller than one SVG unit at this scale." if any(
        0 < height(o) < 1 for o in outliers
    ) else ""
    parts.append(f'<figcaption>Finite observations: {sum(counts)}; outliers: {sum(outliers)} '
                 f'({bounds["pct"]}%); IQR bounds: [{bounds["lower_bound"]:.6g}, '
                 f'{bounds["upper_bound"]:.6g}].{note}</figcaption></figure>')
    return "\n".join(parts)


def prepare_histograms(columns: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Provide tokens for candidates; leave ambiguous measurement semantics to the agent."""
    prepared, charts = [], {}
    excluded = re.compile(r"(?:^|_)(?:id|code|rank|lat|latitude|lon|lng|longitude|date|time|timestamp|"
                          r"zip|postal|phone|cpf|ssn|cc_num)(?:_|$)", re.IGNORECASE)
    for column in columns:
        item = {k: v for k, v in column.items() if k != "outlier_histogram"}
        normalized = re.sub(r"([a-z])([A-Z])", r"\1_\2", column["column"])
        normalized = re.sub(r"[^\w]+", "_", normalized)
        if column.get("outlier_histogram") and not excluded.search(normalized):
            token = f"<!-- MEASURED_HISTOGRAM_{len(charts)} -->"
            charts[token] = render_histogram(column)
            item["histogram_token"] = token
        prepared.append(item)
    return prepared, charts


def insert_histograms(html: str, charts: dict[str, str]) -> str:
    for token in re.findall(r"<!--\s*MEASURED_HISTOGRAM_\d+\s*-->", html):
        if token not in charts:
            raise ValueError("Use only the exact histogram_token supplied in the profile.")
        if html.count(token) != 1:
            raise ValueError("Include each selected histogram_token exactly once.")
        html = html.replace(token, charts[token])
    return html

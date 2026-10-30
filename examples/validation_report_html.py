"""Small HTML page builder for the validation reports of
``examples/ashrae140_bestest/`` and ``examples/iso52016_bestest/``.

Pages are self-contained except for plotly.js, loaded from its CDN at the
exact version of the installed plotly package.
"""

from __future__ import annotations

import html
from pathlib import Path

import plotly.graph_objects as go
import plotly.offline

PLOTLY_JS = f"https://cdn.plot.ly/plotly-{plotly.offline.get_plotlyjs_version()}.min.js"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

COLORS = {"heating": "#b42318", "cooling": "#0f609b", "temperature": "#956000",
          "reference": "#5b6670", "outdoor": "#8a97a3"}

_CSS = """
:root { --ink:#1d252c; --muted:#5b6670; --line:#d8dee4; --soft:#f6f8fa;
        --blue:#0f609b; --green:#1f7a4d; --amber:#956000; --red:#b42318; --bg:#ffffff; }
@media (prefers-color-scheme: dark) {
  :root { --ink:#e6eaee; --muted:#9aa6b2; --line:#33404c; --soft:#1b232b;
          --blue:#5aa9e6; --green:#4cc38a; --amber:#e0a33a; --red:#f07167; --bg:#11171d; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width:1120px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:26px; margin:0 0 4px; }
h2 { font-size:19px; margin:32px 0 10px; padding-bottom:6px; border-bottom:1px solid var(--line); }
h3 { font-size:16px; margin:20px 0 8px; }
p, li { color:var(--ink); }
.muted { color:var(--muted); }
a { color:var(--blue); }
.subtitle { color:var(--muted); margin:0 0 16px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:12px; margin:16px 0; }
.card { border:1px solid var(--line); border-radius:8px; padding:12px 14px; background:var(--soft); }
.card .label { color:var(--muted); font-size:13px; }
.card .value { font-size:22px; font-weight:600; margin-top:2px; }
.card .note { color:var(--muted); font-size:12px; margin-top:2px; }
.badge { display:inline-block; padding:1px 8px; border-radius:999px; font-size:12px; font-weight:600;
         border:1px solid currentColor; }
.pass { color:var(--green); } .fail { color:var(--red); } .info { color:var(--muted); }
.table-wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; margin:8px 0 16px; font-size:14px; }
th, td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:right; white-space:nowrap; }
th { color:var(--muted); font-weight:600; background:var(--soft); }
th:first-child, td:first-child { text-align:left; }
th:nth-child(2), td:nth-child(2) { text-align:left; white-space:normal; min-width:180px; }
td .badge { margin-left:4px; }
.verdict { white-space:normal; display:inline-block; min-width:240px; text-align:left; }
.callout { border-left:4px solid var(--amber); background:var(--soft); padding:10px 14px; margin:12px 0; }
.chart { width:100%; min-height:340px; }
footer { color:var(--muted); font-size:13px; margin-top:40px; }
"""


def esc(value) -> str:
    return html.escape(str(value))


def badge(status: str, text: str | None = None) -> str:
    cls = {"PASS": "pass", "FAIL": "fail"}.get(status, "info")
    return f'<span class="badge {cls}">{esc(text or status)}</span>'


def cards(items) -> str:
    """items: iterable of (label, value, note)."""
    out = ['<div class="cards">']
    for label, value, note in items:
        out.append(f'<div class="card"><div class="label">{esc(label)}</div>'
                   f'<div class="value">{value}</div><div class="note">{note}</div></div>')
    out.append("</div>")
    return "".join(out)


def table(headers, rows, raw_columns=()) -> str:
    """rows: list of lists; columns whose index is in raw_columns are not escaped."""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{cell if i in raw_columns else esc(cell)}</td>" for i, cell in enumerate(row))
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def figure(fig: go.Figure, height: int = 360) -> str:
    fig.update_layout(height=height, margin=dict(l=50, r=20, t=40, b=40),
                      template="plotly_white", legend=dict(orientation="h", y=-0.2),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displaylogo": False, "responsive": True})


def fmt(value, digits=3) -> str:
    if value is None:
        return "—"
    try:
        if value != value:  # NaN
            return "—"
    except TypeError:
        pass
    text = f"{value:,.{digits}f}"
    return text[1:] if text.startswith("-") and float(value) == 0 or text.lstrip("-").strip("0.,") == "" and text.startswith("-") else text


def page(title: str, subtitle: str, body: str, footer: str = "") -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<script src="{PLOTLY_JS}"></script>
<style>{_CSS}</style></head>
<body><main>
<h1>{esc(title)}</h1>
<p class="subtitle">{subtitle}</p>
{body}
<footer>{footer}</footer>
</main></body></html>
"""


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path

import os
import re
import json
import requests
import pandas as pd
import plotly.graph_objects as go
from dash import (Dash, html, dcc, dash_table,
                  Input, Output, State, ALL, callback_context, no_update)
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import base64
import io

# ── Constants ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL      = "claude-sonnet-4-6"

COLORS = {
    "red":    "#b5493f",   # rust
    "blue":   "#2a6f6f",   # accent teal
    "purple": "#6d4c6f",   # plum
    "teal":   "#1f4f4f",   # deep teal
    "coral":  "#c17a4a",   # burnt gold
    "gray":   "#6b6259",   # muted ink
    "green":  "#5c7a52",   # sage
    "pink":   "#a15c6b",   # dusty rose
    "amber":  "#d9a441",   # gold
}
AGE_COLORS   = [COLORS["coral"], COLORS["purple"], COLORS["teal"]]
GROUP_COLORS = [COLORS["red"],   COLORS["blue"],   COLORS["gray"], COLORS["green"]]
SEX_COLORS   = [COLORS["blue"],  COLORS["pink"]]
RACE_COLORS  = [COLORS["coral"], COLORS["blue"],   COLORS["gray"], COLORS["green"], COLORS["purple"]]

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_csv_candidates = sorted(
    [f for f in os.listdir(_APP_DIR) if f.endswith(".csv")],
    key=lambda f: os.path.getmtime(os.path.join(_APP_DIR, f)),
    reverse=True,
)
DEFAULT_CSV = os.path.join(_APP_DIR, _csv_candidates[0]) if _csv_candidates else None


def get_chart_layout(theme="light"):
    dark = theme == "dark"
    text_color = "#a49c8c" if dark else "#6b6259"
    grid_color  = "#3a423d" if dark else "#dfd8c8"
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Helvetica Neue', Arial, sans-serif", size=12, color=text_color),
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return base, grid_color, dark


# ── Data helpers ──────────────────────────────────────────────────────────────
def parse_age(s):
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else None


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["Age"] = df["Missing Age"].apply(parse_age)

    def age_group(a):
        if a is None: return "Unknown"
        if a < 18:    return "Under 18"
        if a < 30:    return "18–29"
        if a < 50:    return "30–49"
        return "50+"

    def child_group(a):
        if a is None or a >= 18: return "Adult"
        if a <= 5:               return "0–5"
        if a <= 12:              return "6–12"
        return "13–17"

    df["Age Group"]    = df["Age"].apply(age_group)
    df["Child Group"]  = df["Age"].apply(child_group)
    df["DLC"]          = pd.to_datetime(df["DLC"], errors="coerce")
    df["Year Missing"] = df["DLC"].dt.year
    df["Days Missing"] = (pd.Timestamp.today() - df["DLC"]).dt.days
    return df


def load_csv(path: str) -> pd.DataFrame:
    return enrich(pd.read_csv(path, encoding="utf-8-sig"))


def parse_uploaded(contents: str, filename: str) -> pd.DataFrame:
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return enrich(pd.read_csv(io.StringIO(decoded.decode("utf-8-sig"))))


# ── Claude helpers ────────────────────────────────────────────────────────────
def ask_claude(system: str, user: str, max_tokens: int = 1000) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY environment variable is not set."
    try:
        messages = [{"role": "user", "content": user}]
        texts = []
        for _ in range(5):
            resp = requests.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": CLAUDE_MODEL,
                    "max_tokens": max_tokens,
                    "system": system,
                    "tools": [{"type": "web_search_20260209", "name": "web_search"}],
                    "messages": messages,
                },
                timeout=60,
            )
            data = resp.json()
            if "error" in data:
                return f"Claude API error: {data['error'].get('message', data['error'])}"
            content    = data.get("content", [])
            stop_reason = data.get("stop_reason")
            texts = [b["text"] for b in content if b.get("type") == "text"]
            if stop_reason != "pause_turn":
                return "\n".join(texts) if texts else "(no response)"
            messages.append({"role": "assistant", "content": content})
        return "\n".join(texts) if texts else "(max continuations reached)"
    except Exception as e:
        return f"Error calling Claude API: {e}"


def news_search_prompt(row: dict) -> str:
    return (
        f"Search the web for any news coverage of this missing child case:\n"
        f"Name: {row.get('Legal First Name','')} {row.get('Legal Last Name','')}\n"
        f"Age at disappearance: {row.get('Missing Age','')}\n"
        f"City: {row.get('City','')}, {row.get('County','')} County, CA\n"
        f"Date last seen: {row.get('DLC','')}\n"
        f"Case: {row.get('Case Number','')}\n\n"
        "Reply with: (1) whether you found any news coverage, (2) a brief summary if found, "
        "(3) a media attention level: High / Medium / Low / None. "
        "Be concise — 3-5 sentences max."
    )


# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    title="NamUs Dashboard",
    suppress_callback_exceptions=True,
)

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        <style>
/* ═══════════════════════════════════════════════════════════════
   Theme tokens — light (default) and dark
   Editorial / print palette to match intentional-tech.aatwal.com
═══════════════════════════════════════════════════════════════ */
:root {
    --bg:          #f6f3ec;
    --bg-card:     #ffffff;
    --bg-surface:  #efe9dc;
    --bg-input:    #ffffff;
    --bg-chat:     #efe9dc;
    --border:      #dfd8c8;
    --text:        #1c2b33;
    --text-muted:  #6b6259;
    --shadow:      0 2px 10px rgba(28,43,51,0.06);
    --shadow-md:   0 4px 16px rgba(28,43,51,0.10);
    --radius:      2px;
    --radius-sm:   2px;
    --accent:      #2a6f6f;
    --accent-hl:   rgba(42,111,111,0.08);
    --blue:        #2a6f6f;
    --gold:        #d9a441;
    --rust:        #b5493f;
    --grid:        #dfd8c8;
    --font-serif:  "Iowan Old Style", "Palatino Linotype", Georgia, serif;
    --font-sans:   "Helvetica Neue", Arial, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

[data-bs-theme="dark"] {
    --bg:          #1b201e;
    --bg-card:     #242b28;
    --bg-surface:  #1f2624;
    --bg-input:    #242b28;
    --bg-chat:     #1f2624;
    --border:      #3a423d;
    --text:        #e9e4d8;
    --text-muted:  #a49c8c;
    --shadow:      0 2px 10px rgba(0,0,0,0.35);
    --shadow-md:   0 4px 16px rgba(0,0,0,0.45);
    --accent:      #4a9d9d;
    --accent-hl:   rgba(74,157,157,0.14);
    --blue:        #4a9d9d;
    --gold:        #e3b866;
    --rust:        #d17164;
    --grid:        #3a423d;
}

/* ═══════════════════════════════════════════════════════════════
   Base
═══════════════════════════════════════════════════════════════ */
body {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    transition: background-color 0.22s ease, color 0.22s ease;
    font-family: var(--font-sans);
}

h1, h2, h3.editorial, h5 {
    font-family: var(--font-serif);
}

/* ═══════════════════════════════════════════════════════════════
   Eyebrow / editorial header
═══════════════════════════════════════════════════════════════ */
.eyebrow {
    font-family: var(--font-sans);
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 700;
    margin-bottom: 8px;
}

.app-title {
    font-family: var(--font-serif);
    font-weight: 600;
    font-size: 1.85rem;
    line-height: 1.15;
    color: var(--text);
    margin: 0;
}

.app-sub {
    font-family: var(--font-sans);
    color: var(--text-muted);
    font-size: 13.5px;
    line-height: 1.5;
    max-width: 640px;
    margin: 8px 0 0;
}

/* ═══════════════════════════════════════════════════════════════
   Cards
═══════════════════════════════════════════════════════════════ */
.card {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
    border-radius: var(--radius) !important;
    transition: background-color 0.22s ease, border-color 0.22s ease, box-shadow 0.15s ease;
}

.card-body { color: var(--text) !important; }

/* ═══════════════════════════════════════════════════════════════
   Metric cards
═══════════════════════════════════════════════════════════════ */
.metric-card {
    border-radius: var(--radius) !important;
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
}

.metric-card:hover {
    box-shadow: var(--shadow-md) !important;
    border-color: var(--accent) !important;
}

/* ═══════════════════════════════════════════════════════════════
   Nav / Tabs
═══════════════════════════════════════════════════════════════ */
.nav-tabs {
    border-bottom: 2px solid var(--border) !important;
    background: transparent !important;
}

.nav-tabs .nav-link {
    color: var(--text-muted) !important;
    border: none !important;
    border-radius: 0 !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px;
    padding: 0.65rem 1.1rem;
    font-size: 13px;
    font-family: var(--font-sans);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: transparent !important;
    transition: color 0.15s ease, border-color 0.15s ease;
}

.nav-tabs .nav-link:hover { color: var(--text) !important; }

.nav-tabs .nav-link.active {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}

/* ═══════════════════════════════════════════════════════════════
   Form inputs
═══════════════════════════════════════════════════════════════ */
.form-control {
    background-color: var(--bg-input) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-sans);
    transition: background-color 0.22s ease, border-color 0.15s ease;
}

.form-control::placeholder { color: var(--text-muted) !important; }

.form-control:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-hl) !important;
    background-color: var(--bg-input) !important;
    color: var(--text) !important;
}

.form-check-label { color: var(--text) !important; }

/* ═══════════════════════════════════════════════════════════════
   Dash Dropdown (react-select v2 / Select2 style)
═══════════════════════════════════════════════════════════════ */
.Select-control {
    background-color: var(--bg-input) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    transition: background-color 0.22s ease, border-color 0.15s ease;
}

.Select-placeholder { color: var(--text-muted) !important; }
.Select-value-label { color: var(--text) !important; }
.has-value .Select-value-label { color: var(--text) !important; }

.Select-menu-outer {
    background-color: var(--bg-card) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-md) !important;
    z-index: 9999 !important;
}

.Select-option {
    background-color: var(--bg-card) !important;
    color: var(--text) !important;
}

.Select-option.is-focused {
    background-color: var(--bg-surface) !important;
    color: var(--text) !important;
}

.Select-option.is-selected {
    background-color: var(--accent-hl) !important;
    color: var(--accent) !important;
}

.Select-arrow { border-color: var(--text-muted) transparent transparent !important; }
.Select.is-open .Select-arrow { border-color: transparent transparent var(--text-muted) !important; }

.Select-multi-value-wrapper .Select-value {
    background-color: var(--accent-hl) !important;
    border-color: var(--border) !important;
    color: var(--accent) !important;
    border-radius: 2px !important;
}

.Select-value-icon {
    border-right-color: var(--border) !important;
    color: var(--text-muted) !important;
}

.Select-value-icon:hover {
    color: var(--accent) !important;
    background: transparent !important;
}

/* ═══════════════════════════════════════════════════════════════
   DataTable
═══════════════════════════════════════════════════════════════ */
.dash-spreadsheet-container .dash-spreadsheet-inner {
    background-color: var(--bg-card) !important;
    font-family: var(--font-sans) !important;
}

.dash-spreadsheet .dash-header,
.dash-spreadsheet th {
    background-color: var(--bg-surface) !important;
    color: var(--text-muted) !important;
    border-bottom: 2px solid var(--border) !important;
    border-right: none !important;
}

.dash-spreadsheet td.dash-cell,
.dash-spreadsheet .dash-cell {
    background-color: var(--bg-card) !important;
    color: var(--text) !important;
    border-bottom: 1px solid var(--border) !important;
    border-right: none !important;
}

.dash-spreadsheet tr:hover td.dash-cell {
    background-color: var(--bg-surface) !important;
}

.dash-spreadsheet .dash-filter input {
    background-color: var(--bg-input) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}

/* ═══════════════════════════════════════════════════════════════
   Alerts
═══════════════════════════════════════════════════════════════ */
.alert-light {
    background-color: var(--bg-surface) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
}

.alert-light hr { border-color: var(--border) !important; }

/* ═══════════════════════════════════════════════════════════════
   Upload zone
═══════════════════════════════════════════════════════════════ */
.upload-zone {
    border: 1.5px dashed var(--border);
    border-radius: var(--radius-sm);
    padding: 14px;
    text-align: center;
    cursor: pointer;
    color: var(--text-muted);
    font-family: var(--font-sans);
    font-size: 13px;
    transition: border-color 0.2s ease, background-color 0.2s ease;
}

.upload-zone:hover {
    border-color: var(--accent);
    background-color: var(--accent-hl);
}

/* ═══════════════════════════════════════════════════════════════
   Chat window + bubbles
═══════════════════════════════════════════════════════════════ */
.chat-window {
    background-color: var(--bg-chat) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    height: 380px;
    overflow-y: auto;
    padding: 14px;
    margin-bottom: 10px;
    transition: background-color 0.22s ease, border-color 0.22s ease;
}

.chat-bubble {
    display: inline-block;
    padding: 9px 14px;
    border-radius: 3px;
    font-family: var(--font-sans);
    font-size: 13px;
    max-width: 85%;
    white-space: pre-wrap;
    line-height: 1.5;
}

.user-bubble {
    background-color: var(--accent);
    color: #ffffff;
}

.assistant-bubble {
    background-color: var(--bg-card);
    color: var(--text);
    border: 1px solid var(--border);
}

/* ═══════════════════════════════════════════════════════════════
   Section labels
═══════════════════════════════════════════════════════════════ */
.section-label {
    font-family: var(--font-sans);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 8px;
}

.section-idx {
    color: var(--accent);
    font-weight: 700;
    margin-right: 8px;
}

/* ═══════════════════════════════════════════════════════════════
   Theme toggle button
═══════════════════════════════════════════════════════════════ */
.theme-toggle {
    width: 34px;
    height: 34px;
    border-radius: 50% !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    color: var(--text-muted) !important;
    display: flex !important;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
    padding: 0 !important;
    font-size: 14px;
}

.theme-toggle:hover {
    color: var(--accent) !important;
    border-color: var(--accent) !important;
    box-shadow: var(--shadow-md) !important;
}

/* ═══════════════════════════════════════════════════════════════
   Buttons
═══════════════════════════════════════════════════════════════ */
.btn {
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-sans);
}

.btn-outline-secondary {
    border-color: var(--border) !important;
    color: var(--text-muted) !important;
}

.btn-outline-secondary:hover {
    background-color: var(--bg-surface) !important;
    color: var(--text) !important;
    border-color: var(--text-muted) !important;
}

.btn-primary {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
}

.btn-outline-primary {
    color: var(--accent) !important;
    border-color: var(--accent) !important;
}

.btn-outline-primary:hover {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
}

.btn-outline-danger {
    color: var(--rust) !important;
    border-color: var(--rust) !important;
}

.btn-outline-danger:hover {
    background-color: var(--rust) !important;
    border-color: var(--rust) !important;
}

.btn-outline-warning {
    color: var(--gold) !important;
    border-color: var(--gold) !important;
}

.btn-outline-warning:hover {
    background-color: var(--gold) !important;
    border-color: var(--gold) !important;
    color: #1c2b33 !important;
}

/* ═══════════════════════════════════════════════════════════════
   Slider
═══════════════════════════════════════════════════════════════ */
.dash-slider-range  { background-color: var(--accent) !important; }
.dash-slider-track  { background-color: var(--border) !important; }
.dash-slider-thumb  { border-color: var(--accent) !important; background-color: var(--bg-card) !important; }
.dash-slider-mark   { color: var(--text-muted) !important; font-family: var(--font-sans); }
.dash-slider-tooltip { font-family: var(--font-sans); }

/* ═══════════════════════════════════════════════════════════════
   Scrollbars
═══════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ═══════════════════════════════════════════════════════════════
   HR separator
═══════════════════════════════════════════════════════════════ */
hr { border-color: var(--border) !important; opacity: 1 !important; }

/* ═══════════════════════════════════════════════════════════════
   Spinner
═══════════════════════════════════════════════════════════════ */
[data-bs-theme="dark"] .text-muted { color: var(--text-muted) !important; }
        </style>
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""


# ── Layout helpers ────────────────────────────────────────────────────────────
def metric_card(label, value, color=COLORS["red"], icon="bi-person"):
    return dbc.Card(dbc.CardBody([
        html.Div([
            html.I(className=f"bi {icon} me-2", style={"color": color, "fontSize": "20px"}),
            html.Span(label, style={
                "fontSize": "11px", "fontWeight": 700,
                "textTransform": "uppercase", "letterSpacing": "0.6px",
                "color": "var(--text-muted)",
            }),
        ], className="d-flex align-items-center mb-2"),
        html.H3(str(value), style={
            "fontWeight": 700, "color": color, "margin": 0, "fontSize": "1.9rem",
        }),
    ]), className="metric-card h-100")


def section_header(title, idx=None):
    children = []
    if idx is not None:
        children.append(html.Span(f"{idx:02d}", className="section-idx"))
    children.append(title)
    return html.P(children, className="section-label")


# ── Main layout ───────────────────────────────────────────────────────────────
app.layout = dbc.Container([
    dcc.Store(id="data-store"),
    dcc.Store(id="chat-history", data=[]),
    dcc.Store(id="theme-store", data="light", storage_type="local"),

    # ── Header ────────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Div("NamUs · California", className="eyebrow"),
            html.H1("Missing Persons Dashboard", className="app-title"),
            html.P(
                "Explore active NamUs missing-persons cases across California. "
                "Filter by county, age, and race to surface patterns and check "
                "cases for media coverage.",
                className="app-sub",
            ),
        ]),
        dbc.Col(
            html.Div(
                dbc.Button(
                    html.I(id="theme-icon", className="bi bi-moon-stars-fill"),
                    id="theme-toggle-btn",
                    className="theme-toggle",
                    n_clicks=0,
                    title="Toggle dark mode",
                ),
                className="d-flex justify-content-end align-items-start h-100",
            ),
            md=2,
        ),
    ], className="py-4 mb-4",
       style={"borderBottom": "2px solid var(--text)"}),

    # ── Upload ────────────────────────────────────────────────────────────────
    dbc.Card(dbc.CardBody([
        dcc.Upload(id="upload", children=html.Div([
            html.I(className="bi bi-cloud-upload me-2"),
            "Drop a NamUs CSV here, or ",
            html.A("browse", style={"color": COLORS["blue"]}),
            f" — auto-loaded: {os.path.basename(DEFAULT_CSV) if DEFAULT_CSV else 'none found'}",
        ], className="upload-zone")),
        html.Div(id="upload-status",
                 style={"fontSize": "12px", "marginTop": "6px",
                        "color": "var(--text-muted)"}),
    ]), className="mb-3"),

    # ── Filter bar ────────────────────────────────────────────────────────────
    dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("View", style={
                    "fontSize": "11px", "fontWeight": 700,
                    "textTransform": "uppercase", "letterSpacing": "0.5px",
                    "color": "var(--text-muted)",
                }),
                dcc.RadioItems(
                    id="view-toggle",
                    options=[
                        {"label": " All cases",           "value": "all"},
                        {"label": " Children only (< 18)", "value": "children"},
                    ],
                    value="children", inline=True,
                    style={"fontSize": "13px"},
                    inputStyle={"marginRight": "4px", "marginLeft": "8px"},
                ),
            ], md=4),
            dbc.Col([
                html.Label("County", style={
                    "fontSize": "11px", "fontWeight": 700,
                    "textTransform": "uppercase", "letterSpacing": "0.5px",
                    "color": "var(--text-muted)",
                }),
                dcc.Dropdown(id="county-filter", placeholder="All counties",
                             multi=True, style={"fontSize": "13px"}),
            ], md=4),
            dbc.Col([
                html.Label("Days missing (max)", style={
                    "fontSize": "11px", "fontWeight": 700,
                    "textTransform": "uppercase", "letterSpacing": "0.5px",
                    "color": "var(--text-muted)",
                }),
                dcc.Slider(
                    id="days-slider", min=0, max=365 * 10, step=30,
                    value=365 * 10,
                    marks={0: "0", 365: "1yr", 365*3: "3yr",
                           365*5: "5yr", 365*10: "All"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], md=4),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Input(id="search-box",
                          placeholder="Search name, city, county…",
                          type="text", size="sm", style={"fontSize": "13px"}),
            ], md=6),
            dbc.Col([
                html.Label("Race / Ethnicity", style={
                    "fontSize": "11px", "fontWeight": 700,
                    "textTransform": "uppercase", "letterSpacing": "0.5px",
                    "color": "var(--text-muted)",
                }),
                dcc.Dropdown(id="race-filter", placeholder="All",
                             multi=True, style={"fontSize": "13px"}),
            ], md=6),
        ], className="mt-2"),
    ]), className="mb-4"),

    # ── Metric cards ──────────────────────────────────────────────────────────
    dbc.Row(id="metric-cards", className="mb-4 g-3"),

    # ── Tabs ──────────────────────────────────────────────────────────────────
    dbc.Tabs([

        # Charts
        dbc.Tab(label="📊 Charts", tab_id="tab-charts", children=[
            dbc.Row([
                dbc.Col([
                    section_header("Age group", 1),
                    dcc.Graph(id="age-bar",
                              config={"displayModeBar": False},
                              style={"height": "220px"}),
                ], md=6),
                dbc.Col([
                    section_header("Sex distribution", 2),
                    dcc.Graph(id="sex-pie",
                              config={"displayModeBar": False},
                              style={"height": "220px"}),
                ], md=6),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col([
                    section_header("Top 10 counties", 3),
                    dcc.Graph(id="county-bar",
                              config={"displayModeBar": False},
                              style={"height": "300px"}),
                ], md=6),
                dbc.Col([
                    section_header("Race / ethnicity", 4),
                    dcc.Graph(id="race-pie",
                              config={"displayModeBar": False},
                              style={"height": "300px"}),
                ], md=6),
            ]),
            dbc.Row([
                dbc.Col([
                    section_header("Cases by year reported missing", 5),
                    dcc.Graph(id="time-line",
                              config={"displayModeBar": False},
                              style={"height": "200px"}),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    section_header("Age distribution (histogram)", 6),
                    dcc.Graph(id="age-hist",
                              config={"displayModeBar": False},
                              style={"height": "200px"}),
                ]),
            ]),
        ]),

        # Map
        dbc.Tab(label="🗺️ Map", tab_id="tab-map", children=[
            html.Div([
                section_header("Cases by county", 1),
                html.P("Bubble size and color indicate case count.",
                       style={"fontSize": "12px", "color": "var(--text-muted)",
                              "marginBottom": "6px"}),
                dcc.Graph(id="county-map",
                          config={"displayModeBar": False},
                          style={"height": "520px"}),
            ], className="mt-3"),
        ]),

        # Cases
        dbc.Tab(label="📋 Cases", tab_id="tab-cases", children=[
            html.Div([
                dbc.Row([
                    dbc.Col([
                        section_header("Case list", 1),
                        html.P(id="table-count",
                               style={"fontSize": "12px",
                                      "color": "var(--text-muted)"}),
                    ], md=8),
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="bi bi-filetype-pdf me-1"),
                             "Export PDF report"],
                            id="export-pdf-btn", color="outline-secondary",
                            size="sm", className="float-end mt-2",
                        ),
                        dcc.Download(id="pdf-download"),
                    ], md=4),
                ], className="mt-3"),
                dash_table.DataTable(
                    id="case-table",
                    columns=[
                        {"name": "Case #",       "id": "Case Number"},
                        {"name": "Last",          "id": "Legal Last Name"},
                        {"name": "First",         "id": "Legal First Name"},
                        {"name": "Age",           "id": "Missing Age"},
                        {"name": "City",          "id": "City"},
                        {"name": "County",        "id": "County"},
                        {"name": "Sex",           "id": "Biological Sex"},
                        {"name": "Race",          "id": "Race / Ethnicity"},
                        {"name": "Date Missing",  "id": "DLC_str"},
                        {"name": "Days Missing",  "id": "Days Missing"},
                    ],
                    page_size=20,
                    sort_action="native",
                    filter_action="native",
                    row_selectable="single",
                    selected_rows=[],
                    style_table={"overflowX": "auto", "borderRadius": "2px"},
                    style_cell={
                        "fontSize": "12px", "padding": "8px 12px",
                        "fontFamily": "'Helvetica Neue', Arial, sans-serif",
                        "maxWidth": "180px",
                        "overflow": "hidden", "textOverflow": "ellipsis",
                        "border": "none", "borderBottom": "1px solid var(--border)",
                    },
                    style_header={
                        "fontWeight": 700, "backgroundColor": "var(--bg-surface)",
                        "fontSize": "11px", "textTransform": "uppercase",
                        "letterSpacing": "0.4px", "color": "var(--text-muted)",
                        "border": "none", "borderBottom": "2px solid var(--border)",
                    },
                    style_data_conditional=[
                        {"if": {"row_index": "odd"},
                         "backgroundColor": "var(--bg-surface)"},
                        {"if": {"state": "selected"},
                         "backgroundColor": "var(--accent-hl)",
                         "border": "none"},
                        {"if": {"filter_query": "{Age} < 6 && {Age} > -1"},
                         "backgroundColor": "rgba(181,73,63,0.08)", "color": COLORS["red"]},
                    ],
                ),
                html.Div(id="case-detail", className="mt-3"),
            ]),
        ]),

        # News Check
        dbc.Tab(label="📰 News Check", tab_id="tab-news", children=[
            html.Div([
                section_header("Media attention checker", 1),
                html.P([
                    "Select a case from the Cases tab, then click below to search the web "
                    "for any news coverage for that child."
                ], style={"fontSize": "13px", "color": "var(--text-muted)"}),
                dbc.Alert(
                    id="selected-case-info",
                    color="light",
                    children="No case selected — go to the Cases tab and click a row.",
                    className="mb-2",
                    style={"fontSize": "13px"},
                ),
                dbc.Button(
                    [html.I(className="bi bi-newspaper me-2"),
                     "Check news coverage with Claude"],
                    id="news-check-btn", color="danger", outline=True,
                    disabled=True, className="mb-3",
                ),
                dbc.Spinner(html.Div(id="news-result"), color="danger", size="sm"),

                html.Hr(style={"margin": "1.5rem 0"}),
                section_header("Batch: flag children with no media coverage", 2),
                html.P(
                    "Runs Claude on the youngest cases (age 0–12) to identify those "
                    "with zero news coverage.",
                    style={"fontSize": "13px", "color": "var(--text-muted)"},
                ),
                dbc.Row([
                    dbc.Col(
                        dbc.Input(id="batch-limit", type="number", value=5,
                                  min=1, max=20, placeholder="# cases",
                                  style={"fontSize": "13px"}),
                        md=2,
                    ),
                    dbc.Col(
                        dbc.Button(
                            [html.I(className="bi bi-robot me-2"),
                             "Run batch news scan"],
                            id="batch-news-btn", color="warning", outline=True,
                        ),
                        md=3,
                    ),
                ]),
                dbc.Spinner(
                    html.Div(id="batch-news-result", className="mt-3"),
                    color="warning", size="sm",
                ),
            ], className="mt-3"),
        ]),

        # AI Chat
        dbc.Tab(label="🤖 AI Chat", tab_id="tab-chat", children=[
            html.Div([
                section_header("Ask Claude about the data", 1),
                html.P("Ask any question about the missing persons dataset.",
                       style={"fontSize": "13px", "color": "var(--text-muted)"}),

                dcc.Loading(
                    html.Div(id="chat-messages", className="chat-window"),
                    type="circle",
                    color=COLORS["blue"],
                ),

                dbc.Row([
                    dbc.Col(
                        dbc.Input(
                            id="chat-input",
                            placeholder="e.g. Which counties have the most children under 6 missing?",
                            type="text",
                            style={"fontSize": "13px"},
                        ),
                        md=10,
                    ),
                    dbc.Col(
                        dbc.Button(
                            html.I(className="bi bi-send-fill"),
                            id="chat-send-btn", color="primary",
                            className="w-100",
                        ),
                        md=2,
                    ),
                ], className="g-2"),

                html.Div([
                    html.P("Quick questions:", style={
                        "fontSize": "12px", "color": "var(--text-muted)",
                        "marginTop": "10px", "marginBottom": "6px",
                    }),
                    *[dbc.Button(q, id={"type": "quick-q", "index": i},
                                 size="sm", color="outline-secondary",
                                 className="me-1 mb-1",
                                 style={"fontSize": "11px"})
                      for i, q in enumerate([
                          "Which counties have most children under 6 missing?",
                          "Are there patterns by race/ethnicity in child cases?",
                          "What age group has the most cases and why?",
                          "Which cases have been missing the longest?",
                          "Summarize the dataset in 3 bullet points",
                      ])]
                ]),
            ], className="mt-3"),
        ]),

    ], id="main-tabs", active_tab="tab-charts"),

    html.P(
        "Data: NamUs (namus.nij.ojp.gov) · AI: Anthropic Claude · For research purposes only",
        className="text-center",
        style={"fontSize": "11px", "color": "var(--text-muted)",
               "marginTop": "2.5rem", "marginBottom": "1.5rem"},
    ),

], fluid=True, style={"maxWidth": "1200px"})


# ── Theme callbacks ───────────────────────────────────────────────────────────
app.clientside_callback(
    """
    function(theme) {
        const t = theme || 'light';
        document.documentElement.setAttribute('data-bs-theme', t);
        return 'bi ' + (t === 'dark' ? 'bi-sun-fill' : 'bi-moon-stars-fill');
    }
    """,
    Output("theme-icon", "className"),
    Input("theme-store", "data"),
)


@app.callback(
    Output("theme-store", "data"),
    Input("theme-toggle-btn", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def toggle_theme(_, current):
    return "dark" if (current or "light") == "light" else "light"


# ── Data load callback ────────────────────────────────────────────────────────
@app.callback(
    Output("data-store", "data"),
    Output("upload-status", "children"),
    Output("county-filter", "options"),
    Output("race-filter", "options"),
    Input("upload", "contents"),
    State("upload", "filename"),
    prevent_initial_call=False,
)
def load_data(contents, filename):
    if contents:
        try:
            df     = parse_uploaded(contents, filename)
            status = f"✅ Loaded {filename} — {len(df):,} cases"
        except Exception as e:
            return no_update, f"❌ Error: {e}", [], []
    elif DEFAULT_CSV:
        df     = load_csv(DEFAULT_CSV)
        status = f"Auto-loaded {os.path.basename(DEFAULT_CSV)} — {len(df):,} cases"
    else:
        return no_update, "❌ No CSV file found in app directory. Please upload one.", [], []

    counties    = sorted(df["County"].dropna().unique())
    races       = sorted(df["Race / Ethnicity"].dropna().unique())
    county_opts = [{"label": c, "value": c} for c in counties]
    race_opts   = [{"label": r, "value": r} for r in races]
    return df.to_json(date_format="iso"), status, county_opts, race_opts


# ── Filter helper ─────────────────────────────────────────────────────────────
def apply_filters(df, view, counties, races, days_max, search):
    if view == "children":
        df = df[df["Age"] < 18]
    if counties:
        df = df[df["County"].isin(counties)]
    if races:
        df = df[df["Race / Ethnicity"].isin(races)]
    if days_max is not None:
        df = df[df["Days Missing"] <= days_max]
    if search:
        s    = search.lower()
        mask = (
            df["Legal Last Name"].str.lower().str.contains(s, na=False)  |
            df["Legal First Name"].str.lower().str.contains(s, na=False) |
            df["City"].str.lower().str.contains(s, na=False)             |
            df["County"].str.lower().str.contains(s, na=False)
        )
        df = df[mask]
    return df


# ── Charts + metrics + table callback ────────────────────────────────────────
@app.callback(
    Output("metric-cards", "children"),
    Output("age-bar",      "figure"),
    Output("sex-pie",      "figure"),
    Output("county-bar",   "figure"),
    Output("race-pie",     "figure"),
    Output("time-line",    "figure"),
    Output("age-hist",     "figure"),
    Output("county-map",   "figure"),
    Output("case-table",   "data"),
    Output("table-count",  "children"),
    Input("data-store",    "data"),
    Input("view-toggle",   "value"),
    Input("county-filter", "value"),
    Input("race-filter",   "value"),
    Input("days-slider",   "value"),
    Input("search-box",    "value"),
    Input("theme-store",   "data"),
    prevent_initial_call=True,
)
def update_all(json_data, view, counties, races, days_max, search, theme):
    if not json_data:
        raise PreventUpdate

    df  = pd.read_json(io.StringIO(json_data))
    df["DLC"] = pd.to_datetime(df["DLC"], errors="coerce")
    dff = apply_filters(df, view, counties, races, days_max, search)

    total   = len(dff)
    child_n = int((dff["Age"] < 18).sum())
    pct     = f"{round(child_n / total * 100)}%" if total else "—"
    top_cty = dff["County"].value_counts().idxmax() if total else "—"

    cards = dbc.Row([
        dbc.Col(metric_card("Total cases",     f"{total:,}",   COLORS["blue"],   "bi-people"),         md=3),
        dbc.Col(metric_card("Children (< 18)", f"{child_n:,}", COLORS["red"],    "bi-person-hearts"),  md=3),
        dbc.Col(metric_card("% children",      pct,            COLORS["purple"], "bi-pie-chart"),      md=3),
        dbc.Col(metric_card("Top county",      top_cty,        COLORS["teal"],   "bi-geo-alt"),        md=3),
    ], className="g-3").children

    base_layout, grid_color, dark = get_chart_layout(theme)

    # Age bar
    if view == "children":
        ac        = dff[dff["Age"] < 18]["Child Group"].value_counts() \
                       .reindex(["0–5", "6–12", "13–17"], fill_value=0)
        ac_colors = AGE_COLORS
    else:
        ac        = dff["Age Group"].value_counts() \
                       .reindex(["Under 18", "18–29", "30–49", "50+"], fill_value=0)
        ac_colors = GROUP_COLORS

    age_fig = go.Figure(go.Bar(
        x=ac.index.tolist(), y=ac.values.tolist(),
        marker_color=ac_colors, marker_line_width=0,
    ))
    age_fig.update_layout(**base_layout,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=grid_color),
    )

    # Sex pie
    sc      = dff["Biological Sex"].value_counts()
    sex_fig = go.Figure(go.Pie(
        labels=sc.index.tolist(), values=sc.values.tolist(),
        hole=0.5, marker_colors=SEX_COLORS,
        textinfo="label+percent", textfont_size=11,
    ))
    sex_fig.update_layout(**base_layout)

    # County bar (top 10)
    cc      = dff["County"].value_counts().head(10).sort_values()
    cty_fig = go.Figure(go.Bar(
        y=cc.index.tolist(), x=cc.values.tolist(),
        orientation="h", marker_color=COLORS["red"], marker_line_width=0,
    ))
    cty_fig.update_layout(**base_layout,
        xaxis=dict(showgrid=True, gridcolor=grid_color),
        yaxis=dict(showgrid=False),
    )

    # Race pie (top 6)
    rc       = dff["Race / Ethnicity"].value_counts().head(6)
    race_fig = go.Figure(go.Pie(
        labels=rc.index.tolist(), values=rc.values.tolist(),
        hole=0.5, marker_colors=RACE_COLORS,
        textinfo="label+percent", textfont_size=10,
    ))
    race_fig.update_layout(**base_layout)

    # Timeline
    yr = (dff.dropna(subset=["DLC"])
             .groupby(dff["DLC"].dt.year)
             .size()
             .reset_index(name="count"))
    yr.columns = ["Year", "Count"]
    time_fig = go.Figure(go.Scatter(
        x=yr["Year"], y=yr["Count"],
        mode="lines+markers",
        line=dict(color=COLORS["blue"], width=2),
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(24,95,165,0.10)",
    ))
    time_fig.update_layout(**base_layout,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=grid_color),
    )

    # Age histogram
    ages     = dff["Age"].dropna()
    hist_fig = go.Figure(go.Histogram(
        x=ages, nbinsx=40,
        marker_color=COLORS["purple"], marker_line_width=0,
    ))
    hist_fig.update_layout(**base_layout,
        xaxis=dict(title="Age", showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=grid_color),
    )

    # County map (California bubble scatter)
    county_coords = {
        "Los Angeles":   (34.05, -118.24), "San Diego":      (32.72, -117.15),
        "San Francisco": (37.77, -122.42), "San Bernardino": (34.10, -117.29),
        "Orange":        (33.70, -117.83), "Alameda":        (37.60, -122.00),
        "Riverside":     (33.98, -117.37), "Sacramento":     (38.58, -121.49),
        "Santa Clara":   (37.35, -121.90), "Ventura":        (34.27, -119.23),
        "Contra Costa":  (37.92, -122.00), "Fresno":         (36.74, -119.79),
        "Kern":          (35.37, -119.02), "San Mateo":      (37.54, -122.31),
        "Sonoma":        (38.44, -122.72), "Tulare":         (36.22, -119.35),
        "Santa Barbara": (34.42, -119.70), "Solano":         (38.27, -122.04),
        "Monterey":      (36.24, -121.31), "San Joaquin":    (37.93, -121.27),
        "Stanislaus":    (37.56, -120.99), "Merced":         (37.30, -120.48),
        "Marin":         (38.09, -122.73), "Butte":          (39.73, -121.84),
        "Shasta":        (40.59, -122.39), "Kings":          (36.08, -119.82),
        "Madera":        (37.21, -119.77), "Yolo":           (38.68, -121.90),
        "El Dorado":     (38.77, -120.52), "Imperial":       (33.04, -115.37),
        "San Luis Obispo": (35.27, -120.66), "Napa":         (38.50, -122.27),
        "Lake":          (39.10, -122.75), "Mendocino":      (39.31, -123.43),
        "Humboldt":      (40.87, -124.10), "Nevada":         (39.26, -121.01),
        "Sutter":        (39.03, -121.69), "Placer":         (39.09, -120.80),
        "Santa Cruz":    (36.97, -122.03), "Tehama":         (40.13, -122.23),
        "Lassen":        (40.68, -120.60), "Tuolumne":       (37.96, -119.95),
        "San Benito":    (36.61, -121.07), "Colusa":         (39.18, -122.24),
        "Glenn":         (39.60, -122.39), "Del Norte":      (41.74, -123.92),
        "Siskiyou":      (41.59, -122.54), "Trinity":        (40.65, -123.11),
        "Plumas":        (40.00, -120.84), "Modoc":          (41.59, -120.73),
        "Sierra":        (39.58, -120.52), "Alpine":         (38.60, -119.82),
        "Calaveras":     (38.19, -120.55), "Amador":         (38.45, -120.65),
        "Mono":          (37.94, -118.89), "Inyo":           (36.51, -117.41),
        "Mariposa":      (37.58, -119.91),
    }
    cty_counts = dff["County"].value_counts()
    map_lats, map_lons, map_names, map_vals = [], [], [], []
    for county, (lat, lon) in county_coords.items():
        v = cty_counts.get(county, 0)
        map_lats.append(lat)
        map_lons.append(lon)
        map_names.append(county)
        map_vals.append(v)

    map_fig = go.Figure(go.Scattergeo(
        lat=map_lats, lon=map_lons,
        text=[f"{n}: {v}" for n, v in zip(map_names, map_vals)],
        marker=dict(
            size=[max(6, v**0.5 * 3) for v in map_vals],
            color=map_vals,
            colorscale=[[0, "#fce8e8"], [1, COLORS["red"]]],
            showscale=True,
            colorbar=dict(
                title=dict(text="Cases", font=dict(color=base_layout["font"]["color"])),
                thickness=12, len=0.5,
                tickfont=dict(color=base_layout["font"]["color"]),
            ),
            line=dict(width=0.5, color="rgba(255,255,255,0.5)"),
        ),
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))
    map_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=base_layout["font"]["color"]),
        margin=dict(l=0, r=0, t=0, b=0),
        geo=dict(
            scope="usa",
            showland=True,  landcolor="#1c1f2a"  if dark else "#f5f5f5",
            showlakes=True, lakecolor="#0d1018"   if dark else "#dce9f5",
            showsubunits=True, subunitcolor="#2a2e42" if dark else "#cccccc",
            bgcolor="rgba(0,0,0,0)",
            center=dict(lat=37.5, lon=-119.5),
            projection_scale=4.5,
        ),
    )

    # Table
    tdf = dff[["Case Number", "Legal Last Name", "Legal First Name", "Missing Age",
               "City", "County", "Biological Sex", "Race / Ethnicity",
               "DLC", "Days Missing", "Age"]].copy()
    tdf["DLC_str"] = tdf["DLC"].dt.strftime("%Y-%m-%d")
    table_data = tdf.drop(columns=["DLC"]).to_dict("records")
    count_str  = f"Showing {len(table_data):,} cases"

    return (cards, age_fig, sex_fig, cty_fig, race_fig,
            time_fig, hist_fig, map_fig, table_data, count_str)


# ── Case detail panel ─────────────────────────────────────────────────────────
@app.callback(
    Output("case-detail",       "children"),
    Output("selected-case-info", "children"),
    Output("news-check-btn",    "disabled"),
    Input("case-table",         "selected_rows"),
    State("case-table",         "data"),
    prevent_initial_call=True,
)
def show_case_detail(selected_rows, data):
    if not selected_rows:
        return (html.Div(),
                "No case selected — go to the Cases tab and click a row.",
                True)

    row  = data[selected_rows[0]]
    name = f"{row.get('Legal First Name','')} {row.get('Legal Last Name','')}"
    info = (f"Selected: {name} | Age {row.get('Missing Age','')} | "
            f"{row.get('City','')}, {row.get('County','')} County | "
            f"Missing since {row.get('DLC_str','')}")

    def td_label(text):
        return html.Td(text, style={
            "color": "var(--text-muted)", "paddingRight": "12px",
            "fontSize": "13px", "whiteSpace": "nowrap",
        })
    def td_value(text, **extra):
        return html.Td(text, style={"fontSize": "13px", **extra})

    detail = dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H5(name, style={"fontWeight": 600}),
                html.P(f"Case #{row.get('Case Number','')}",
                       style={"fontSize": "12px", "color": "var(--text-muted)"}),
            ], md=6),
            dbc.Col([
                html.Table([
                    html.Tr([td_label("Age:"),         td_value(row.get("Missing Age",""))]),
                    html.Tr([td_label("Sex:"),         td_value(row.get("Biological Sex",""))]),
                    html.Tr([td_label("Race:"),        td_value(row.get("Race / Ethnicity",""))]),
                    html.Tr([td_label("City:"),        td_value(f"{row.get('City','')}, {row.get('County','')} Co.")]),
                    html.Tr([td_label("Last seen:"),   td_value(row.get("DLC_str",""))]),
                    html.Tr([td_label("Days missing:"),
                             td_value(f"{int(row.get('Days Missing',0)):,}",
                                      fontWeight=600, color=COLORS["red"])]),
                ]),
            ], md=6),
        ]),
        html.Div([
            dbc.Button(
                [html.I(className="bi bi-box-arrow-up-right me-1"), "View on NamUs"],
                href=(f"https://namus.nij.ojp.gov/case#/"
                      f"{row.get('Case Number','').replace('MP','')}"),
                target="_blank", color="outline-primary", size="sm", className="me-2",
            ),
        ], className="mt-2"),
    ]), style={"borderRadius": "2px",
               "borderLeft": f"3px solid {COLORS['red']}"}),

    return detail, info, False


# ── News check callback ───────────────────────────────────────────────────────
@app.callback(
    Output("news-result", "children"),
    Input("news-check-btn",    "n_clicks"),
    State("case-table",        "selected_rows"),
    State("case-table",        "data"),
    prevent_initial_call=True,
)
def check_news(n_clicks, selected_rows, data):
    if not n_clicks or not selected_rows:
        raise PreventUpdate
    row    = data[selected_rows[0]]
    name   = f"{row.get('Legal First Name','')} {row.get('Legal Last Name','')}"
    prompt = news_search_prompt(row)
    result = ask_claude(
        "You are a research assistant helping identify missing children who may have received "
        "news coverage. Be concise and factual. If you are uncertain, say so clearly.",
        prompt,
    )
    return dbc.Alert([
        html.Strong(f"News coverage check — {name}"),
        html.Hr(),
        html.P(result, style={"fontSize": "13px", "whiteSpace": "pre-wrap"}),
    ], color="light", className="mt-2")


# ── Batch news callback ───────────────────────────────────────────────────────
@app.callback(
    Output("batch-news-result", "children"),
    Input("batch-news-btn",     "n_clicks"),
    State("data-store",         "data"),
    State("batch-limit",        "value"),
    prevent_initial_call=True,
)
def batch_news(n_clicks, json_data, limit):
    if not n_clicks or not json_data:
        raise PreventUpdate
    df = pd.read_json(io.StringIO(json_data))
    df["DLC"]          = pd.to_datetime(df["DLC"], errors="coerce")
    df["Days Missing"] = (pd.Timestamp.today() - df["DLC"]).dt.days
    df["Age"]          = df["Missing Age"].apply(parse_age)
    young = df[df["Age"] <= 12].sort_values("Age").head(int(limit or 5))

    results = []
    for _, row in young.iterrows():
        name = f"{row.get('Legal First Name','')} {row.get('Legal Last Name','')}"
        r    = ask_claude(
            "You are helping identify missing children with no media coverage. Be very brief.",
            news_search_prompt(row.to_dict()),
        )
        level_color = "success"
        if   "High"   in r: level_color = "danger"
        elif "Medium" in r: level_color = "warning"
        elif "Low"    in r: level_color = "info"
        results.append(dbc.Alert([
            html.Strong(f"{name} — Age {row.get('Missing Age','')} — "
                        f"{row.get('City','')}, {row.get('County','')}"),
            html.Br(),
            html.Small(f"Case {row.get('Case Number','')}",
                       style={"color": "var(--text-muted)"}),
            html.Hr(style={"margin": "6px 0"}),
            html.P(r, style={"fontSize": "13px", "whiteSpace": "pre-wrap",
                              "margin": 0}),
        ], color=level_color, className="mb-2"))

    return html.Div(results)


# ── AI Chat ───────────────────────────────────────────────────────────────────
def build_data_summary(df):
    children = df[df["Age"] < 18]
    top5_cty = df["County"].value_counts().head(5).to_dict()
    return (
        f"Dataset: {len(df)} total cases, {len(children)} children (under 18). "
        f"State: California. "
        f"Top counties: {top5_cty}. "
        f"Age range: {int(df['Age'].min())}–{int(df['Age'].max())} years. "
        f"Date range: {df['DLC'].min().date()} to {df['DLC'].max().date()}. "
        f"Sex breakdown: {df['Biological Sex'].value_counts().to_dict()}. "
        f"Top races: {df['Race / Ethnicity'].value_counts().head(5).to_dict()}."
    )


QUICK_QUESTIONS = [
    "Which counties have most children under 6 missing?",
    "Are there patterns by race/ethnicity in child cases?",
    "What age group has the most cases and why?",
    "Which cases have been missing the longest?",
    "Summarize the dataset in 3 bullet points",
]


@app.callback(
    Output("chat-input", "value", allow_duplicate=True),
    Input({"type": "quick-q", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def fill_quick_question(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered_id or not any(n for n in n_clicks_list if n):
        raise PreventUpdate
    return QUICK_QUESTIONS[ctx.triggered_id["index"]]


@app.callback(
    Output("chat-messages", "children"),
    Output("chat-history",  "data"),
    Output("chat-input",    "value"),
    Input("chat-send-btn",  "n_clicks"),
    State("chat-input",     "value"),
    State("chat-history",   "data"),
    State("data-store",     "data"),
    prevent_initial_call=True,
)
def send_chat(n_clicks, user_text, history, json_data):
    if not n_clicks or not (user_text or "").strip():
        raise PreventUpdate

    if json_data:
        df = pd.read_json(io.StringIO(json_data))
        df["Age"] = df["Missing Age"].apply(parse_age)
        df["DLC"] = pd.to_datetime(df["DLC"], errors="coerce")
        data_summary = build_data_summary(df)
    else:
        data_summary = "No dataset loaded."

    system = (
        "You are a data analyst assistant helping researchers analyze a NamUs missing "
        "persons dataset. "
        f"Dataset summary: {data_summary} "
        "Answer questions clearly and concisely. Use bullet points when helpful. "
        "Focus on insights useful for finding missing children."
    )

    messages = list(history or [])
    messages.append({"role": "user", "content": user_text.strip()})

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        assistant_text = "Error: ANTHROPIC_API_KEY environment variable is not set."
    else:
        try:
            resp = requests.post(ANTHROPIC_API_URL, headers={
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            }, json={
                "model":      CLAUDE_MODEL,
                "max_tokens": 1000,
                "system":     system,
                "messages":   messages,
            }, timeout=30)
            data = resp.json()
            if "error" in data:
                assistant_text = f"Claude API error: {data['error'].get('message', data['error'])}"
            else:
                texts = [b["text"] for b in data.get("content", [])
                         if b.get("type") == "text"]
                assistant_text = "\n".join(texts) if texts else "(no response)"
        except Exception as e:
            assistant_text = f"Error: {e}"

    messages.append({"role": "assistant", "content": assistant_text})

    bubbles = []
    for msg in messages:
        is_user = msg["role"] == "user"
        bubbles.append(html.Div(
            html.Div(msg["content"],
                     className=f"chat-bubble {'user-bubble' if is_user else 'assistant-bubble'}"),
            style={"textAlign": "right" if is_user else "left",
                   "marginBottom": "8px"},
        ))

    return bubbles, messages, ""


if __name__ == "__main__":
    app.run(debug=False, port=8050)

"""
app.py
------
Interactive front-end for the credit risk engine (Dash), dark "terminal" theme
to match the Fed Dual Mandate Dashboard.

Type a real US ticker -> live SEC EDGAR financials + Yahoo price -> probability
of default, rating, expected credit loss, a plain-English verdict, and the exact
financial figures behind it, with a CECL / IFRS 9 recession overlay.
"""
import sys; sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

import features as F
from scoring import score, pd_to_rating, LGD
from macro import SCENARIOS, apply_overlay
from edgar import fetch_financials

# --- Score the historical population once (for the percentile) ---
_raw = pd.read_csv("data/raw/american_bankruptcy.csv").rename(columns=F.COLUMN_MAP)
_latest = _raw.sort_values("year").groupby("company_name").tail(1)
POP_PD = np.sort(score(_latest)["PD"].values)

# --- Neon palette (matches the Fed dashboard) ---
GOLD, PINK, CYAN, GREEN, AMBER, RED = "#f5b301", "#f43f5e", "#38bdf8", "#34d399", "#fbbf24", "#f43f5e"
BG, CARD, BORDER, MUTED, INK = "#0a0e17", "#0f1522", "#1e293b", "#8b97a7", "#e6edf3"
STATUS_COLOR = {"good": GREEN, "warn": AMBER, "bad": RED}
SCENARIO_DESC = {"baseline": "normal economy", "adverse": "mild recession", "severe": "2008-style crisis"}

LABEL = {"letterSpacing": "1.5px", "textTransform": "uppercase", "fontSize": "0.7rem",
         "color": MUTED, "fontWeight": "700"}
SUB = {"fontSize": "0.72rem", "color": MUTED, "marginTop": "2px"}
CARD_STYLE = {"background": CARD, "border": f"1px solid {BORDER}", "borderRadius": "10px",
              "padding": "16px 18px", "height": "100%"}


def money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    ax = abs(x)
    if ax >= 1e12: return f"${x/1e12:.2f}T"
    if ax >= 1e9:  return f"${x/1e9:.1f}B"
    if ax >= 1e6:  return f"${x/1e6:.0f}M"
    return f"${x:,.0f}"


def tier_color(rating):
    if rating in {"AAA", "AA", "A", "BBB"}: return GREEN
    if rating in {"BB", "B"}: return AMBER
    return RED


def safer_than(pd_value):
    return 100.0 * (POP_PD > pd_value).mean()


def indicators(ratios, raw):
    """Each key ratio with its value, status, meaning, and the raw $ behind it."""
    def status(v, good, warn, higher_is_better):
        if v is None or (isinstance(v, float) and np.isnan(v)): return "warn"
        if higher_is_better: return "good" if v >= good else ("warn" if v >= warn else "bad")
        return "good" if v <= good else ("warn" if v <= warn else "bad")

    L, R = ratios, raw
    return [
        ("Leverage", f"{L['leverage']:.0%}", status(L['leverage'], .5, .75, False),
         "share of the company financed by debt", f"{money(R['total_liabilities'])} debt / {money(R['total_assets'])} assets"),
        ("Liquidity", f"{L['current_ratio']:.2f}", status(L['current_ratio'], 1.5, 1.0, True),
         "short-term assets per $1 of short-term bills", f"{money(R['current_assets'])} / {money(R['current_liabilities'])} current liab"),
        ("Return on assets", f"{L['roa']:.1%}", status(L['roa'], .05, 0, True),
         "profit per $1 of assets", f"{money(R['net_income'])} income / {money(R['total_assets'])} assets"),
        ("Debt service", f"{L['ebit_to_liab']:.1%}", status(L['ebit_to_liab'], .15, 0, True),
         "operating profit vs total debt", f"{money(R['ebit'])} EBIT / {money(R['total_liabilities'])} debt"),
        ("Market cushion", ("n/a" if np.isnan(L['mve_to_liab']) else f"{L['mve_to_liab']:.1f}x"),
         status(None if np.isnan(L['mve_to_liab']) else L['mve_to_liab'], 3, 1, True),
         "equity buffer the market assigns vs debt", f"{money(R['market_value'])} market value / {money(R['total_liabilities'])} debt"),
    ]


def verdict(rating, bullets):
    strengths = [b[0] for b in bullets if b[2] == "good"]
    concerns = [b[0] for b in bullets if b[2] == "bad"]
    if rating in {"AAA", "AA", "A"}:
        head, lead, col = "Strong investment-grade credit", "Low default risk — financially solid.", GREEN
    elif rating == "BBB":
        head, lead, col = "Investment grade, lowest tier", "Investment grade, but less margin for error.", GREEN
    elif rating in {"BB", "B"}:
        head, lead, col = "Speculative grade", "Elevated default risk — a lender would price this carefully.", AMBER
    else:
        head, lead, col = "Distressed credit", "High probability of default — deep sub-investment grade.", RED
    extra = (f" ✓ Strengths: {', '.join(strengths)}." if strengths else "") + \
            (f"  ✕ Concerns: {', '.join(concerns)}." if concerns else "")
    return head, lead + extra, col


app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], title="Credit Risk Engine")
server = app.server
app.index_string = """<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>body{background:#0a0e17;color:#e6edf3;} .form-control,.input-group-text{background:#0f1522!important;
color:#e6edf3!important;border-color:#1e293b!important;} </style></head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""

header = html.Div([
    html.Div([
        html.Span("◈ CORPORATE CREDIT RISK ENGINE", style={"fontWeight": "800", "letterSpacing": "2px", "color": INK}),
        html.Span("  |  PD · RATING · EXPECTED LOSS", style={"color": MUTED, "letterSpacing": "1px"}),
    ]),
    html.Span([html.Span("● ", style={"color": GREEN}), "LIVE · SEC EDGAR + YAHOO"],
              style={"color": MUTED, "fontSize": "0.75rem", "letterSpacing": "1px"}),
], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
          "padding": "14px 4px", "borderBottom": f"2px solid {GOLD}", "marginTop": "14px"})

tagline = html.P("Type any US ticker and get a bank-style credit verdict — default probability, rating and "
                 "expected loss — straight from the company's real financials.",
                 style={"color": MUTED, "textAlign": "center", "padding": "10px 0"})

controls = html.Div([
    html.Div("COMPANY TICKER", style=LABEL),
    dbc.InputGroup([dbc.Input(id="ticker", value="AAPL", type="text", debounce=True),
                    dbc.Button("Analyze", id="go", color="warning")], className="mt-1"),
    html.Small("e.g. AAPL, MSFT, F, GM, DAL, XRX", style=SUB),
    html.Hr(style={"borderColor": BORDER}),
    html.Div("LOAN AMOUNT (EAD)", style=LABEL),
    dcc.Input(id="ead", type="number", value=1_000_000, step=100_000, className="form-control mt-1"),
    html.Br(),
    html.Div("ECONOMIC SCENARIO", style=LABEL),
    dcc.RadioItems(id="scenario",
        options=[{"label": f"  {k.title()} (×{v}) — {SCENARIO_DESC[k]}", "value": k} for k, v in SCENARIOS.items()],
        value="baseline", labelStyle={"display": "block", "padding": "3px", "color": INK},
        inputStyle={"marginRight": "6px"}, className="mt-1"),
    html.Small("Scales every PD by how far corporate defaults rise in a downturn. Anchored in Moody's / S&P "
               "default studies: the speculative-grade default rate runs ~2% in benign years and hit ~10% "
               "in 2009 — about a 2.5× move.", style=SUB),
], style=CARD_STYLE)


def stat(label, vid, sid, color):
    return html.Div([html.Div(label, style=LABEL),
                     html.Div(id=vid, style={"fontSize": "2.3rem", "fontWeight": "800", "color": color, "lineHeight": "1.15"}),
                     html.Div(id=sid, style=SUB)], style=CARD_STYLE)

app.layout = dbc.Container([
    header, tagline,
    dbc.Row([
        dbc.Col(controls, md=4),
        dbc.Col(html.Div([
            html.H4(id="headline", className="fw-bold", style={"color": INK}),
            html.Div(id="subline", style=SUB),
            html.Br(),
            dbc.Row([
                dbc.Col(stat("RATING", "kpi-rating", "kpi-rating-sub", GREEN), width=3),
                dbc.Col(stat("PROBABILITY OF DEFAULT", "kpi-pd", "kpi-pd-sub", PINK), width=3),
                dbc.Col(stat("EXPECTED CREDIT LOSS", "kpi-ecl", "kpi-ecl-sub", GOLD), width=3),
                dbc.Col(stat("MARKET CAP", "kpi-mc", "kpi-mc-sub", CYAN), width=3),
            ], className="g-2"),
            html.Div(id="verdict", style={"margin": "16px 0", "padding": "12px 16px",
                                          "borderRadius": "10px", "background": CARD}),
            dbc.Row([
                dbc.Col(dcc.Loading(dcc.Graph(id="gauge", config={"displayModeBar": False})), md=5),
                dbc.Col([html.Div("KEY FINANCIAL INDICATORS", style=LABEL),
                         html.Div(id="bullets", className="mt-2")], md=7),
            ], className="mt-1"),
            html.Div(id="compare", className="text-center mt-2",
                     style={"color": INK, "fontSize": "1.05rem"}),
        ]), md=8),
    ], className="mt-2"),
    dcc.Store(id="base"),
], fluid=True, style={"maxWidth": "1200px"})


@app.callback(Output("base", "data"), Input("go", "n_clicks"), Input("ticker", "value"))
def analyze(_, ticker):
    try:
        df = fetch_financials(ticker)
        s = score(df).iloc[0]
        ratios = {k: (None if pd.isna(v) else float(v)) for k, v in F.build_ratios(df).iloc[0].items()}
        keep = ["total_assets", "total_liabilities", "current_assets", "current_liabilities",
                "net_income", "ebit", "market_value"]
        raw = {k: (None if pd.isna(df[k].iloc[0]) else float(df[k].iloc[0])) for k in keep}
        return {"ok": True, "name": s["company_name"], "year": int(df["year"].iloc[0]),
                "base_pd": float(s["PD"]), "ratios": ratios, "raw": raw}
    except Exception as e:
        return {"ok": False, "msg": f"Could not analyze '{ticker}': {e}"}


def dark_gauge(pd_adj, color):
    g = go.Figure(go.Indicator(
        mode="gauge+number", value=pd_adj * 100,
        number={"suffix": "%", "valueformat": ".2f", "font": {"size": 40, "color": color}},
        gauge={"axis": {"range": [0, 15], "tickcolor": MUTED},
               "bar": {"color": color},
               "bgcolor": "rgba(0,0,0,0)",
               "steps": [{"range": [0, 1], "color": "#132b1f"},
                         {"range": [1, 3], "color": "#2a2513"},
                         {"range": [3, 15], "color": "#2a1417"}]}))
    g.update_layout(height=250, margin=dict(t=20, b=10, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)", font_color=MUTED)
    return g


@app.callback(
    Output("headline", "children"), Output("subline", "children"),
    Output("kpi-rating", "children"), Output("kpi-rating", "style"), Output("kpi-rating-sub", "children"),
    Output("kpi-pd", "children"), Output("kpi-pd-sub", "children"),
    Output("kpi-ecl", "children"), Output("kpi-ecl-sub", "children"),
    Output("kpi-mc", "children"), Output("kpi-mc-sub", "children"),
    Output("verdict", "children"), Output("verdict", "style"),
    Output("gauge", "figure"), Output("bullets", "children"), Output("compare", "children"),
    Input("base", "data"), Input("ead", "value"), Input("scenario", "value"),
)
def render(base, ead, scenario):
    rstyle = {"fontSize": "2.3rem", "fontWeight": "800", "lineHeight": "1.15"}
    if not base or not base.get("ok"):
        msg = (base or {}).get("msg", "Type a ticker and press Analyze.")
        return ("", msg, "—", {**rstyle, "color": MUTED}, "", "—", "", "—", "", "—", "",
                "", {"display": "none"}, dark_gauge(0, MUTED), "", "")

    ead = ead or 0
    pd_adj = float(apply_overlay(np.array([base["base_pd"]]), scenario)[0])
    rating = pd_to_rating(pd_adj)
    col = tier_color(rating)
    ecl = pd_adj * LGD * ead
    raw, ratios = base["raw"], base["ratios"]
    mc = raw["market_value"]
    sub = f"FY{base['year']} · financials from SEC EDGAR"

    bullets = indicators(ratios, raw)
    v_head, v_body, v_col = verdict(rating, bullets)

    bullet_ui = [html.Div([
        html.Span("●", style={"color": STATUS_COLOR[st], "marginRight": "8px"}),
        html.Span(f"{label}: ", style={"fontWeight": "700", "color": INK}),
        html.Span(val, style={"color": STATUS_COLOR[st], "fontWeight": "800"}),
        html.Span(f"  — {read}", style={"color": MUTED, "fontSize": "0.8rem"}),
        html.Div(detail, style={"color": MUTED, "fontSize": "0.75rem", "marginLeft": "18px", "fontFamily": "monospace"}),
    ], className="mb-2") for (label, val, st, read, detail) in bullets]

    pct = safer_than(pd_adj)
    verdict_ui = [html.Span(v_head + ". ", style={"color": v_col, "fontWeight": "800"}),
                  html.Span(v_body, style={"color": INK})]
    vstyle = {"margin": "16px 0", "padding": "12px 16px", "borderRadius": "10px",
              "background": CARD, "borderLeft": f"4px solid {v_col}"}

    return (base["name"], sub,
            rating, {**rstyle, "color": col}, "implied credit grade",
            f"{pd_adj:.2%}", "over a 2-year horizon",
            f"${ecl:,.0f}", f"on {money(ead)} loan · LGD 45%",
            money(mc) if mc else "n/a", "equity value · Yahoo",
            verdict_ui, vstyle,
            dark_gauge(pd_adj, col), bullet_ui,
            f"🏦 Safer than {pct:.0f}% of US public companies (1999–2018)")


if __name__ == "__main__":
    app.run(debug=False, port=8050)

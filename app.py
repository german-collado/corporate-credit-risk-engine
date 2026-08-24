"""
app.py
------
Interactive front-end for the credit risk engine (Dash).

Type a real US ticker. The app pulls that company's latest annual financials
from SEC EDGAR + its live share price from Yahoo, runs them through the model,
and returns its probability of default, letter rating and expected credit loss,
a plain-English verdict, the financial ratios behind it, and where it sits
versus the 1999-2018 historical population.
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

# --- At startup: score the historical population once, for the percentile ---
_raw = pd.read_csv("data/raw/american_bankruptcy.csv").rename(columns=F.COLUMN_MAP)
_latest = _raw.sort_values("year").groupby("company_name").tail(1)
POP_PD = np.sort(score(_latest)["PD"].values)

# Palette
GREEN, AMBER, RED, INK = "#16a34a", "#f59e0b", "#dc2626", "#0f172a"
STATUS_COLOR = {"good": GREEN, "warn": AMBER, "bad": RED}

# Plain-English label for each macro scenario (the ×1 / ×1.5 / ×2.5 dials)
SCENARIO_DESC = {"baseline": "normal economy", "adverse": "mild recession", "severe": "2008-style crisis"}


def tier_color(rating):
    if rating in {"AAA", "AA", "A", "BBB"}:
        return GREEN
    if rating in {"BB", "B"}:
        return AMBER
    return RED


def safer_than(pd_value):
    return 100.0 * (POP_PD > pd_value).mean()


def indicators(ratios, market_cap):
    """Build the key-ratio bullets: (label, value, status, one-line read)."""
    def band(v, good, warn, higher_is_better=True, fmt="{:.2f}"):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a", "warn"
        if higher_is_better:
            status = "good" if v >= good else ("warn" if v >= warn else "bad")
        else:
            status = "good" if v <= good else ("warn" if v <= warn else "bad")
        return fmt.format(v), status

    L = ratios
    out = []
    val, st = band(L["leverage"], 0.5, 0.75, higher_is_better=False, fmt="{:.0%}")
    out.append(("Leverage — Debt / Assets", val, st, "how much of the company is financed by debt"))
    val, st = band(L["current_ratio"], 1.5, 1.0, higher_is_better=True)
    out.append(("Liquidity — Current ratio", val, st, "short-term assets vs short-term bills"))
    val, st = band(L["roa"], 0.05, 0.0, higher_is_better=True, fmt="{:.1%}")
    out.append(("Profitability — Return on assets", val, st, "profit generated per dollar of assets"))
    val, st = band(L["ebit_to_liab"], 0.15, 0.0, higher_is_better=True, fmt="{:.1%}")
    out.append(("Debt service — EBIT / Debt", val, st, "operating profit vs total debt"))
    mv = None if (market_cap is None or np.isnan(L["mve_to_liab"])) else L["mve_to_liab"]
    val, st = band(mv, 3.0, 1.0, higher_is_better=True, fmt="{:.1f}x")
    out.append(("Market cushion — Market value / Debt", val, st, "equity buffer the market assigns"))
    return out


def verdict(rating, pd_adj, bullets):
    strengths = [b[0].split(" — ")[0] for b in bullets if b[2] == "good"]
    concerns = [b[0].split(" — ")[0] for b in bullets if b[2] == "bad"]
    if rating in {"AAA", "AA", "A"}:
        head, color = "Strong investment-grade credit", "success"
        lead = "Low default risk — this company looks financially solid."
    elif rating == "BBB":
        head, color = "Investment grade, lowest tier", "success"
        lead = "Still investment grade, but with less margin for error."
    elif rating in {"BB", "B"}:
        head, color = "Speculative grade (below investment grade)", "warning"
        lead = "Elevated default risk — a lender would price this carefully."
    else:
        head, color = "Distressed credit", "danger"
        lead = "High probability of default — deep sub-investment grade."
    extra = ""
    if strengths:
        extra += f" Strengths: {', '.join(strengths)}."
    if concerns:
        extra += f" Concerns: {', '.join(concerns)}."
    return head, f"{lead}{extra}", color


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Credit Risk Engine")
server = app.server

# Soft tinted page background so the white cards pop (plain white looked default).
app.index_string = """<!DOCTYPE html>
<html>
  <head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>
      body { background: linear-gradient(160deg,#eaf0fb 0%, #eef1f7 45%, #f3ecf8 100%);
             background-attachment: fixed; }
      .card { border: none !important; }
    </style>
  </head>
  <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""

hero = html.Div([
    html.H2("Corporate Credit Risk Engine", className="fw-bold mb-1", style={"color": "white"}),
    html.P("Type any US ticker and get a bank-style credit verdict — default probability, rating "
           "and expected loss — straight from the company's real financials.",
           className="mb-0", style={"color": "#e2e8f0", "fontSize": "1.05rem"}),
], style={"background": f"linear-gradient(120deg,{INK},#1e3a8a)", "padding": "22px 26px",
          "borderRadius": "14px", "marginTop": "16px"})

controls = dbc.Card(dbc.CardBody([
    html.H6("Company ticker", className="text-muted"),
    dbc.InputGroup([
        dbc.Input(id="ticker", value="AAPL", type="text", debounce=True),
        dbc.Button("Analyze", id="go", color="primary"),
    ]),
    html.Small("e.g. AAPL, MSFT, F, GM, DAL, XRX", className="text-muted"),
    html.Hr(),
    html.H6("Loan amount (EAD)", className="text-muted"),
    dcc.Input(id="ead", type="number", value=1_000_000, step=100_000, className="form-control"),
    html.Br(), html.Br(),
    html.H6("Economic scenario", className="text-muted"),
    dcc.RadioItems(
        id="scenario",
        options=[{"label": f"  {k.title()} (×{v}) — {SCENARIO_DESC[k]}", "value": k}
                 for k, v in SCENARIOS.items()],
        value="baseline", labelStyle={"display": "block", "padding": "3px"}),
    html.Small(
        "The multiplier scales every PD by how far corporate defaults rise in a downturn. "
        "Anchored in published default studies (Moody's / S&P): the speculative-grade default "
        "rate runs ~2% in benign years and spiked to ~10% in 2009 — roughly a 2.5× move.",
        className="text-muted", style={"fontSize": "0.72rem", "display": "block", "marginTop": "6px"}),
], ), className="shadow-sm")


def big_badge(id_):
    return dbc.Card(dbc.CardBody([
        html.Div("Rating", className="text-muted small"),
        html.Div(id=id_, style={"fontSize": "3.2rem", "fontWeight": "800", "lineHeight": "1"}),
    ], className="text-center"), className="shadow-sm h-100")


def kpi(title, id_):
    return dbc.Card(dbc.CardBody([html.Div(title, className="text-muted small"),
                                  html.H3(id=id_, className="fw-bold mb-0")]),
                    className="shadow-sm h-100")

app.layout = dbc.Container([
    hero,
    html.Br(),
    dbc.Row([
        dbc.Col(controls, md=4),
        dbc.Col(dcc.Loading(html.Div([
            html.H4(id="headline", className="fw-bold"),
            html.Div(id="subline", className="text-muted mb-3"),
            dbc.Row([
                dbc.Col(big_badge("kpi-rating"), width=3),
                dbc.Col(kpi("Probability of default", "kpi-pd"), width=4),
                dbc.Col(kpi("Expected credit loss", "kpi-ecl"), width=5),
            ], className="g-2"),
            html.Br(),
            dbc.Alert(id="verdict", className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="gauge", config={"displayModeBar": False}), md=6),
                dbc.Col([html.H6("Key financial indicators", className="text-muted"),
                         html.Div(id="bullets")], md=6),
            ]),
            html.Div(id="compare", className="lead text-center mt-2"),
        ])), md=8),
    ]),
    dcc.Store(id="base"),
], fluid=True)


@app.callback(
    Output("base", "data"),
    Input("go", "n_clicks"), Input("ticker", "value"),
    prevent_initial_call=False,
)
def analyze(_, ticker):
    try:
        df = fetch_financials(ticker)
        s = score(df).iloc[0]
        ratios = F.build_ratios(df).iloc[0].to_dict()
        mv = df["market_value"].iloc[0]
        return {"ok": True, "name": s["company_name"], "year": int(df["year"].iloc[0]),
                "base_pd": float(s["PD"]), "mktcap": None if pd.isna(mv) else float(mv),
                "ratios": {k: (None if pd.isna(v) else float(v)) for k, v in ratios.items()}}
    except Exception as e:
        return {"ok": False, "msg": f"Could not analyze '{ticker}': {e}"}


@app.callback(
    Output("headline", "children"), Output("subline", "children"),
    Output("kpi-rating", "children"), Output("kpi-rating", "style"),
    Output("kpi-pd", "children"), Output("kpi-ecl", "children"),
    Output("verdict", "children"), Output("verdict", "color"),
    Output("gauge", "figure"), Output("bullets", "children"), Output("compare", "children"),
    Input("base", "data"), Input("ead", "value"), Input("scenario", "value"),
)
def render(base, ead, scenario):
    blank = go.Figure(); blank.update_layout(height=260, margin=dict(t=10, b=10))
    if not base or not base.get("ok"):
        msg = (base or {}).get("msg", "Type a ticker and press Analyze.")
        return "", msg, "—", {}, "—", "—", "", "secondary", blank, "", ""

    ead = ead or 0
    pd_adj = float(apply_overlay(np.array([base["base_pd"]]), scenario)[0])
    rating = pd_to_rating(pd_adj)
    ecl = pd_adj * LGD * ead
    mc = base["mktcap"]
    sub = f"FY{base['year']} financials from SEC EDGAR" + (f"  ·  market cap ${mc/1e9:,.0f}B" if mc else "")

    bullets = indicators(base["ratios"], mc)
    v_head, v_body, v_color = verdict(rating, pd_adj, bullets)

    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=pd_adj * 100,
        number={"suffix": "%", "valueformat": ".2f", "font": {"size": 44}},
        title={"text": "Probability of default", "font": {"size": 13}},
        gauge={"axis": {"range": [0, 15]}, "bar": {"color": tier_color(rating)},
               "steps": [{"range": [0, 1], "color": "#dcfce7"},
                         {"range": [1, 3], "color": "#fef9c3"},
                         {"range": [3, 15], "color": "#fee2e2"}]}))
    gauge.update_layout(height=260, margin=dict(t=40, b=10))

    bullet_ui = [
        html.Div([
            html.Span("●", style={"color": STATUS_COLOR[st], "fontSize": "1.1rem", "marginRight": "8px"}),
            html.Span(label.split(" — ")[0] + ": ", className="fw-semibold"),
            html.Span(val, style={"color": STATUS_COLOR[st], "fontWeight": "700"}),
            html.Div(read, className="text-muted small", style={"marginLeft": "20px"}),
        ], className="mb-2")
        for (label, val, st, read) in bullets
    ]

    pct = safer_than(pd_adj)
    compare = f"🏦 Safer than {pct:.0f}% of US public companies (1999–2018)"

    return (base["name"], sub,
            rating, {"fontSize": "3.2rem", "fontWeight": "800", "lineHeight": "1", "color": tier_color(rating)},
            f"{pd_adj:.2%}", f"${ecl:,.0f}",
            [html.Strong(v_head + ". "), v_body], v_color,
            gauge, bullet_ui, compare)


if __name__ == "__main__":
    app.run(debug=False, port=8050)

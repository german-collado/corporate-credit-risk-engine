"""
app.py
------
Interactive front-end for the credit risk engine (Dash).

Type a real US ticker. The app pulls that company's latest annual financials
from SEC EDGAR + its live share price from Yahoo, runs them through the model,
and returns its probability of default, letter rating and expected credit loss —
plus where it sits versus the 1999-2018 historical population.
"""
import sys; sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

import features as F
from scoring import score, pd_to_rating, LGD
from macro import SCENARIOS, apply_overlay
from edgar import fetch_financials

# --- At startup: score the historical population once, for the percentile bar ---
_raw = pd.read_csv("data/raw/american_bankruptcy.csv").rename(columns=F.COLUMN_MAP)
_latest = _raw.sort_values("year").groupby("company_name").tail(1)
POP_PD = np.sort(score(_latest)["PD"].values)  # sorted PDs of ~9k US companies

IG = {"AAAA", "AAA", "AA", "A", "BBB"}


def rating_color(letter):
    return "#16a34a" if letter in {"AAA", "AA", "A", "BBB"} else "#dc2626"


def safer_than(pd_value):
    """Percentile: share of the historical population riskier than this PD."""
    return 100.0 * (POP_PD > pd_value).mean()


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Credit Risk Engine")
server = app.server

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
        options=[{"label": f"  {k.title()} (×{v})", "value": k} for k, v in SCENARIOS.items()],
        value="baseline", labelStyle={"display": "block", "padding": "3px"},
    ),
]))


def kpi(title, id_):
    return dbc.Card(dbc.CardBody([html.H6(title, className="text-muted"),
                                  html.H2(id=id_, className="fw-bold")]))

app.layout = dbc.Container([
    html.Br(),
    html.H2("Corporate Credit Risk Engine"),
    html.P("Type a US ticker → live financials from SEC EDGAR → probability of default, "
           "rating and expected credit loss, with a CECL / IFRS 9 recession overlay.",
           className="text-muted"),
    html.Hr(),
    dbc.Row([
        dbc.Col(controls, md=4),
        dbc.Col(dcc.Loading(html.Div([
            html.H4(id="headline"), html.Div(id="subline", className="text-muted"),
            html.Br(),
            dbc.Row([dbc.Col(kpi("Rating", "kpi-rating")),
                     dbc.Col(kpi("Probability of default", "kpi-pd")),
                     dbc.Col(kpi("Expected credit loss", "kpi-ecl"))]),
            dcc.Graph(id="gauge", config={"displayModeBar": False}),
            html.Div(id="compare", className="lead text-center"),
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
    """Fetch + score the company once; store its baseline PD and identity."""
    try:
        df = fetch_financials(ticker)
        s = score(df).iloc[0]
        mv = df["market_value"].iloc[0]
        return {"ok": True, "name": s["company_name"], "year": int(df["year"].iloc[0]),
                "base_pd": float(s["PD"]), "mktcap": None if pd.isna(mv) else float(mv)}
    except Exception as e:
        return {"ok": False, "msg": f"Could not analyze '{ticker}': {e}"}


@app.callback(
    Output("headline", "children"), Output("subline", "children"),
    Output("kpi-rating", "children"), Output("kpi-rating", "style"),
    Output("kpi-pd", "children"), Output("kpi-ecl", "children"),
    Output("gauge", "figure"), Output("compare", "children"),
    Input("base", "data"), Input("ead", "value"), Input("scenario", "value"),
)
def render(base, ead, scenario):
    if not base or not base.get("ok"):
        msg = base.get("msg", "Type a ticker and press Analyze.") if base else "Type a ticker and press Analyze."
        return "", msg, "—", {}, "—", "—", go.Figure(), ""

    ead = ead or 0
    pd_adj = float(apply_overlay(np.array([base["base_pd"]]), scenario)[0])
    rating = pd_to_rating(pd_adj)
    ecl = pd_adj * LGD * ead
    mc = base["mktcap"]
    sub = f"FY{base['year']} financials from SEC EDGAR" + (f" · market cap ${mc/1e9:,.0f}B" if mc else "")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=pd_adj * 100,
        number={"suffix": "%", "valueformat": ".2f"},
        gauge={"axis": {"range": [0, 15]}, "bar": {"color": rating_color(rating)},
               "steps": [{"range": [0, 1], "color": "#dcfce7"},
                         {"range": [1, 3], "color": "#fef9c3"},
                         {"range": [3, 15], "color": "#fee2e2"}]}))
    gauge.update_layout(height=280, margin=dict(t=20, b=10))

    pct = safer_than(pd_adj)
    compare = f"Safer than {pct:.0f}% of US public companies (1999–2018 population)"

    return (base["name"], sub, rating, {"color": rating_color(rating)},
            f"{pd_adj:.2%}", f"${ecl:,.0f}", gauge, compare)


if __name__ == "__main__":
    app.run(debug=False, port=8050)

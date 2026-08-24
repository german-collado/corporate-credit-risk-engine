"""
app.py
------
Interactive front-end for the credit risk engine (Dash).

Pick a company, set a loan amount and an economic scenario, and read off its
probability of default, letter rating and expected credit loss in dollars.
Same stack (Dash / Plotly) as the Fed Dual Mandate Dashboard, on purpose.
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

# --- Score every company's latest financials once, at startup ---
raw = pd.read_csv("data/raw/american_bankruptcy.csv").rename(columns=F.COLUMN_MAP)
latest = raw.sort_values("year").groupby("company_name").tail(1)
scored = score(latest).reset_index(drop=True)

# Dropdown: a spread across the risk spectrum, labelled with rating + PD
spread = scored.sort_values("PD").iloc[:: max(1, len(scored) // 800)]
options = [
    {"label": f"{r.company_name} — {r.rating} (PD {r.PD:.1%})", "value": r.company_name}
    for r in spread.itertuples()
]

IG = {"AAA", "AA", "A", "BBB"}  # investment grade
RATING_COLOR = {g: "#16a34a" for g in IG}  # green for IG, red for the rest


def rating_color(letter):
    return RATING_COLOR.get(letter, "#dc2626")


# --- Build the app ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Credit Risk Engine")
server = app.server

controls = dbc.Card(dbc.CardBody([
    html.H6("Company", className="text-muted"),
    dcc.Dropdown(id="company", options=options, value=options[len(options) // 2]["value"], clearable=False),
    html.Br(),
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
    html.P("Probability of default → rating → expected credit loss, with a CECL / IFRS 9 macro overlay.",
           className="text-muted"),
    html.Hr(),
    dbc.Row([
        dbc.Col(controls, md=4),
        dbc.Col([
            dbc.Row([dbc.Col(kpi("Rating", "kpi-rating")),
                     dbc.Col(kpi("Probability of default", "kpi-pd")),
                     dbc.Col(kpi("Expected credit loss", "kpi-ecl"))]),
            html.Br(),
            dcc.Graph(id="gauge", config={"displayModeBar": False}),
        ], md=8),
    ]),
], fluid=True)


@app.callback(
    Output("kpi-rating", "children"), Output("kpi-rating", "style"),
    Output("kpi-pd", "children"), Output("kpi-ecl", "children"),
    Output("gauge", "figure"),
    Input("company", "value"), Input("ead", "value"), Input("scenario", "value"),
)
def update(company, ead, scenario):
    ead = ead or 0
    base_pd = float(scored.loc[scored.company_name == company, "PD"].iloc[0])
    pd_adj = float(apply_overlay(np.array([base_pd]), scenario)[0])  # scenario shift
    rating = pd_to_rating(pd_adj)
    ecl = pd_adj * LGD * ead

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pd_adj * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 40]},
            "bar": {"color": rating_color(rating)},
            "steps": [{"range": [0, 3], "color": "#dcfce7"},
                      {"range": [3, 10], "color": "#fef9c3"},
                      {"range": [10, 40], "color": "#fee2e2"}],
        },
    ))
    gauge.update_layout(height=300, margin=dict(t=20, b=10))

    return (rating, {"color": rating_color(rating)},
            f"{pd_adj:.1%}", f"${ecl:,.0f}", gauge)


if __name__ == "__main__":
    app.run(debug=False, port=8050)

"""
macro.py
--------
Macro-scenario overlay — the forward-looking piece required by CECL / IFRS 9.

Idea: the same company defaults more often in a recession. We shift every PD
by a scenario multiplier, then recompute ratings and ECL. This is a simplified
"scalar overlay": one dial for the whole book.

WHY the multipliers are external (and not fit on our own data):
    Our dataset's year-by-year default rate is distorted by right-censoring
    (we can't see who fails after 2018) and panel fill-in, so it does NOT track
    the economic cycle. Instead we anchor the dial in published default studies
    (Moody's / S&P), where corporate default rates roughly triple to quintuple
    from a benign year to a severe recession (e.g. ~2% -> ~10% in 2009).
"""
import numpy as np
import pandas as pd

import features as F
from scoring import score, pd_to_rating, LGD, EAD

# Scenario multipliers applied to every company's PD. Documented assumptions.
SCENARIOS = {
    "baseline": 1.0,   # normal conditions
    "adverse":  1.5,   # mild recession
    "severe":   2.5,   # 2008-style downturn
}
PD_CAP = 0.99  # a probability can never exceed 1


def apply_overlay(pd_values, scenario: str):
    """Scale PDs by the scenario multiplier, capped at PD_CAP."""
    mult = SCENARIOS[scenario]
    return np.minimum(pd_values * mult, PD_CAP)


def stress_portfolio(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Score a loan book and report PD, rating mix and total ECL per scenario."""
    base = score(df_raw)                      # baseline PD / rating / ECL
    rows = []
    for name in SCENARIOS:
        pd_adj = apply_overlay(base["PD"].values, name)
        ecl = pd_adj * LGD * EAD
        # count how many names fall below investment grade (BBB and above = IG)
        ig = {"AAA", "AA", "A", "BBB"}
        sub_ig = sum(pd_to_rating(p) not in ig for p in pd_adj)
        rows.append({
            "scenario": name,
            "multiplier": SCENARIOS[name],
            "avg_PD": pd_adj.mean(),
            "total_ECL_$": ecl.sum(),
            "sub_investment_grade": sub_ig,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")

    # Build an example loan book: latest financials of 500 companies, $1M each.
    raw = pd.read_csv("data/raw/american_bankruptcy.csv").rename(columns=F.COLUMN_MAP)
    book = raw.sort_values("year").groupby("company_name").tail(1).sample(500, random_state=1)

    n = len(book)
    print(f"Loan book: {n} companies x ${EAD:,.0f} = ${n*EAD:,.0f} total exposure\n")
    out = stress_portfolio(book)
    out["avg_PD"] = (out["avg_PD"] * 100).round(2).astype(str) + "%"
    out["total_ECL_$"] = out["total_ECL_$"].map(lambda x: f"${x:,.0f}")
    print(out.to_string(index=False))

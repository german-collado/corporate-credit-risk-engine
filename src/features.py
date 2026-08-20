"""
features.py
-----------
Turns the raw American bankruptcy dataset (columns X1..X18, verified against
accounting identities) into credit-risk features and a default label.

Target definition (Option 2 — "credit-clean"):
    A firm-year is a POSITIVE (label=1, "about to default") if the company
    eventually failed AND the year is within its last `HORIZON` years of data
    (i.e. the financials shortly before the bankruptcy filing).
    Alive companies contribute NEGATIVES (label=0). The earlier, ambiguous
    years of failed companies are dropped so we never train on a doomed
    company's healthy years.
"""
import numpy as np
import pandas as pd

# --- Verified column dictionary (confirmed via accounting identities) ---
COLUMN_MAP = {
    "X1": "current_assets",
    "X2": "cogs",
    "X3": "depreciation_amortization",
    "X4": "ebitda",
    "X5": "inventory",
    "X6": "net_income",
    "X7": "receivables",
    "X8": "market_value",
    "X9": "net_sales",
    "X10": "total_assets",
    "X11": "long_term_debt",
    "X12": "ebit",
    "X13": "gross_profit",
    "X14": "current_liabilities",
    "X15": "retained_earnings",
    "X16": "total_revenue",
    "X17": "total_liabilities",
    "X18": "operating_expenses",
}

HORIZON = 2  # years before failure that count as "about to default"


def _safe_div(numer, denom):
    """Divide, returning NaN where the denominator is 0 (avoids inf)."""
    out = numer / denom.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def build_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer credit ratios from the named accounting columns."""
    r = pd.DataFrame(index=df.index)

    # Leverage — how much debt vs. what they own / earn
    r["leverage"]         = _safe_div(df.total_liabilities, df.total_assets)
    r["lt_debt_ratio"]    = _safe_div(df.long_term_debt,    df.total_assets)
    r["debt_to_ebitda"]   = _safe_div(df.total_liabilities, df.ebitda)

    # Liquidity — can they cover short-term obligations
    r["current_ratio"]    = _safe_div(df.current_assets,               df.current_liabilities)
    r["quick_ratio"]      = _safe_div(df.current_assets - df.inventory, df.current_liabilities)
    r["working_capital_ta"] = _safe_div(df.current_assets - df.current_liabilities, df.total_assets)  # Altman

    # Profitability — are they making money
    r["roa"]              = _safe_div(df.net_income, df.total_assets)
    r["net_margin"]       = _safe_div(df.net_income, df.net_sales)
    r["ebit_margin"]      = _safe_div(df.ebit,       df.net_sales)
    r["retained_earn_ta"] = _safe_div(df.retained_earnings, df.total_assets)  # Altman
    r["ebit_ta"]          = _safe_div(df.ebit,              df.total_assets)  # Altman

    # Debt-servicing proxy (dataset has no interest expense column)
    r["ebit_to_liab"]     = _safe_div(df.ebit, df.total_liabilities)

    # Activity & market
    r["asset_turnover"]   = _safe_div(df.net_sales, df.total_assets)          # Altman
    r["mve_to_liab"]      = _safe_div(df.market_value, df.total_liabilities)  # Altman (market leverage)

    # Size
    r["log_assets"]       = np.log(df.total_assets.clip(lower=1e-3))

    return r


def build_dataset(path: str = "data/raw/american_bankruptcy.csv") -> pd.DataFrame:
    """Load raw data, rename columns, engineer ratios, build the Option-2 label."""
    raw = pd.read_csv(path).rename(columns=COLUMN_MAP)
    raw = raw.sort_values(["company_name", "year"]).reset_index(drop=True)

    failed = raw["status_label"].eq("failed")
    last_year = raw.groupby("company_name")["year"].transform("max")

    # Option 2 label
    about_to_fail = failed & (raw["year"] >= last_year - (HORIZON - 1))
    ambiguous     = failed & ~about_to_fail  # doomed company's earlier healthy years -> drop

    df = raw.loc[~ambiguous].copy()
    df["default"] = about_to_fail.loc[~ambiguous].astype(int)

    ratios = build_ratios(df)
    out = pd.concat(
        [df[["company_name", "year", "default"]].reset_index(drop=True),
         ratios.reset_index(drop=True)],
        axis=1,
    )
    return out


if __name__ == "__main__":
    data = build_dataset()
    print("Rows:", len(data), "| default rate: {:.2%}".format(data["default"].mean()))
    print(data.head())

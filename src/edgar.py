"""
edgar.py
--------
Pull a real company's latest annual financials from SEC EDGAR (free, official)
and shape them into the same columns the model was trained on.

Flow:  ticker  ->  CIK  ->  companyfacts JSON  ->  latest 10-K line items  ->  row
"""
import requests
import numpy as np
import pandas as pd

import features as F

# SEC requires a descriptive User-Agent with contact info.
HEADERS = {"User-Agent": "German Collado credit-risk-engine german.collado.blanco@gmail.com"}
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
# Free, no-API-key price quote so the repo runs out of the box.
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
BROWSER_UA = {"User-Agent": "Mozilla/5.0"}

# Map each column the model needs to the us-gaap tags EDGAR might report it under
# (tried in order; first one found wins). `duration=True` = income-statement item
# (needs a full-year window); `duration=False` = balance-sheet snapshot.
CONCEPTS = {
    "total_assets":        (["Assets"], False),
    "current_assets":      (["AssetsCurrent"], False),
    "total_liabilities":   (["Liabilities"], False),
    "current_liabilities": (["LiabilitiesCurrent"], False),
    "long_term_debt":      (["LongTermDebtNoncurrent", "LongTermDebt"], False),
    "inventory":           (["InventoryNet"], False),
    "retained_earnings":   (["RetainedEarningsAccumulatedDeficit"], False),
    "net_income":          (["NetIncomeLoss", "ProfitLoss"], True),
    "net_sales":           (["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                             "SalesRevenueNet"], True),
    "ebit":                (["OperatingIncomeLoss"], True),
    "depreciation_amortization": (["DepreciationDepletionAndAmortization",
                                   "DepreciationAmortizationAndAccretionNet",
                                   "DepreciationAndAmortization"], True),
    "cogs":                (["CostOfGoodsAndServicesSold", "CostOfRevenue"], True),
    "gross_profit":        (["GrossProfit"], True),
    "operating_expenses":  (["OperatingExpenses", "CostsAndExpenses"], True),
    "receivables":         (["AccountsReceivableNetCurrent"], False),
}

_ticker_map = None


def _load_ticker_map():
    global _ticker_map
    if _ticker_map is None:
        j = requests.get(TICKERS_URL, headers=HEADERS, timeout=30).json()
        _ticker_map = {v["ticker"].upper(): (v["cik_str"], v["title"]) for v in j.values()}
    return _ticker_map


def resolve(ticker: str):
    """Ticker -> (CIK, company name). Raises if unknown."""
    m = _load_ticker_map()
    t = ticker.strip().upper()
    if t not in m:
        raise ValueError(f"Ticker '{t}' not found in SEC EDGAR.")
    return m[t]


def _concept_by_year(facts, tags, duration):
    """Return {fiscal_year: value} for annual 10-K figures, merged across
    candidate tags (for each year, the entry with the latest reporting date)."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    by_year = {}  # fy -> (end_date, value)
    for tag in tags:
        if tag not in gaap or "USD" not in gaap[tag]["units"]:
            continue
        for r in gaap[tag]["units"]["USD"]:
            if r.get("form") not in ("10-K", "10-K/A") or r.get("fp") != "FY":
                continue
            if duration:  # income-statement item: keep full-year windows only
                if not r.get("start"):
                    continue
                days = (pd.Timestamp(r["end"]) - pd.Timestamp(r["start"])).days
                if not (300 <= days <= 400):
                    continue
            fy = r.get("fy")
            if fy is None:
                continue
            if fy not in by_year or r["end"] > by_year[fy][0]:
                by_year[fy] = (r["end"], r["val"])
    return {fy: v for fy, (_, v) in by_year.items()}


def get_price(ticker: str) -> float:
    """Current share price from Yahoo Finance (free, no key). NaN if unavailable."""
    try:
        j = requests.get(YAHOO_URL.format(ticker=ticker.upper()), headers=BROWSER_UA, timeout=20).json()
        return float(j["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except (KeyError, TypeError, ValueError, requests.RequestException):
        return np.nan


def _shares_outstanding(facts):
    """Latest reported common shares outstanding (from the 'dei' cover-page facts)."""
    dei = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {})
    rows = dei.get("units", {}).get("shares", [])
    if not rows:
        return np.nan
    return max(rows, key=lambda r: r.get("end", ""))["val"]


def fetch_financials(ticker: str) -> pd.DataFrame:
    """Return a one-row DataFrame (all model columns) for a real company,
    all line items taken from the SAME (latest available) fiscal year."""
    cik, name = resolve(ticker)
    facts = requests.get(FACTS_URL.format(cik=cik), headers=HEADERS, timeout=30).json()

    # Build a {year: value} series for every concept first.
    series = {col: _concept_by_year(facts, tags, dur) for col, (tags, dur) in CONCEPTS.items()}

    # Anchor on the latest year where Total Assets is reported (the newest 10-K).
    target_fy = max(series["total_assets"])

    def at_target(col):
        s = series[col]
        if target_fy in s:
            return s[target_fy]
        past = [fy for fy in s if fy <= target_fy]     # fall back to nearest prior year
        return s[max(past)] if past else np.nan

    row = {col: np.nan for col in F.COLUMN_MAP.values()}
    for col in CONCEPTS:
        row[col] = at_target(col)

    # Derived: EBITDA = EBIT + D&A (the identity we verified on the training data)
    if not np.isnan(row["ebit"]) and not np.isnan(row["depreciation_amortization"]):
        row["ebitda"] = row["ebit"] + row["depreciation_amortization"]
    row["total_revenue"] = row["net_sales"]

    # Market value = last price x shares outstanding (the Merton-style equity cushion)
    row["market_value"] = get_price(ticker) * _shares_outstanding(facts)

    df = pd.DataFrame([row])
    df.insert(0, "company_name", f"{name} ({ticker.upper()})")
    df.insert(1, "year", int(target_fy))
    return df


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    for tk in ["AAPL", "F"]:
        d = fetch_financials(tk)
        print(f"\n=== {d['company_name'].iloc[0]} ===")
        show = ["total_assets", "total_liabilities", "net_sales", "net_income", "ebit", "ebitda"]
        print(d[show].T.to_string(header=False, float_format=lambda x: f"{x:,.0f}"))

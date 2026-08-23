"""
scoring.py
----------
Turn the model's Probability of Default (PD) into the two things a credit
officer actually reads: a letter RATING and an Expected Credit Loss (ECL) in $.

    ECL = PD x LGD x EAD
      PD  = probability of default (from the model)
      LGD = loss given default   (fraction not recovered if they fail)
      EAD = exposure at default  (how much is on the line, i.e. the loan)
"""
import numpy as np
import pandas as pd
from joblib import load

import features as F

# --- Assumptions (documented so they are defensible, not hidden) ---
LGD = 0.45   # 45% loss given default: typical for senior unsecured corporate debt
EAD = 1_000_000  # $1M example loan, so ECL is expressed per $1M of exposure

# PD -> letter rating. Ordinal, agency-style (S&P/Moody's) scale.
# Thresholds are set for this model's 2-year horizon, from safest to riskiest.
RATING_BANDS = [
    (0.001, "AAA"),
    (0.0025, "AA"),
    (0.005, "A"),
    (0.010, "BBB"),   # investment-grade floor
    (0.030, "BB"),
    (0.070, "B"),
    (0.150, "CCC"),
    (0.300, "CC"),
    (1.001, "C"),
]


def pd_to_rating(pd_value: float) -> str:
    """Map a probability of default to its letter grade."""
    for upper, letter in RATING_BANDS:
        if pd_value < upper:
            return letter
    return "C"


def score(df_raw: pd.DataFrame, model_path="models/pd_model.joblib") -> pd.DataFrame:
    """Given raw company rows (X1..X18 renamed), return PD, rating and ECL."""
    bundle = load(model_path)
    pipe, feats = bundle["pipe"], bundle["features"]

    ratios = F.build_ratios(df_raw)                 # same feature recipe as training
    pd_hat = pipe.predict_proba(ratios[feats])[:, 1]

    out = df_raw[["company_name", "year"]].copy()
    out["PD"] = pd_hat
    out["rating"] = [pd_to_rating(p) for p in pd_hat]
    out["ECL_$"] = (pd_hat * LGD * EAD).round(0)    # expected loss on a $1M loan
    return out


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")

    # Demo: score the most recent year of a spread of real (anonymized) companies.
    raw = pd.read_csv("data/raw/american_bankruptcy.csv").rename(columns=F.COLUMN_MAP)
    latest = raw.sort_values("year").groupby("company_name").tail(1)

    scored = score(latest)
    scored["actually_failed"] = latest["status_label"].values == "failed"

    # Show a spread: safest, middle, riskiest
    s = scored.sort_values("PD")
    print("=== Safest 3 ===")
    print(s.head(3).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Riskiest 3 ===")
    print(s.tail(3).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Rating distribution across all companies ===")
    print(scored["rating"].value_counts().reindex(
        [b[1] for b in RATING_BANDS]).fillna(0).astype(int).to_string())

    # Sanity: do worse ratings actually fail more often?
    print("\n=== Real default rate by predicted rating (should rise A -> C) ===")
    chk = scored.assign(failed=scored["actually_failed"]).groupby("rating")["failed"].mean()
    print(chk.reindex([b[1] for b in RATING_BANDS]).dropna().to_string(float_format=lambda x: f"{x:.1%}"))

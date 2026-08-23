"""
model.py
--------
Train an interpretable Probability-of-Default (PD) model, validated OUT-OF-TIME.

Pipeline of ideas:
  1. Engineer ratios and the default label (from features.py).
  2. Split by time: train on the past, test on the future.
  3. Remove multicollinearity with VIF, so every coefficient has a sensible sign.
  4. Fit a logistic regression (explainable to a regulator).
  5. Judge it with Gini/AUC (discrimination) and calibration (are the PDs real?).
"""
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import features as F
from features import QuantileClipper  # shared, so the saved model re-loads anywhere


def gini(y, p):
    # Gini is just a rescaled AUC: 0 = random, 1 = perfect ranking.
    return 2 * roc_auc_score(y, p) - 1


def select_features_by_vif(X, threshold=10.0):
    """
    Drop multicollinear ratios one at a time.
    VIF (Variance Inflation Factor) measures how well a feature can be predicted
    from the OTHER features. High VIF = redundant twin -> unstable, flipped signs.
    We iteratively drop the worst offender until every VIF is below `threshold`.
    """
    keep = list(X.columns)
    while True:
        # Compute VIF for each surviving feature: VIF = 1 / (1 - R^2)
        vifs = {}
        for col in keep:
            others = X[keep].drop(columns=col)
            r2 = LinearRegression().fit(others, X[col]).score(others, X[col])
            vifs[col] = np.inf if r2 >= 1 else 1.0 / (1.0 - r2)
        worst = max(vifs, key=vifs.get)
        # Stop once the highest VIF is acceptable
        if vifs[worst] < threshold:
            return keep, vifs
        # Otherwise drop the most redundant feature and repeat
        keep.remove(worst)


def main():
    # --- 1. Build the modeling table (ratios + default label) ---
    data = F.build_dataset()
    all_feats = [c for c in data.columns if c not in ("company_name", "year", "default")]

    # --- 2. Out-of-time split: learn on old years, test on new ones ---
    tr = data[data.year <= 2011]
    va = data[(data.year >= 2012) & (data.year <= 2014)]
    te = data[data.year >= 2015]

    # --- 3. Feature selection: kill multicollinearity with VIF ---
    # Impute + scale the TRAIN features first so VIF is computed on clean numbers.
    prep = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clip",   QuantileClipper(0.01, 0.99)),
                     ("scale",  StandardScaler())])
    X_tr_clean = pd.DataFrame(prep.fit_transform(tr[all_feats]), columns=all_feats)
    # Threshold 5 is the stricter, textbook cutoff — leaves only truly independent ratios.
    feats, vifs = select_features_by_vif(X_tr_clean, threshold=5.0)
    print("Kept features:", feats)
    print("Dropped for multicollinearity:", sorted(set(all_feats) - set(feats)))

    # --- 4. Final model on the surviving features ---
    # No class re-weighting: keep the PDs calibrated to the true ~1.6% base rate.
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip",   QuantileClipper(0.01, 0.99)),
        ("scale",  StandardScaler()),
        ("clf",    LogisticRegression(max_iter=2000)),
    ])
    pipe.fit(tr[feats], tr["default"])

    # --- 5a. Discrimination: does it rank risky firms above safe ones? ---
    print("\n=== Discrimination (AUC / Gini) — out-of-time ===")
    for name, part in [("train 99-11", tr), ("val 12-14", va), ("test 15-18", te)]:
        p = pipe.predict_proba(part[feats])[:, 1]
        print(f"  {name:12s}  AUC={roc_auc_score(part['default'], p):.3f}  Gini={gini(part['default'], p):.3f}")

    # --- 5b. Calibration: when it says 5%, do ~5% actually default? ---
    p_te = pipe.predict_proba(te[feats])[:, 1]
    print(f"\n=== Calibration (test) ===  Brier={brier_score_loss(te['default'], p_te):.4f}")
    dec = pd.DataFrame({"pd": p_te, "y": te["default"].values})
    dec["bucket"] = pd.qcut(dec["pd"], 10, labels=False, duplicates="drop")
    tab = dec.groupby("bucket").agg(pred_PD=("pd", "mean"), actual_rate=("y", "mean"), n=("y", "size"))
    print(tab.to_string(float_format=lambda x: f"{x:.3f}"))

    # --- 5c. Interpretability: which ratios drive the score, and in which direction? ---
    coef = pd.Series(pipe.named_steps["clf"].coef_[0], index=feats).sort_values()
    print("\n=== Standardized coefficients (sign = direction of risk) ===")
    print(coef.to_string(float_format=lambda x: f"{x:+.3f}"))

    # --- 6. Persist the trained pipeline for scoring later ---
    dump({"pipe": pipe, "features": feats}, "models/pd_model.joblib")
    print("\nSaved -> models/pd_model.joblib")


if __name__ == "__main__":
    import sys; sys.path.insert(0, "src")
    main()

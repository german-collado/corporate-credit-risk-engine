# Corporate Credit Risk Engine

**Estimating expected credit loss the way a bank does under CECL / IFRS 9.**

A probability-of-default model on US corporate fundamentals → implied credit rating → expected credit loss (`ECL = PD × LGD × EAD`), with a macro-scenario overlay built on Federal Reserve data.

> ⚠️ Work in progress. This README grows as each stage lands.

---

## Why this project

Credit-loss forecasting is the core of a bank's risk function. This engine reproduces that workflow end to end on public data:

1. **PD** — a model that reads a company's financial ratios and estimates its probability of default over a 2-year horizon.
2. **Rating** — the PD is mapped to an agency-style letter grade (AAA … D).
3. **ECL** — `PD × LGD × EAD` turns that into an expected dollar loss.
4. **Macro overlay** — a recession scenario shifts the PDs using the relationship between macro conditions and defaults (extends the [Fed Dual Mandate Dashboard](https://github.com/german-collado/fed-dual-mandate-dashboard)).

## Data

[American Companies Bankruptcy dataset](https://github.com/sowide/bankruptcy_dataset) — 8,971 US public companies (NYSE / NASDAQ), 1999–2018, with real Chapter 11 / Chapter 7 outcomes. 609 companies failed.

The raw columns arrive anonymized (`X1…X18`); their meaning was **recovered and verified against accounting identities** (e.g. `EBITDA = EBIT + D&A` holds in 100% of rows) before any modeling — see [`src/features.py`](src/features.py).

**Target (credit-clean framing):** a firm-year is labeled *default* if the company later failed and the year falls within its last 2 years of data — i.e. the financials shortly before the filing. Doomed companies' earlier healthy years are dropped, so the model never learns from mislabeled data.

## Approach

- **Features:** leverage, liquidity, profitability and activity ratios — including the five Altman Z-score inputs — engineered from the raw fundamentals.
- **Validation:** out-of-time (train 1999–2011 · validate 2012–2014 · test 2015–2018), because a credit model must work on the future, not a random shuffle of the past.
- **Metrics:** Gini / AUC and calibration, not accuracy — defaults are a ~1.6% rare event.
- **Interpretability first:** in regulated credit you must justify every decision.

## Structure

```
data/raw/     raw dataset
src/          feature engineering, model, scoring
notebooks/    exploration
models/        trained artifacts
reports/       figures & metrics
```

## Setup

```bash
pip install -r requirements.txt
python src/features.py   # builds the modeling dataset
```

---

<sub>German Collado · credit & market risk analytics</sub>
